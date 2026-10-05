import dataclasses
import numpy as np
import jax.random as jr
import jax
import copy
import optax
import jax.numpy as jnp

from dataclasses import dataclass
from flax import struct
from cyclopts import Parameter
from jaxtyping import Array
from typing import Annotated

from .module.value import Value
from .module.actor import ActorVectorField, StdTanhNormalPolicy
from .module.utils import ModuleDict, TrainState
from .module.temperature import Temperature
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..utils.typing import Obs, Action, PRNGKey, Params


@Parameter(name="*", group="AgentConfig")
@dataclass
class DSRLCfg:
    """Configuration for the DSRL agent.

    Parameters
    ----------
    lr : float
        Learning rate for all networks.
    max_grad_norm : float
        Maximum global gradient norm for each optimizer update.
    actor_hidden_dims : list[int]
        Hidden layer widths for the BC flow and latent noise policy.
    value_hidden_dims : list[int]
        Hidden layer widths for each critic network.
    actor_layer_norm : bool
        Use layer normalization in the actor networks.
    value_layer_norm : bool
        Use layer normalization in the critic networks.
    n_critic_ensembles : int
        Number of Q-functions in the critic ensemble; must be a positive integer.
    gamma : float
        Discount factor for returns and TD targets.
    tau : float
        Target critic update rate (weight on the online critic).
    pessimism_coef : float
        Penalty in mean(Q) - pessimism_coef * std(Q) for TD targets. Zero uses mean Q.
    horizon_length : int
        Number of steps in TD targets and, when enabled, action chunks; must be a positive integer.
    action_chunking : bool
        Predict and execute horizon_length actions per policy sample.
    noise_bound : float
        Absolute bound on each coordinate sampled by the latent noise policy; must be positive and finite.
    action_flow_steps : int
        Number of Euler steps for decoding latent noise into actions with the BC flow; must be a positive integer.
    target_entropy : float | None
        Noise policy target entropy. None uses -0.5 times the full latent dimension, including action chunks.
    action_dim : int | None
        Dimension of a single action; set automatically during agent creation.
    """
    # Optimization.
    lr: float = 3e-4
    max_grad_norm: float = 10.0

    # Network architecture.
    actor_hidden_dims: Annotated[list[int], Parameter(consume_multiple=True)] = (512, 512, 512, 512)
    value_hidden_dims: Annotated[list[int], Parameter(consume_multiple=True)] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    value_layer_norm: bool = True
    n_critic_ensembles: int = 8

    # Critic learning and ensemble objectives.
    gamma: float = 0.995
    tau: float = 0.005
    pessimism_coef: float = 0.5

    # Actor.
    horizon_length: int = 1
    action_chunking: bool = False
    noise_bound: float = 1.0
    action_flow_steps: int = 10
    target_entropy: float | None = None

    # Environment-derived values (set by DSRLAgent.create).
    action_dim: int | None = None

    def __post_init__(self):
        for name in ("horizon_length", "action_flow_steps", "n_critic_ensembles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer (got {value!r}).")
        if (isinstance(self.noise_bound, (bool, np.bool_)) or
                not np.isfinite(self.noise_bound) or self.noise_bound <= 0):
            raise ValueError(f"noise_bound must be a positive finite number (got {self.noise_bound!r}).")

    @property
    def agent_name(self):
        return "dsrl"


class DSRLAgent(struct.PyTreeNode):
    """Flow-based DSRL agent."""
    network: TrainState
    cfg: DSRLCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: DSRLCfg,
    ):
        key = jr.PRNGKey(seed)
        key, init_key = jr.split(key, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        if cfg.action_chunking:
            full_actions = jnp.concatenate([ex_actions] * cfg.horizon_length, axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define networks.
        critic_def = Value(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.value_layer_norm,
            num_ensembles=cfg.n_critic_ensembles,
        )
        bc_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        actor_noise_generator_def = StdTanhNormalPolicy(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
            low=-cfg.noise_bound,
            high=cfg.noise_bound,
        )
        temperature_def = Temperature()
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            bc_flow=(bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_noise_generator=(actor_noise_generator_def, (ex_observations,)),
            temperature=(temperature_def, ()),
        )

        # Initialize networks and optimizers.
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=cfg.lr)
        network_params = network_def.init(init_key, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        # Update cfg with environment information.
        cfg = dataclasses.replace(
            cfg,
            action_dim=action_dim,
            target_entropy=-full_action_dim / 2 if cfg.target_entropy is None else cfg.target_entropy,
        )

        return cls(network=network, cfg=cfg)

    @jax.jit
    def update(self, batch: dict, key: PRNGKey):

        def loss_fn_(params):
            total_loss_, info_ = self.total_loss(batch, params, key)
            return total_loss_, info_

        grad, info = jax.grad(loss_fn_, has_aux=True)(self.network.params)
        grad_ill = has_any_nan_or_inf(grad)
        grad, grad_norm = compute_norm_and_clip(grad, self.cfg.max_grad_norm)
        new_network = self.network.apply_gradients(grads=grad)

        # Update target network.
        self.target_update(new_network, "critic")

        return self.replace(network=new_network), info | {
            "total/grad_norm": grad_norm,
            "total/grad_ill": grad_ill,
        }

    def target_update(self, network: TrainState, module_name: str):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.cfg.tau + tp * (1 - self.cfg.tau),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def total_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Combine critic, BC flow, latent-policy, and temperature losses."""
        info = {}
        key, actor_key, critic_key = jr.split(key, 3)

        # Critic loss.
        critic_loss, critic_info = self.critic_loss(batch, params, critic_key)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Actor loss.
        actor_loss, actor_info = self.actor_loss(batch, params, actor_key)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        # Temperature loss.
        temp_loss, temp_info = self.temperature_loss(actor_info['entropy'], params)
        for k, v in temp_info.items():
            info[f'temperature/{k}'] = v

        loss = critic_loss + actor_loss + temp_loss

        return loss, info

    def critic_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Critic TD loss."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        key, sample_key = jr.split(key)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], key=sample_key)

        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        next_q = next_qs.mean(axis=0) if self.cfg.n_critic_ensembles > 1 else next_qs
        if self.cfg.n_critic_ensembles > 1 and self.cfg.pessimism_coef != 0.0:
            next_q = next_q - self.cfg.pessimism_coef * next_qs.std(axis=0)

        target_q = (batch['rewards'][..., -1] +
                    (self.cfg.gamma ** self.cfg.horizon_length) * batch['masks'][..., -1] * next_q)

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q_ill': has_any_nan_or_inf(q),
            'target_q_ill': has_any_nan_or_inf(target_q),
        }

    def actor_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Train the BC flow and the latent policy with an entropy-regularized Q objective."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        key, x_key, t_key = jr.split(key, 3)

        # Straight paths from Gaussian noise to dataset actions.
        x_0 = jr.normal(x_key, (batch_size, action_dim))
        x_1 = batch_actions
        t = jr.uniform(t_key, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('bc_flow')(batch['observations'], x_t, t, params=params)
        bc_flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])

        # Train the latent policy through the fixed BC flow and critic.
        key, noise_key = jr.split(key)
        noise_dist = self.network.select('actor_noise_generator')(batch['observations'], params=params)
        noise, log_probs = noise_dist.sample_and_log_prob(seed=noise_key)
        actions = self.latent2action(batch['observations'], noise)
        actions = jnp.clip(actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        q = qs.mean(axis=0)
        temp = self.network.select('temperature')()
        noise_loss = (temp * log_probs - q).mean()

        # Total loss.
        actor_loss = noise_loss + bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'noise_loss': noise_loss,
            'q': q.mean(),
            'entropy': -log_probs.mean(),
            'actor_loss_ill': has_any_nan_or_inf(actor_loss),
        }

    def temperature_loss(self, entropy: Array, params: Params):
        """Adjust the entropy temperature toward the target entropy."""
        temp = self.network.select('temperature')(params=params)
        temp_loss = temp * jax.lax.stop_gradient(entropy - self.cfg.target_entropy).mean()
        return temp_loss, {
            'temperature_loss': temp_loss,
            'temperature': temp,
        }

    @jax.jit
    def latent2action(self, obs: Obs, latents: Array):
        """Decode latents with the BC flow using Euler integration."""
        actions = latents

        def euler_step_(i_, actions_):
            t = jnp.full((*obs.shape[:-1], 1), i_ / self.cfg.action_flow_steps)
            vels = self.network.select('bc_flow')(obs, actions_, t)
            actions_ = actions_ + vels / self.cfg.action_flow_steps
            return actions_

        actions = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, actions)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        """Sample the latent policy and decode it with the BC flow."""
        noise_dist = self.network.select('actor_noise_generator')(obs)
        noise, _ = noise_dist.sample_and_log_prob(seed=key)
        actions = self.latent2action(obs, noise)
        actions = jnp.clip(actions, -1, 1)
        return actions
