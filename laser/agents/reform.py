import dataclasses
import numpy as np
import jax.random as jr
import jax
import copy
import optax
import jax.numpy as jnp
import jax.tree_util as jtu

from dataclasses import dataclass
from flax import struct
from cyclopts import Parameter
from optax.tree_utils import tree_where
from typing import Annotated

from .module.value import Value
from .module.actor import ActorVectorField
from .module.utils import ModuleDict, TrainState
from .utils import sample_uniform_in_hypersphere
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..utils.typing import Obs, Action, PRNGKey, Params


@Parameter(name="*", group="AgentConfig")
@dataclass
class ReFORMCfg:
    """Configuration for the ReFORM agent.

    Parameters
    ----------
    lr : float
        Learning rate for all networks.
    max_grad_norm : float
        Maximum global gradient norm for each optimizer update.
    actor_hidden_dims : list[int]
        Hidden layer widths for the BC flow, one-step decoder, and latent flow.
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
    normalize_q_loss : bool
        Scale the actor Q loss by the inverse mean absolute Q-value.
    q_loss_coef : float
        Weight of the Q-maximization loss for the latent flow.
    action_flow_steps : int
        Number of Euler steps for the BC decoder and latent flow; must be a positive integer.
    action_dim : int | None
        Dimension of a single action; set automatically during agent creation.
    obs_dims : tuple[int, ...] | None
        Observation shape without the batch dimension; set automatically during agent creation.
    r_max : float | None
        Latent ball radius; set to the square root of the full action dimension during agent creation.
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
    normalize_q_loss: bool = False
    q_loss_coef: float = 0.1
    action_flow_steps: int = 10

    # Environment-derived values (set by ReFORMAgent.create).
    action_dim: int | None = None
    obs_dims: tuple[int, ...] | None = None
    r_max: float | None = None

    def __post_init__(self):
        for name in ("horizon_length", "action_flow_steps", "n_critic_ensembles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer (got {value!r}).")

    @property
    def agent_name(self):
        return "reform"


class ReFORMAgent(struct.PyTreeNode):
    network: TrainState
    cfg: ReFORMCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: ReFORMCfg,
    ):
        key = jr.PRNGKey(seed)
        key, init_key = jr.split(key, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        obs_dims = ex_observations.shape[1:]

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
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        actor_noise_generator_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            bc_flow=(bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            actor_noise_generator=(actor_noise_generator_def, (ex_observations, full_actions, ex_times)),
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
        r_max = np.sqrt(full_action_dim)
        cfg = dataclasses.replace(
            cfg,
            action_dim=action_dim,
            obs_dims=obs_dims,
            r_max=r_max,
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
        grad = tree_where(grad_ill, jtu.tree_map(jnp.zeros_like, grad), grad)
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
        """Combine critic, BC flow, distillation, and latent-policy losses."""
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

        loss = critic_loss + actor_loss
        return loss, info

    def critic_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Critic TD loss."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        key, sample_key = jr.split(key)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], key=sample_key)
        next_actions = jnp.clip(next_actions, -1, 1)

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
        """Train the BC flow, one-step decoder, and Q-guided latent flow."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        key, x_key, t_key = jr.split(key, 3)

        # Straight paths from uniform latents to dataset actions.
        x_0 = sample_uniform_in_hypersphere(x_key, self.cfg.r_max, (batch_size, action_dim))
        x_1 = batch_actions
        t = jr.uniform(t_key, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('bc_flow')(batch['observations'], x_t, t, params=params)
        bc_flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])

        # Distill the BC flow into the one-step decoder.
        key, noise_key = jr.split(key)
        latents = sample_uniform_in_hypersphere(noise_key, self.cfg.r_max, (batch_size, action_dim))
        target_flow_actions = self.latent2action(batch['observations'], latents=latents)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], latents, params=params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Improve the latent flow through the fixed decoder and critic.
        key, noise_key = jr.split(key)
        base_samples = sample_uniform_in_hypersphere(noise_key, self.cfg.r_max, (batch_size, action_dim))
        latents = self.base2latent(batch['observations'], base_samples=base_samples, params=params)
        targeted_actions = self.network.select('actor_onestep_flow')(batch['observations'], latents)
        targeted_actions = jnp.clip(targeted_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=targeted_actions)
        q = qs.mean(axis=0) if self.cfg.n_critic_ensembles > 1 else qs
        q_loss = -q.mean()
        if self.cfg.normalize_q_loss:
            lam = jax.lax.stop_gradient(1 / (jnp.abs(q).mean() + 1e-5))
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + distill_loss + q_loss * self.cfg.q_loss_coef

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'actor_loss_ill': has_any_nan_or_inf(actor_loss),
        }

    @jax.jit
    def latent2action(self, obs: Obs, latents: jnp.ndarray):
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

    def base2latent(self, obs: Obs, base_samples: jnp.ndarray, params: Params = None):
        """Integrate the latent policy with boundary reflection."""
        latents = base_samples

        def euler_step_(i_, carry_):
            latents_ = carry_
            t = jnp.full((*obs.shape[:-1], 1), i_ / self.cfg.action_flow_steps)
            vels = self.network.select('actor_noise_generator')(obs, latents_, t, params=params)
            latent_step = vels / self.cfg.action_flow_steps
            latents_ = latents_ + latent_step

            # Reflect latents that go out of bounds.
            norms = jnp.linalg.norm(latents_, axis=-1, keepdims=True)
            out_bound = norms > self.cfg.r_max
            reflect_dir = -latents_ / (norms + 1e-8)
            latent_step_proj = jnp.sum(latent_step * reflect_dir, axis=-1, keepdims=True)

            latents_ = jnp.where(
                out_bound,
                latents_ - latent_step_proj * reflect_dir,
                latents_,
            )
            return latents_

        latents = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, latents)
        return latents

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        """Sample the latent flow and decode it with the one-step policy."""
        action_key, noise_key = jr.split(key)
        action_dim = self.cfg.action_dim * (self.cfg.horizon_length if self.cfg.action_chunking else 1)
        base_samples = sample_uniform_in_hypersphere(
            noise_key, self.cfg.r_max, (
                *obs.shape[: -len(self.cfg.obs_dims)],
                action_dim,
            )
        )
        latents = self.base2latent(obs, base_samples)
        actions = self.network.select('actor_onestep_flow')(obs, latents)
        actions = jnp.clip(actions, -1, 1)
        return actions
