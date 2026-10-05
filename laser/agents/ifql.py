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
from typing import Annotated

from .module.value import Value
from .module.actor import ActorVectorField
from .module.utils import ModuleDict, TrainState
from .utils import expectile_loss
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..utils.typing import Obs, Action, PRNGKey, Params


@Parameter(name="*", group="AgentConfig")
@dataclass
class IFQLCfg:
    """Configuration for the implicit flow Q-learning (IFQL) agent.

    Parameters
    ----------
    lr : float
        Learning rate for all networks.
    max_grad_norm : float
        Maximum global gradient norm for each optimizer update.
    actor_hidden_dims : list[int]
        Hidden layer widths for the BC flow policy.
    value_hidden_dims : list[int]
        Hidden layer widths for the state-value and critic networks.
    actor_layer_norm : bool
        Use layer normalization in the actor networks.
    value_layer_norm : bool
        Use layer normalization in the state-value and critic networks.
    n_critic_ensembles : int
        Number of Q-functions in the critic ensemble; must be a positive integer.
    gamma : float
        Discount factor for returns and TD targets.
    tau : float
        Target critic update rate (weight on the online critic).
    expectile : float
        Expectile level for fitting the state value to target critic values.
    pessimism_coef : float
        Penalty in mean(Q) - pessimism_coef * std(Q) for value fitting. Zero uses mean Q.
    horizon_length : int
        Number of steps in TD targets and, when enabled, action chunks; must be a positive integer.
    action_chunking : bool
        Predict and execute horizon_length actions per policy sample.
    num_action_samples : int
        Number of flow candidates to sample before selecting the one with the highest minimum Q-value.
        Must be a positive integer.
    action_flow_steps : int
        Number of Euler steps for generating each candidate action with the BC flow; must be a positive integer.
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

    # Critic and value learning.
    gamma: float = 0.995
    tau: float = 0.005
    expectile: float = 0.9
    pessimism_coef: float = 0.5

    # Actor.
    horizon_length: int = 1
    action_chunking: bool = False
    num_action_samples: int = 32
    action_flow_steps: int = 10

    # Environment-derived values (set by IFQLAgent.create).
    action_dim: int | None = None

    def __post_init__(self):
        for name in ("horizon_length", "action_flow_steps", "n_critic_ensembles", "num_action_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer (got {value!r}).")

    @property
    def agent_name(self):
        return "ifql"


class IFQLAgent(struct.PyTreeNode):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """
    network: TrainState
    cfg: IFQLCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: IFQLCfg,
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
        value_def = Value(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.value_layer_norm,
            num_ensembles=1,
        )
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
        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            bc_flow=(bc_flow_def, (ex_observations, full_actions, ex_times)),
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
        cfg = dataclasses.replace(cfg, action_dim=action_dim)

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
        """Combine value, critic, and BC flow-matching losses."""
        info = {}

        # Value loss.
        value_loss, value_info = self.value_loss(batch, params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        # Critic loss.
        critic_loss, critic_info = self.critic_loss(batch, params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Actor loss.
        key, actor_key = jr.split(key)
        actor_loss, actor_info = self.flow_matching_loss(batch, params, actor_key)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def value_loss(self, batch: dict, params: Params):
        """IQL value loss."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        qs = self.network.select('target_critic')(batch['observations'], actions=batch_actions)
        q = qs.mean(axis=0) if self.cfg.n_critic_ensembles > 1 else qs
        if self.cfg.n_critic_ensembles > 1 and self.cfg.pessimism_coef != 0.0:
            q = q - self.cfg.pessimism_coef * qs.std(axis=0)
        v = self.network.select('value')(batch['observations'], params=params)
        value_loss = expectile_loss(q - v, q - v, self.cfg.expectile).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'v_ill': has_any_nan_or_inf(v),
        }

    def critic_loss(self, batch: dict, params: Params):
        """IQL critic loss."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        next_v = self.network.select('value')(batch['next_observations'][..., -1, :])
        target_q = (batch['rewards'][..., -1] +
                    (self.cfg.gamma ** self.cfg.horizon_length) * batch['masks'][..., -1] * next_v)

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

    def flow_matching_loss(self, batch: dict, params: Params, key: PRNGKey):
        """BC flow-matching actor loss."""
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
        actor_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])

        return actor_loss, {
            'actor_loss': actor_loss,
            'actor_loss_ill': has_any_nan_or_inf(actor_loss),
        }

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        """Decode flow candidates and select the best one with the critic ensemble."""
        action_key, noise_key = jr.split(key)
        action_dim = self.cfg.action_dim * (self.cfg.horizon_length if self.cfg.action_chunking else 1)

        # Decode num_action_samples candidates for each observation.
        actions = jr.normal(
            action_key,
            (
                self.cfg.num_action_samples,
                *obs.shape[:-1],
                action_dim,
            ),
        )
        n_obs = jnp.repeat(jnp.expand_dims(obs, 0), self.cfg.num_action_samples, axis=0)

        def euler_step_(i_, actions_):
            t = jnp.full((self.cfg.num_action_samples, *obs.shape[:-1], 1), i_ / self.cfg.action_flow_steps)
            vels = self.network.select('bc_flow')(n_obs, actions_, t)
            actions_ = actions_ + vels / self.cfg.action_flow_steps
            return actions_

        actions = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, actions)
        actions = jnp.clip(actions, -1, 1)

        # Select the candidate with the highest minimum ensemble Q-value.
        qs = self.network.select('critic')(n_obs, actions=actions)
        q = qs.min(axis=0) if self.cfg.n_critic_ensembles > 1 else qs
        best_indices = jnp.argmax(q, axis=0)
        actions = jnp.take_along_axis(actions, best_indices[None, ..., None], axis=0).squeeze(axis=0)
        return actions
