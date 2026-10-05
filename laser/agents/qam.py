import dataclasses
import numpy as np
import jax.random as jr
import jax
import copy
import optax
import jax.numpy as jnp
import functools as ft

from dataclasses import dataclass
from flax import struct
from cyclopts import Parameter
from typing import Annotated

from .module.value import Value
from .module.actor import ActorVectorField
from .module.utils import ModuleDict, TrainState
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip, clip_top_k_percent_grads
from ..utils.typing import Obs, Action, PRNGKey, Params


@Parameter(name="*", group="AgentConfig")
@dataclass
class QAMCfg:
    """Configuration for the QAM agent.

    Parameters
    ----------
    lr : float
        Learning rate for all networks.
    max_grad_norm : float
        Maximum global gradient norm for each optimizer update.
    actor_hidden_dims : list[int]
        Hidden layer widths for the BC and Q-guided flow networks.
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
        Target critic and BC flow update rate (weight on the corresponding online network).
    pessimism_coef : float
        Penalty in mean(Q) - pessimism_coef * std(Q) for TD targets. Zero uses mean Q.
    horizon_length : int
        Number of steps in TD targets and, when enabled, action chunks; must be a positive integer.
    action_chunking : bool
        Predict and execute horizon_length actions per policy sample.
    action_flow_steps : int
        Number of integration steps for action sampling and adjoint matching; must be a positive integer.
    inv_temp : float
        Inverse temperature scaling the terminal Q-gradient in adjoint matching.
        Use the BC flow for action sampling when zero.
    target_actor : bool
        Use the target BC flow as the reference flow for adjoint matching.
    use_target_grad : bool
        Use the target critic to compute terminal Q-gradients for adjoint matching.
    clip_q_grad : bool
        Cap terminal action Q-gradient norms at the batch's 90th percentile.
    action_dim : int | None
        Dimension of a single action; set automatically during agent creation.
    obs_dims : tuple[int, ...] | None
        Observation shape without the batch dimension; set automatically during agent creation.
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
    action_flow_steps: int = 10
    inv_temp: float = 10.0
    target_actor: bool = True
    use_target_grad: bool = True
    clip_q_grad: bool = True

    # Environment-derived values (set by QAMAgent.create).
    action_dim: int | None = None
    obs_dims: tuple[int, ...] | None = None

    def __post_init__(self):
        for name in ("horizon_length", "action_flow_steps", "n_critic_ensembles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer (got {value!r}).")

    @property
    def agent_name(self):
        return "qam"


class QAMAgent(struct.PyTreeNode):
    network: TrainState
    cfg: QAMCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: QAMCfg,
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
        actor_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            guided_flow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            bc_flow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_bc_flow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
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
        params['modules_target_bc_flow'] = params['modules_bc_flow']

        # Update cfg with environment information.
        cfg = dataclasses.replace(
            cfg,
            action_dim=action_dim,
            obs_dims=obs_dims,
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

        # Update target networks.
        self.target_update(new_network, "critic")
        self.target_update(new_network, 'bc_flow')

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
        """Combine critic, BC flow, and adjoint-matching losses."""
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
        """Train the reference flow by flow matching and the guided flow by adjoint matching."""
        if self.cfg.action_chunking:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        key, x_key, t_key, _, adj_key = jr.split(key, 5)

        # Straight paths from Gaussian noise to dataset actions.
        x_0 = jr.normal(x_key, (batch_size, action_dim))
        x_1 = batch_actions

        t = jr.uniform(t_key, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('bc_flow')(batch['observations'], x_t, t, params=params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss

        info = {}
        total_guided_loss = 0
        bc_flow = self.network.select("target_bc_flow" if self.cfg.target_actor else "bc_flow")

        # Sample trajectories and construct fixed adjoint targets.
        xs, adjs, ts, pre_adj_info = self.compute_adjoint_targets(batch["observations"], adj_key)
        h = 1 / self.cfg.action_flow_steps
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(batch["observations"][None], self.cfg.action_flow_steps, axis=0)
        vf_fine = self.network.select("guided_flow")(observations, xs, ts, params=params)
        vf_base = bc_flow(observations, xs, ts)

        # Fit the guided flow to the fixed reference field and adjoint targets.
        adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)
        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info["adj_loss"] = adj_loss
        info.update(pre_adj_info)
        total_guided_loss += adj_loss

        return actor_loss + total_guided_loss, {'flow_loss': flow_loss, "adj_loss": adj_loss, **info}

    @ft.partial(jax.jit, static_argnames=("flow_steps",))
    def compute_adjoint_targets(self, obs, key, flow_steps=None):
        """Sample action trajectories and compute their lean-adjoint targets."""
        flow_steps = self.cfg.action_flow_steps if flow_steps is None else flow_steps

        action_dim = self.cfg.action_dim * (self.cfg.horizon_length if self.cfg.action_chunking else 1)
        x0_key, key = jr.split(key, 2)
        x = jr.normal(x0_key, shape=obs.shape[:-1] + (action_dim,))

        bc_flow = self.network.select("target_bc_flow" if self.cfg.target_actor else "bc_flow")

        # Sample a training trajectory using the regularized SDE.
        h = 1 / flow_steps
        xs = [x]
        ts = []
        noise_key, key = jr.split(key)
        for i, key in zip(range(flow_steps), jr.split(noise_key, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            # Shift endpoint coefficients by h to avoid the t=0 singularity.
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jr.normal(key, x.shape)
            if i != flow_steps - 1:
                v = self.network.select("guided_flow")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:
                # Use a noiseless reference-flow step at the endpoint.
                x = x + h * bc_flow(obs, x, t)

            xs.append(x)
            ts.append(t)

        # Terminal adjoint = -inv_temp * grad_a Q(s, a).
        critic_network = "target_critic" if self.cfg.use_target_grad else "critic"
        grad_fn = jax.grad(
            lambda x, y: self.network.select(critic_network)(
                x, jnp.clip(y, -1., 1.)).sum() / self.cfg.n_critic_ensembles, 1)

        if self.cfg.clip_q_grad:
            grad = grad_fn(obs, xs[-1])
            # Cap terminal action gradient norms at the 90th percentile.
            grad = clip_top_k_percent_grads(grad, k=10.0)
        else:
            grad = grad_fn(obs, xs[-1])

        adj = -grad * self.cfg.inv_temp
        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
        }

        # Integrate the lean adjoint backward via the reference drift VJP.
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * bc_flow(obs, xi, t + h) - xi / (t + h)

            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            adjs.append(adj)

        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    @ft.partial(jax.jit, static_argnames="model")
    def latent2action(self, obs: Obs, latents: jnp.ndarray, model="bc"):
        """Integrate the selected action flows with Euler steps."""
        actions = latents
        networks = [self.network.select(f'{m}_flow') for m in model.split(",")]

        def euler_step_(i_, actions_):
            t = jnp.full((*obs.shape[:-1], 1), i_ / self.cfg.action_flow_steps)
            vels = sum([network(obs, actions_, t) for network in networks])
            actions_ = actions_ + vels / self.cfg.action_flow_steps
            return actions_

        actions = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, actions)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        """Sample the guided action flow, or the BC flow when inv_temp is zero."""
        action_dim = self.cfg.action_dim * (self.cfg.horizon_length if self.cfg.action_chunking else 1)
        action_key, noise_key = jr.split(key)
        latents = jr.normal(noise_key, shape=obs.shape[:-1] + (action_dim,))
        model = "bc" if self.cfg.inv_temp == 0.0 else "guided"
        actions = self.latent2action(obs, latents, model=model)
        actions = jnp.clip(actions, -1, 1)
        return actions
