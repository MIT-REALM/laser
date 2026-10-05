import dataclasses
import numpy as np
import jax.random as jr
import jax
import copy
import optax
import jax.numpy as jnp
import jax.tree_util as jtu
import functools as ft

from dataclasses import dataclass
from scipy.stats import chi2
from flax import struct
from cyclopts import Parameter
from optax.tree_utils import tree_where
from typing import Annotated

from .module.value import Value
from .module.actor import ActorVectorField
from .module.utils import ModuleDict, TrainState
from .utils import sample_uniform_in_hypersphere
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip, clip_top_k_percent_grads
from ..utils.typing import Obs, Action, PRNGKey, Params


@Parameter(name="*", group="AgentConfig")
@dataclass
class LASERCfg:
    """Configuration for the LASER agent.

    Parameters
    ----------
    lr : float
        Learning rate for all networks.
    max_grad_norm : float
        Maximum gradient norm for each loss group before combining gradients.
    actor_hidden_dims : list[int]
        Hidden layer widths for the BC decoder and latent flow networks.
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
    action_flow_steps : int
        Number of integration steps for flow sampling and adjoint matching; must be a positive integer.
    sigma0_conf_level : float
        Probability mass of the initial Gaussian inside the latent ball; determines sigma0.
        Must be strictly between 0 and 1.
    inv_temp : float
        Inverse entropy temperature (1/alpha), scaling terminal adjoint targets.
    clip_q_grad : bool
        Cap terminal latent Q-gradient norms at the batch's 90th percentile.
    normalize_q_grad : bool
        Divide terminal latent Q-gradients by their mean norm before applying inv_temp.
    action_dim : int | None
        Dimension of a single action; set automatically during agent creation.
    obs_dims : tuple[int, ...] | None
        Observation shape without the batch dimension; set automatically during agent creation.
    r_max : float | None
        Latent ball radius; set to the square root of the full action dimension during agent creation.
    sigma0 : float | None
        Initial Gaussian standard deviation; computed from sigma0_conf_level during agent creation.
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
    sigma0_conf_level: float = 0.999
    inv_temp: float = 10.0
    clip_q_grad: bool = True
    normalize_q_grad: bool = True

    # Environment-derived values (set by LASERAgent.create).
    action_dim: int | None = None
    obs_dims: tuple[int, ...] | None = None
    r_max: float | None = None
    sigma0: float | None = None

    def __post_init__(self):
        for name in ("horizon_length", "action_flow_steps", "n_critic_ensembles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer (got {value!r}).")
        if not 0 < self.sigma0_conf_level < 1:
            raise ValueError(
                f"sigma0_conf_level must be strictly between 0 and 1 (got {self.sigma0_conf_level!r})."
            )

    @property
    def agent_name(self):
        return "laser"


class LASERAgent(struct.PyTreeNode):
    network: TrainState
    cfg: LASERCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: LASERCfg,
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
        # Critic.
        critic_def = Value(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.value_layer_norm,
            num_ensembles=cfg.n_critic_ensembles,
        )
        # Decode uniform ball latents into dataset actions.
        bc_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        # State-independent Gaussian-to-uniform base latent flow.
        latent_base_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        # State-conditioned latent policy trained by adjoint matching.
        latent_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=full_action_dim,
            layer_norm=cfg.actor_layer_norm,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            bc_flow=(bc_flow_def, (ex_observations, full_actions, ex_times)),
            latent_base_flow=(latent_base_flow_def, (full_actions, ex_times)),
            latent_flow=(latent_flow_def, (ex_observations, full_actions, ex_times)),
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

        # Choose sigma0 so P(||base_samples|| <= r_max) = sigma0_conf_level.
        r_max = np.sqrt(full_action_dim)
        q = chi2.ppf(cfg.sigma0_conf_level, df=full_action_dim)
        sigma0 = r_max / np.sqrt(q)
        cfg = dataclasses.replace(
            cfg,
            action_dim=action_dim,
            obs_dims=obs_dims,
            r_max=r_max,
            sigma0=sigma0,
        )

        return cls(network=network, cfg=cfg)

    @jax.jit
    def update(self, batch: dict, key: PRNGKey):

        def loss_fn_(params):
            total_loss_, info_ = self.total_loss(batch, params, key)
            return total_loss_, info_

        # Clip the latent-policy gradients separately from the other networks.
        grad, info = jax.grad(loss_fn_, has_aux=True)(self.network.params)
        adj_grad = grad.pop('modules_latent_flow')
        grad_ill = has_any_nan_or_inf(grad)
        grad, grad_norm = compute_norm_and_clip(grad, self.cfg.max_grad_norm)
        grad = tree_where(grad_ill, jtu.tree_map(jnp.zeros_like, grad), grad)
        adj_grad_ill = has_any_nan_or_inf(adj_grad)
        adj_grad, adj_grad_norm = compute_norm_and_clip(adj_grad, self.cfg.max_grad_norm)
        adj_grad = tree_where(adj_grad_ill, jtu.tree_map(jnp.zeros_like, adj_grad), adj_grad)

        # Reassemble the gradient groups for a single Adam step.
        grad['modules_latent_flow'] = adj_grad
        new_network = self.network.apply_gradients(grads=grad)

        # Update target network.
        self.target_update(new_network, "critic")

        return self.replace(network=new_network), info | {
            "total/grad_norm": jnp.hypot(grad_norm, adj_grad_norm),
            "total/grad_ill": grad_ill | adj_grad_ill,
            "adj/grad_norm": adj_grad_norm,
            "adj/grad_ill": adj_grad_ill,
        }

    def target_update(self, network: TrainState, module_name: str):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.cfg.tau + tp * (1 - self.cfg.tau),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def total_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Combine critic, flow-matching, and adjoint-matching losses."""
        info = {}
        adj_key, flow_key, critic_key = jr.split(key, 3)

        # Critic loss.
        critic_loss, critic_info = self.critic_loss(batch, params, critic_key)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Flow-matching loss.
        flow_matching_loss, flow_info = self.flow_matching_loss(batch, params, flow_key)
        for k, v in flow_info.items():
            info[f'flow_matching/{k}'] = v

        # Adjoint-matching loss.
        adj_loss, adj_info = self.adjoint_matching_loss(batch, params, adj_key)
        info.update(adj_info)

        loss = critic_loss + flow_matching_loss + adj_loss
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

    @ft.partial(jax.jit, static_argnames=("flow_steps",))
    def compute_adjoint_targets(self, obs, key, flow_steps=None):
        """Sample latent trajectories and compute their lean-adjoint targets."""
        flow_steps = self.cfg.action_flow_steps if flow_steps is None else flow_steps
        action_dim = self.cfg.action_dim * (self.cfg.horizon_length if self.cfg.action_chunking else 1)

        x0_key, key = jr.split(key, 2)
        x = jr.normal(x0_key, shape=obs.shape[:-1] + (action_dim,)) * self.cfg.sigma0

        latent_base_flow = self.network.select("latent_base_flow")

        # Sample a training trajectory using the regularized SDE.
        h = 1 / flow_steps
        xs = [x]
        ts = []
        noise_key, key = jr.split(key)
        for i, cur_noise_key in zip(range(flow_steps), jr.split(noise_key, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            # Shift endpoint coefficients by h to avoid the t=0 singularity.
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h)) * self.cfg.sigma0
            noise = jr.normal(cur_noise_key, x.shape)
            if i != flow_steps - 1:
                v = self.network.select("latent_flow")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:
                # A noiseless reference-flow step, then latent-ball projection.
                x = x + h * latent_base_flow(x, t)
                x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
                x = jnp.where(x_norm > self.cfg.r_max, x / (x_norm + 1e-8) * self.cfg.r_max, x)

            xs.append(x)
            ts.append(t)

        # Terminal adjoint = -(1/alpha) * grad_z Q(s, decoder(s, z)).
        # Differentiate through the decoder with its parameters held fixed.
        grad_fn = jax.grad(
            lambda x, y: self.network.select("target_critic")(
                x, jnp.clip(self.latent2action(x, y), -1., 1.)).sum() / self.cfg.n_critic_ensembles, 1)

        if self.cfg.clip_q_grad:
            grad = grad_fn(obs, xs[-1])
            # Cap terminal latent gradient norms at the 90th percentile.
            grad = clip_top_k_percent_grads(grad, k=10.0)
        else:
            grad = grad_fn(obs, xs[-1])

        if self.cfg.normalize_q_grad:
            grad_norms = jnp.linalg.norm(grad, axis=-1, keepdims=True)
            mean_grad_norm = jnp.mean(grad_norms)
            normalized_grad = grad / jax.lax.stop_gradient(jnp.maximum(mean_grad_norm, 1e-6))
            adj = -normalized_grad * self.cfg.inv_temp
        else:
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
                return 2 * self.network.select("latent_base_flow")(xi, t + h) - xi / (t + h)

            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            adjs.append(adj)

        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    def adjoint_matching_loss(self, batch: dict, params: Params, key: PRNGKey):
        # Rollouts, adjoints, and the reference field use fixed self.network weights.
        adj_key, key = jr.split(key)
        xs, adjs, ts, pre_adj_info = self.compute_adjoint_targets(batch["observations"], adj_key)
        h = 1 / self.cfg.action_flow_steps
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h)) * self.cfg.sigma0
        observations = jnp.repeat(batch["observations"][None], self.cfg.action_flow_steps, axis=0)

        # Only the latent policy field receives params; the regression target stays detached.
        vf_latent = self.network.select("latent_flow")(observations, xs, ts, params=params)
        vf_base = self.network.select("latent_base_flow")(xs, ts)

        # Residual = 2 * (latent_flow - latent_base_flow) / sigma + sigma * adjoint.
        adj_loss = jnp.sum(jnp.square((vf_latent - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)
        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info = {}
        info.update(pre_adj_info)

        return adj_loss, info | {'adj_loss': adj_loss}

    def flow_matching_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Flow matching for the BC decoder and uniform latent reference."""
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

        # Pair Gaussian samples with independent uniform latents.
        key, x0_key, x1_key, t_key = jr.split(key, 4)
        x_0 = jr.normal(x0_key, (batch_size, action_dim)) * self.cfg.sigma0
        x_1 = sample_uniform_in_hypersphere(x1_key, self.cfg.r_max, (batch_size, action_dim))
        t = jr.uniform(t_key, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('latent_base_flow')(x_t, t, params=params)
        latent_base_flow_loss = jnp.mean((pred - vel) ** 2)

        flow_matching_loss = bc_flow_loss + latent_base_flow_loss

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
            'bc_flow_loss': bc_flow_loss,
            'latent_base_flow_loss': latent_base_flow_loss,
            'flow_matching_loss_ill': has_any_nan_or_inf(flow_matching_loss),
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
        """Integrate the latent policy ODE and project its endpoint."""
        latents = base_samples

        def euler_step_(i_, latents_):
            t = jnp.full((*obs.shape[:-1], 1), i_ / self.cfg.action_flow_steps)
            vels = self.network.select('latent_flow')(obs, latents_, t, params=params)
            latent_step = vels / self.cfg.action_flow_steps
            next_latents_ = latents_ + latent_step
            latents_ = next_latents_
            return latents_

        latents = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, latents)

        # Project only the terminal latent onto the ball before BC decoding.
        norms = jnp.linalg.norm(latents, axis=-1, keepdims=True)
        latents = jnp.where(
            norms > self.cfg.r_max,
            latents / (norms + 1e-8) * self.cfg.r_max,
            latents
        )

        return latents

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        # Gaussian base -> projected latent policy -> BC decoder -> action.
        action_key, noise_key = jr.split(key)
        action_dim = self.cfg.action_dim * (self.cfg.horizon_length if self.cfg.action_chunking else 1)
        base_samples = jr.normal(noise_key, shape=obs.shape[:-1] + (action_dim,)) * self.cfg.sigma0
        latents = self.base2latent(obs, base_samples)
        actions = self.latent2action(obs, latents)
        actions = jnp.clip(actions, -1, 1)
        return actions
