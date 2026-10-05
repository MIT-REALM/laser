import flax.linen as nn
import jax.numpy as jnp
import distrax

from typing import Sequence

from .distribution import TanhMultivariateNormalDiag
from ...utils.typing import Obs, Action, FloatScalar
from ...utils.networks import MLP, default_init


class ActorVectorField(nn.Module):
    """Vector field for state-conditioned or state-independent flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: Obs, actions: Action, times: FloatScalar = None):
        """Evaluate the vector field on two or three concatenated inputs.

        State-conditioned flows pass observations, actions, and times.
        State-independent flows pass latents and times as the first two inputs.
        One-step policies pass observations and latents, omitting times.
        """
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)(inputs)

        return v


class StdTanhNormalPolicy(nn.Module):
    """Gaussian policy with tanh squashing and optional action scaling.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Lower bound of the log standard deviation.
        log_std_max: Upper bound of the log standard deviation.
        low: Lower bound of action space.
        high: Upper bound of action space.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: float | None = -20
    log_std_max: float | None = 2
    low: float | None = None
    high: float | None = None

    @nn.compact
    def __call__(self, observations: Obs) -> distrax.Distribution:
        """Return the action distribution given observations."""
        outputs = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution
