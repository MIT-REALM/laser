import jax.numpy as jnp
import jax.random as jr

from ..utils.typing import PRNGKey, FloatScalar, IntScalar


def expectile_loss(adv, diff, expectile):
    """Compute the expectile loss."""
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def sample_uniform_in_hypersphere(key: PRNGKey, R: float, shape: tuple[IntScalar, ...]) -> jnp.ndarray:
    """Sample uniformly from the ball of radius R along the last dimension."""
    key_norm, key_u = jr.split(key)
    d = shape[-1]

    v = jr.normal(key_norm, shape)
    norms = safe_norm(v, axis=-1, keepdims=True)
    directions = v / (norms + 1e-8)

    radius_shape = shape[:-1] + (1,)
    u = jr.uniform(key_u, radius_shape)
    radii = R * (u ** (1.0 / (d + 1e-8)))

    samples = radii * directions

    return samples


def safe_norm(x: jnp.ndarray, axis: int = None, keepdims: bool = False, eps: FloatScalar = 1e-6) -> jnp.ndarray:
    """Compute the Euclidean norm with epsilon added before the square root."""
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis, keepdims=keepdims) + eps)
