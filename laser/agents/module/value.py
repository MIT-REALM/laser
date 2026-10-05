import flax.linen as nn
import jax.numpy as jnp

from typing import Sequence

from ...utils.typing import Obs, Action
from ...utils.networks import MLP, ensemblize


class Value(nn.Module):
    """MLP ensemble for state values V(s) or action values Q(s, a).

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2

    @nn.compact
    def __call__(self, observations: Obs, actions: Action = None):
        """Evaluate state values, or action values when actions are provided."""
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        v = value_net(inputs).squeeze(-1)

        return v
