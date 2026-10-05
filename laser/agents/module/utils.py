import flax.linen as nn
import functools as ft
import optax

from flax import struct
from typing import Mapping, Sequence, Callable

from ...utils.typing import Params


class ModuleDict(nn.Module):
    """Named modules initialized together and callable individually.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(struct.PyTreeNode):
    """Model parameters and optimizer state shared by a collection of networks.

    Attributes:
        step: Counter incremented by each apply_gradients call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: Optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    model_def: ModuleDict = struct.field(pytree_node=False)
    params: Params = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Evaluate the model with explicit parameters or the stored parameters.

        During loss differentiation, pass the traced params to train a network;
        omit params to use the fixed weights stored in this state.

        Args:
            *args: Positional model inputs.
            params: Explicit parameters, or None to use the stored parameters.
            method: Model method name, or None to use its default call.
            **kwargs: Keyword model inputs.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Bind a named module for subsequent calls."""
        return ft.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
