"""Model implementations."""

from typing import Callable

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class Model(eqx.Module):
    """A custom trainable `equinox.Module`."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        self.layers = (
            klax.nn.Linear("scalar", 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, "scalar", weight_init=he_normal(), key=key),
        )

        self.activations = (
            jax.nn.relu,
            jax.nn.relu,
            jax.nn.relu,
            lambda x: x,
        )

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x        # predicted output of the model

class NonNegativeLinear(eqx.Module):
    """Linear layer with non-negative weights (W >= 0) via softplus transformation."""
    linear: klax.nn.Linear

    def __init__(self, in_features, out_features, *, key: PRNGKeyArray):
        self.linear = klax.nn.Linear( 
            in_features, out_features, weight_init=lambda k, s, dtype: he_normal()(k, s, dtype) * 0.1 - 3, key=key
            )
        # weight_init=lambda k, s, dtype: he_normal()(k, s, dtype) * 0.1 - 3 
        # This is a custom weight initialization formula. 
        # It generates standard he_normal random weights, shrinks them (* 0.1), and shifts them down by - 3 

    def __call__(self,x):
        non_negative_linear = eqx.tree_at( 
            lambda l: l.weight, self.linear, jax.nn.softplus(self.linear.weight),
            )
        return non_negative_linear(x)


class ICNNModel(eqx.Module):
    """ A custom trainable Input Convex Neural Network """

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):

        self.layers = (
            # Layer 0: First hidden layer -> unconstrained weights
            # 1. Input -> Hidden 1: unconstrained weights
            klax.nn.Linear("scalar", 16, weight_init=he_normal(), key=key),

            # Layer 1: Subsequent hidden layer -> non-negative weights (W >= 0)
            # 2. Hidden 1 -> Hidden 2: non-negative weights (16 -> 16)
            NonNegativeLinear(16, 16, key=key),
            NonNegativeLinear(16, 16, key=key),
            
            # Layer 2: Output layer -> non-negative weights (W >= 0)
            # 3. Hidden 2 -> Output: non-negative weights (16 -> scalar)
            NonNegativeLinear(16, "scalar", key=key),
        )

        self.activations = (
            jax.nn.relu,
            jax.nn.relu,
            jax.nn.relu,
            lambda x: x,
        )

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


def build(*, model_type: str, key: PRNGKeyArray):
    """Build and return either FFNN or ICNN."""
    if model_type == "icnn":
        return ICNNModel(key=key)
    elif model_type == "ffnn": 
        return Model(key=key)