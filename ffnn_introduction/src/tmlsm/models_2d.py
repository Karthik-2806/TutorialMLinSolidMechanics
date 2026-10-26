"""Model implementations for 2D data."""

from typing import Callable

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax

class FFNN2D(eqx.Module):
    """ A standard non convex feedforward neural network mapping 2d inputs to 1d outputs."""

    layers: tuple[callable, ...]
    activations: tuple[callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        self.layers = (
            klax.nn.Linear(2, 16, weight_init=he_normal(), key=key),
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

    linear: klax.nn.Linear

    def __init__(self, in_features, out_features, *, key: PRNGKeyArray):

        self.linear = klax.nn.Linear(
            in_features, out_features, weight_init=lambda k, s, dtype: he_normal()(k, s, dtype) * 0.1 -2, key=key
        )

    def __call__(self, x):
        non_negative_linear = eqx.tree_at( 
                    lambda l: l.weight, self.linear, jax.nn.softplus(self.linear.weight),
                    )
        return non_negative_linear(x)

class ICNN2D(eqx.Module):

    layers: tuple[callable, ...]
    activations: tuple[callable, ...]


    def __init__(self, *, key: PRNGKeyArray):

        self.layers = (

            klax.nn.Linear(2, 16, weight_init=he_normal(), key=key),
            NonNegativeLinear(16, 16, key=key),
            NonNegativeLinear(16, 16, key=key),
            NonNegativeLinear(16, "scalar", key=key)
            
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x ,
        )


    def __call__(self, x):
            for layer, activation in zip(self.layers, self.activations):
                x = activation(layer(x))
            return x

def build(*, model_type: str, key: PRNGKeyArray):

    if model_type == "icnn2d":
        return ICNN2D(key=key)
    elif model_type == "ffnn2d": 
        return FFNN2D(key=key)