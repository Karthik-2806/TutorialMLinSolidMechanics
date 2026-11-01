"""1D Neural Network Energy Potentials and Sobolev Derivative Wrappers."""

from typing import Callable
import equinox as eqx
import jax
from jax.nn.initializers import he_normal
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import klax


class FFNN1D(eqx.Module):
    """Standard unconstrained Feedforward Network representing W(eps)."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        # 1. Split key into 4 separate keys so layers do not have identical weights
        keys = jax.random.split(key, 4)

        # 2. 1D strain input -> 16 hidden units -> 1D scalar energy output
        self.layers = (
            klax.nn.Linear(1, 4, weight_init=he_normal(), key=keys[0]),
            klax.nn.Linear(4, 4, weight_init=he_normal(), key=keys[1]),
            klax.nn.Linear(4, 4, weight_init=he_normal(), key=keys[2]),
            klax.nn.Linear(4, "scalar", weight_init=he_normal(), key=keys[3]),
        )

        # 3. Softplus activation guarantees smooth derivatives for stress dW/deps
        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, eps):
        # Forward pass through layers
        x = eps
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


class NonNegativeLinear1D(eqx.Module):
    """Linear layer forcing weights >= 0 via softplus for convexity."""

    linear: klax.nn.Linear

    def __init__(self, in_features, out_features, *, offset: float = 0.0, key: PRNGKeyArray,):
        def custom_init(k, s, dtype=jnp.float32):
            return he_normal()(k, s, dtype)*0.01 - offset

        self.linear = klax.nn.Linear( in_features, out_features, weight_init=custom_init, key=key,)

    def __call__(self, x):
        positive_weights = jax.nn.softplus(self.linear.weight)
        layer_with_positive_weights = eqx.tree_at(
            lambda l: l.weight,
            self.linear,
            positive_weights,
        )
        return layer_with_positive_weights(x)


class ICNN1D(eqx.Module):
    """1D Input Convex Neural Network guaranteeing convex W(eps)."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, offset: float = 0.0, key: PRNGKeyArray):

        self.layers = (
            klax.nn.Linear(1, 16, weight_init=he_normal(), key=key),
            NonNegativeLinear1D(16, 16, offset=offset, key=key),
            NonNegativeLinear1D(16, 16, offset=offset, key=key),
            NonNegativeLinear1D(16, "scalar", offset=offset, key=key),
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, eps):
        x = eps
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


class HyperelasticSobolevModel(eqx.Module):
    """Wraps an energy model W(eps) and evaluates stress sigma = dW / deps."""

    potential_nn: eqx.Module

    def __init__(self, potential_nn: eqx.Module):
        self.potential_nn = potential_nn

    def __call__(self, eps):
        # 1. Output the internal strain-energy density W
        energy_W = self.potential_nn(eps)

        # 2. Define scalar energy closure for differentiation
        def get_scalar_energy(eps_input):
            raw_val = self.potential_nn(eps_input)
            return jnp.squeeze(raw_val)

        # 3. Compute derivative dW/deps (stress sigma)
        stress_sigma = jax.grad(get_scalar_energy)(eps)

        return energy_W, stress_sigma


def build_1d(*, model_type: str, key: PRNGKeyArray):
    """Builds the Sobolev model using the selected energy potential."""
    if model_type == "ffnn":
        base_potential = FFNN1D(key=key)
    elif model_type == "icnn":
        base_potential = ICNN1D(offset=1.0, key=key)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    return HyperelasticSobolevModel(base_potential)