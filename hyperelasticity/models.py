"""Model implementations for Hyperelasticity (Task 2)."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class NaiveFFNN(eqx.Module):
    """
    Task 2.1: Naive FFNN model M^S.
    Maps the 6 independent components of C = F^T F to the 9 components of P.
    """

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.layers = (
            klax.nn.Linear(6, 16, weight_init=he_normal(), key=keys[0]),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=keys[1]),
            klax.nn.Linear(16, 9, weight_init=he_normal(), key=keys[3]),
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, F: jnp.ndarray) -> jnp.ndarray:
        # 1. Compute Right Cauchy-Green tensor C = F^T @ F (3x3 symmetric)
        C = F.T @ F

        # 2. Extract the 6 independent components of C (Voigt-like vector)
        x = jnp.array([
            C[0, 0],
            C[1, 1],
            C[2, 2],
            C[0, 1],
            C[1, 2],
            C[0, 2],
        ])

        # 3. Forward pass through FFNN layers
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        # 4. Reshape the 9 output components into the 3x3 PK1 stress tensor P
        P_pred = x.reshape(3, 3)
        return P_pred


def build(*, model_type: str, key: PRNGKeyArray):
    if model_type == "naive_ffnn":
        return NaiveFFNN(key=key)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")