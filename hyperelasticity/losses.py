"""Loss functions supporting tuple data containers (F, P) for hyperelasticity models."""

import equinox as eqx
import jax
import jax.numpy as jnp


class MSE(eqx.Module):
    """Mean Squared Error loss for positional tuple batches: (F, P)."""

    def __call__(self, model, batch, batch_axis=0):
        # Unpack tuple directly: F of shape (batch, 3, 3), P of shape (batch, 3, 3)
        F, P_true = batch

        # Vectorized forward pass over batch
        P_pred = jax.vmap(model)(F)

        # Ravel ensures shape compatibility across all 9 stress components
        return jnp.mean((jnp.ravel(P_pred) - jnp.ravel(P_true)) ** 2)