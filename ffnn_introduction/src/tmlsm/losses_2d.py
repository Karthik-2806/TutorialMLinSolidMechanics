"""Loss functions supporting tuple data containers (x, y) for 2D models."""

import equinox as eqx
import jax
import jax.numpy as jnp


class MSE(eqx.Module):
    """Mean Squared Error loss for positional tuple batches: (x, y)."""

    def __call__(self, model, batch, batch_axis=0):
        # Unpack tuple directly instead of dictionary keys
        x, y = batch

        # Vectorized forward pass
        y_pred = jax.vmap(model)(x)

        # Ravel ensures shape compatibility (e.g., (N, 1) vs (N,))
        return jnp.mean((jnp.ravel(y_pred) - jnp.ravel(y)) ** 2)
