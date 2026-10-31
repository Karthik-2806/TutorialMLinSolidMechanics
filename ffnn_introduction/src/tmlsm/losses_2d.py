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

class LossStrategy1_valueOnly(eqx.Module):
    """Train only on f2 function values"""

    def __call__(self, model, batch, batch_axis=0):
        # Unpack the batch data: coords and y_true
        coords, f1_true, f2_true, f2_grad_true = batch

        
        # Vectorize over all points in the batch
        y_pred, _ = jax.vmap(model)(coords)

        # Mean squared error
        loss = jnp.mean((jnp.ravel(y_pred) - jnp.ravel(f2_true))**2)
        return loss

class LossStrategy2_ValueAndGradient(eqx.Module):
    """Strategy 2: Train on BOTH f2 values and analytical gradients."""

    def __call__(self, model, batch, batch_axis=0):
        coords, f1_true, f2_true, grad_true = batch

        # Vectorize to get arrays of shape (batch_size,) and (batch_size, 2)
        y_pred, grad_pred = jax.vmap(model)(coords)

        # 1. Error on values
        loss_val = jnp.mean((jnp.ravel(y_pred) * jnp.ravel(f2_true))**2)

        # 2. Error on gradients
        loss_grad = jnp.mean((grad_pred - grad_true)**2)

        # Total combined loss
        total_loss = loss_val + loss_grad
        return total_loss

class LossStrategy3_GradientOnly(eqx.Module):
    """Strategy 3: Train ONLY on the analytical gradients."""

    def __call__(self, model, batch, batch_axis=0):
        coords, f1_true, f2_true, grad_true = batch

        # Vectorize over the batch
        _, grad_pred = jax.vmap(model)(coords)

        # Mean squared error on the 2D gradient vectors
        loss = jnp.mean((grad_pred - grad_true)**2)
        return loss