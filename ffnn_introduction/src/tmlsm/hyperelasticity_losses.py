"""Loss function for 1D hyperelasticity."""

import equinox as eqx
import jax
import jax.numpy as jnp


class StressMSELoss(eqx.Module):
    """Mean Squared Error calibrated strictly on stress sigma = dW/deps."""

    def __call__(self, model, batch, batch_axis=0):
        # Unpack batch: strain inputs and experimental stress targets
        eps_batch, sigma_true_batch = batch

        # Forward pass: model returns (W, sigma)
        _, sigma_pred = jax.vmap(model)(eps_batch)

        # Mean Squared Error on stress
        loss = jnp.mean((jnp.ravel(sigma_pred) - jnp.ravel(sigma_true_batch)) ** 2)
        return loss