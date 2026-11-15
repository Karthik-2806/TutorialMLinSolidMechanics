import os
import numpy as np
import jax.numpy as jnp


def load_hyperelastic_data(file_path: str):
    """Loads a single test file containing 19 columns:

    Cols 0-8:   F components flattened (row-major)
    Cols 9-17:  P components flattened (row-major)
    Col 18:     W (strain energy density scalar)
    """
    raw_data = np.loadtxt(file_path)
    raw_jax = jnp.array(raw_data, dtype=jnp.float32)

    # Reshape F and P into 3x3 matrices and W into a 1D scalar array
    F = raw_jax[:, 0:9].reshape(-1, 3, 3)
    P = raw_jax[:, 9:18].reshape(-1, 3, 3)
    W = raw_jax[:, 18]

    return {"F": F, "P": P, "W": W}


def load_invariant_data(file_path: str):
    """
    Loads a single invariant reference file and explicitly returns it 
    as an (N, 4) JAX array [I1, J, I4, I5].
    """
    raw_data = np.loadtxt(file_path)
    inv_jax = jnp.array(raw_data, dtype=jnp.float32)
    
    # Ensure it is 2D (N, 4) even if it loaded as a flat 1D array
    if inv_jax.ndim == 1:
        inv_jax = inv_jax.reshape(-1, 4)
        
    return inv_jax