import os
import numpy as np
import jax.numpy as jnp

def load_hyperelastic_data(file_path : str):
    """Loads a single test file containing 19 columns:

    Cols 0-8:   F components flattened (row-major)
    Cols 9-17:  P components flattened (row-major)
    Col 18:     W (strain energy density scalar)
    """
    raw_data = np.loadtxt(file_path)

    # convert to jax arrays and cast to float32

    raw_jax = jnp.array(raw_data, dtype = jnp.float32)

    # Reshape F and P into 3*3 matrices and W into a scalar

    F = raw_jax[:, 0:9].reshape(-1,3,3)
    P = raw_jax[:, 9:18].reshape(-1,3,3)
    W = raw_jax[:, 18]

    return F, P, W

def load_dataset_folder(folder_path: str):
    """Loads all test/calibration files in a directory into a dictionary."""
    dataset = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and not filename.lower().startswith("readme"):
            # (e.g., "uniaxial_tension.txt" becomes "uniaxial_tension")
            case_name = os.path.splitext(filename)[0]  
            # joins the folder path and filename to create a full path to the file
            full_path = os.path.join(folder_path, filename)
            F, P, W = load_hyperelastic_data(full_path)
            dataset[case_name] = {"F": F, "P": P, "W": W}
    return dataset
