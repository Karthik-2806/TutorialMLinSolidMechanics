"""Main script for Exercise 2.4: 1D Hyperelasticity."""

import datetime
import jax
import jax.numpy as jnp
import jax.random as jrandom
import klax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optax

import src.tmlsm.hyperelasticity_losses as tl1d
import src.tmlsm.hyperelasticity_model as tm1d
import src.tmlsm.plot_recipes as tplot


def load_dataset(csv_path: str):
    """Loads experimental data from the CSV file."""
    df = pd.read_csv(csv_path)

    # Convert to float32 NumPy arrays
    eps_values = df["eps"].values.astype(np.float32)
    sigma_values = df["sigma"].values.astype(np.float32)

    # Neural network input must have shape (N, 1)
    eps_jax = jnp.array(eps_values).reshape(-1, 1)
    sigma_jax = jnp.array(sigma_values)

    return eps_jax, sigma_jax


def main():
    # 1. Configuration
    # Options for model_choice: "ffnn" or "icnn"
    # Options for task_mode: "interpolation" or "extrapolation"
    model_choice = "ffnn" # "ffnn" or "icnn"
    task_mode = "extrapolation" # "interpolation" or "extrapolation"
    csv_file = r"C:\TutorialML\TutorialMLinSolidMechanics\ffnn_introduction\src\tmlsm\hyperelasticity_data.csv"

    # 2. PRNG Keys
    key = jrandom.PRNGKey(42)
    keys = jrandom.split(key, 2)

    # 3. Load all 84 points from the CSV
    eps_all, sigma_all = load_dataset(csv_file)

    # 4. Split data into training and testing sets based on task_mode:
    if task_mode == "interpolation":
        # Use all datapoints for calibration
        eps_train, sigma_train = eps_all, sigma_all
        eps_test, sigma_test = eps_all, sigma_all
    elif task_mode == "extrapolation":
        # Use data for epsilon of up to 5.0 for calibration
        train_mask = eps_all.ravel() <= 5.0
        test_mask = eps_all.ravel() > 5.0  # <--- REMAINING DATA FOR TESTING

        eps_train, sigma_train = eps_all[train_mask], sigma_all[train_mask]
        eps_test, sigma_test = eps_all[test_mask], sigma_all[test_mask]
    else:
        raise ValueError(f"Unknown task_mode: '{task_mode}'")

    print(f"--- Running {model_choice.upper()} on {task_mode.upper()} ---")
    print(f"Total points available: {len(eps_all)}")
    print(f"Points used for training: {len(eps_train)}")
    print(f"Points reserved for testing: {len(eps_test)}")
    # 5. Build Model and instantiate the Loss Function
    model = tm1d.build_1d(model_type=model_choice, key=keys[0])
    loss_fn = tl1d.StressMSELoss()

    # 6. Train the model
    start_time = datetime.datetime.now()
    model, history = klax.fit(
        model,
        (eps_train, sigma_train),
        batch_size=16,
        steps=50_000,
        loss_fn=loss_fn,
        history=klax.HistoryCallback(log_every=500),
        key=keys[1],
    )
    end_time = datetime.datetime.now()
    print(f"Training completed in: {end_time - start_time}")

    # Plot convergence history
    history.plot()

    # 7. Evaluate strictly on the discrete experimental splits
    model_final = klax.finalize(model)

    eps_dense = eps_all  # Use all points for plotting
    W_dense, sigma_dense = jax.vmap(model_final)(eps_dense)

    # 8. Plot evaluation figures
    tplot.plot_results(
        eps_all=eps_all,
        sigma_all=sigma_all,
        eps_train=eps_train,
        sigma_train=sigma_train,
        eps_dense=eps_dense,
        W_pred=W_dense,
        sigma_pred=sigma_dense,
        model_name=model_choice.upper(),
        task_mode=task_mode,
    )


if __name__ == "__main__":
    main()