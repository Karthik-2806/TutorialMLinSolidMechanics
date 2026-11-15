"""Main script for Task 2.1 & 2.2: Naive FFNN Model Implementation & Calibration."""

import datetime
import os

import jax
import jax.numpy as jnp
import jax.random as jrandom
import klax
from matplotlib import pyplot as plt

from data import load_hyperelastic_data
import losses as tl
import models as tm
from plot_recipes import plot_stress_parity, plot_componentwise_stress

now = datetime.datetime.now


def combine_dataset_dict(dataset_dict: dict):
    """Concatenates all load paths in a dataset dictionary into single (F, P, W) arrays."""
    F_list, P_list, W_list = [], [], []
    for case_name, data in dataset_dict.items():
        F_list.append(data["F"])
        P_list.append(data["P"])
        W_list.append(data["W"])

    F_all = jnp.concatenate(F_list, axis=0)
    P_all = jnp.concatenate(P_list, axis=0)
    W_all = jnp.concatenate(W_list, axis=0)
    return F_all, P_all, W_all


def evaluate_model_on_dataset(model_, dataset_dict: dict, dataset_name: str):
    """Evaluates finalized model on each load path and returns pred_dict for plotting."""
    print("=" * 70)
    print(f"MODEL EVALUATION ON [{dataset_name.upper()}]")
    print("=" * 70)

    pred_dict = {}
    batched_model = jax.vmap(model_)

    for case_name, data in dataset_dict.items():
        F = data["F"]
        P_true = data["P"]

        # Predict P using the trained FFNN
        P_pred = batched_model(F)

        diff_P = jnp.abs(P_pred - P_true)
        max_err_P = jnp.max(diff_P)
        mse_P = jnp.mean((P_pred - P_true) ** 2)

        pred_dict[case_name] = {"P_pred": P_pred}

        print(f"Case: '{case_name}' ({F.shape[0]} samples)")
        print(f"  [Stress P] : Max Abs Diff = {max_err_P:.4e} | MSE = {mse_P:.4e}")
        print("-" * 70)

    print()
    return pred_dict


def main():
    # 1. Create random keys for reproducible weight initialization and batch splits
    key = jrandom.PRNGKey(42)
    keys = jrandom.split(key, 2)

    # 2. Base directory setup
    base_dir = r"C:\TutorialML\TutorialMLinSolidMechanics\hyperelasticity"
    calib_folder = os.path.join(base_dir, "calibration")
    test_folder = os.path.join(base_dir, "test")

    # 3. Manually Load Calibration Datasets
    calib_data = {
        "biaxial": load_hyperelastic_data(os.path.join(calib_folder, "biaxial.txt")),
        "pure_shear": load_hyperelastic_data(os.path.join(calib_folder, "pure_shear.txt")),
        "uniaxial": load_hyperelastic_data(os.path.join(calib_folder, "uniaxial.txt")),
    }

    # 4. Manually Load Test Datasets
    test_data = {
        "biax_test": load_hyperelastic_data(os.path.join(test_folder, "biax_test.txt")),
        "mixed_test": load_hyperelastic_data(os.path.join(test_folder, "mixed_test.txt")),
    }

    # Concatenate all calibration paths into training tensors (F_train, P_train)
    F_train, P_train, _ = combine_dataset_dict(calib_data)
    print(f"Total calibration samples loaded: {F_train.shape[0]}")

    # 5. Build Naive FFNN model instance (M^S)
    model_choice = "naive_ffnn"
    model = tm.build(model_type=model_choice, key=keys[0])
    print(model)

    # 6. Calibrate model using klax.fit
    t1 = now()
    print("Calibration started at:", t1)

    model, history = klax.fit(
        model,
        (F_train, P_train),
        batch_size=32,
        steps=10_000,
        loss_fn=tl.MSE(),
        history=klax.HistoryCallback(log_every=100),
        key=keys[1],
    )

    t2 = now()
    print("It took", t2 - t1, "(sec) to calibrate the model")

    # Plot training loss history
    history.plot()
    plt.show()

    # 7. Finalize model for evaluation
    model_ = klax.finalize(model)

    # 8. Evaluate on Calibration and Test Datasets
    calib_preds = evaluate_model_on_dataset(model_, calib_data, "Calibration Data")
    test_preds = evaluate_model_on_dataset(model_, test_data, "Test Data")

    # -----------------------------------------------------------------------
    # Visual Stress Comparison (Component-wise & Parity)
    # -----------------------------------------------------------------------
    # Calibration evaluation
    plot_stress_parity(calib_data, calib_preds, title_prefix="Task 2.2 Calibration")
    plot_componentwise_stress(calib_data, calib_preds, title_prefix="Task 2.2 Calibration")

    # Test evaluation
    plot_stress_parity(test_data, test_preds, title_prefix="Task 2.2 Test (Extrapolation)")
    plot_componentwise_stress(test_data, test_preds, title_prefix="Task 2.2 Test (Extrapolation)")


if __name__ == "__main__":
    main()