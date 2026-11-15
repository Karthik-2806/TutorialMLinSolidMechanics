import os
import numpy as np
import jax.numpy as jnp
from jax import grad, vmap

from data import load_dataset_folder
from plot_recipes import (
    plot_3d_constitutive_manifolds,
    plot_stretch_trajectories,
    plot_3d_calibration_vs_test,
    plot_vonmises_stress_comparison,
)

# ---------------------------------------------------------------------------
# Structural Tensor Definition (Eq. 5)
# ---------------------------------------------------------------------------
G_ti = jnp.diag(jnp.array([4.0, 0.5, 0.5]))


# ---------------------------------------------------------------------------
# Core Continuum Mechanics & JAX Formulations
# ---------------------------------------------------------------------------
def compute_invariants(F: jnp.ndarray):
    """Computes I_1, J, I_4, and I_5 from deformation gradient F (Eq. 4)."""
    C = F.T @ F
    J = jnp.linalg.det(F)
    I1 = jnp.trace(C)
    I3 = J**2

    # cof(C) = I_3 * C^{-T} (C is symmetric, so C^{-T} = C^{-1})
    C_inv = jnp.linalg.inv(C)
    cof_C = I3 * C_inv.T

    I4 = jnp.trace(C @ G_ti)
    I5 = jnp.trace(cof_C @ G_ti)
    return I1, J, I4, I5


def strain_energy(F: jnp.ndarray) -> jnp.ndarray:
    """Computes scalar strain energy density W(F) (Eq. 3)."""
    I1, J, I4, I5 = compute_invariants(F)
    W = 8.0 * I1 + 10.0 * (J**2) - 56.0 * jnp.log(J) + 0.2 * (I4**2 + I5**2) - 44.0
    return W


# First Piola-Kirchhoff stress tensor: P = dW/dF (Eq. 6)
compute_PK1_stress = grad(strain_energy)

# Vectorized functions across batch dimension (N, 3, 3)
compute_invariants_batched = vmap(compute_invariants)
strain_energy_batched = vmap(strain_energy)
compute_PK1_stress_batched = vmap(compute_PK1_stress)


# ---------------------------------------------------------------------------
# Task 1: Invariants Check
# ---------------------------------------------------------------------------
def check_invariants_against_folder(dataset_dict: dict, invariants_dir: str):
    """
    Computes invariants in JAX and compares them against reference data in
    the folder 'hyperelasticity/data/invariants'.
    """
    print("=" * 70)
    print("TASK 1: Invariants Check (JAX vs. hyperelasticity/data/invariants)")
    print("=" * 70)

    if not os.path.exists(invariants_dir):
        print(f"Warning: Folder '{invariants_dir}' not found. Skipping reference file check.\n")
        return

    # Check case-specific reference files or split reference files
    for case_name, data in dataset_dict.items():
        F = data["F"]
        I1_pred, J_pred, I4_pred, I5_pred = compute_invariants_batched(F)
        inv_jax = jnp.stack([I1_pred, J_pred, I4_pred, I5_pred], axis=-1)

        # Look for matching reference files: case_name.txt or case_name.npy
        ref_txt = os.path.join(invariants_dir, f"{case_name}.txt")
        ref_npy = os.path.join(invariants_dir, f"{case_name}.npy")

        ref_file = ref_npy if os.path.exists(ref_npy) else (ref_txt if os.path.exists(ref_txt) else None)

        if ref_file:
            inv_ref = np.load(ref_file) if ref_file.endswith(".npy") else np.loadtxt(ref_file)
            inv_ref = jnp.array(inv_ref, dtype=jnp.float32)

            # In case the file contains 4 columns [I1, J, I4, I5]
            if inv_ref.ndim == 2 and inv_ref.shape[-1] >= 4:
                diff = jnp.max(jnp.abs(inv_jax - inv_ref[:, :4]), axis=0)
                print(f"Case '{case_name}': Verified against '{os.path.basename(ref_file)}'")
                print(f"  Max Diff: I1={diff[0]:.2e}, J={diff[1]:.2e}, I4={diff[2]:.2e}, I5={diff[3]:.2e}")
            else:
                max_diff = jnp.max(jnp.abs(inv_jax - inv_ref))
                print(f"Case '{case_name}': Max Invariant Discrepancy = {max_diff:.2e}")
        else:
            print(f"Case '{case_name}': Computed ({len(F)} points). (No dedicated file '{case_name}.txt' in {invariants_dir})")
            print(f"  Sample ranges -> J: [{jnp.min(J_pred):.3f}, {jnp.max(J_pred):.3f}] | I4: [{jnp.min(I4_pred):.3f}, {jnp.max(I4_pred):.3f}]")

    print("Task 1 verification complete.\n")


# ---------------------------------------------------------------------------
# Tasks 2 & 3: Potential Check (W) and Stress Evaluation (P)
# ---------------------------------------------------------------------------
def evaluate_potential_and_stress(dataset_dict: dict, dataset_name: str):
    """
    Evaluates:
    - Task 2: W(F) vs. imported W
    - Task 3: P = dW/dF vs. imported P
    Returns a predictions dictionary containing P_pred for plotting.
    """
    print("=" * 70)
    print(f"TASKS 2 & 3: Potential & Stress Evaluation [{dataset_name.upper()}]")
    print("=" * 70)

    pred_dict = {}

    for case_name, data in dataset_dict.items():
        F = data["F"]
        P_true = data["P"]
        W_true = data["W"]

        # Task 2: Compute W(F)
        W_pred = strain_energy_batched(F)
        diff_W = jnp.abs(W_pred - W_true)
        max_err_W = jnp.max(diff_W)
        mae_W = jnp.mean(diff_W)

        # Task 3: Compute P = dW/dF via jax.grad
        P_pred = compute_PK1_stress_batched(F)
        diff_P = jnp.abs(P_pred - P_true)
        max_err_P = jnp.max(diff_P)
        mae_P = jnp.mean(diff_P)

        pred_dict[case_name] = {"P_pred": P_pred, "W_pred": W_pred}

        print(f"Case: '{case_name}' ({F.shape[0]} samples)")
        print(f"  [Task 2 - Potential W] : Max Abs Diff = {max_err_W:.4e} | MAE = {mae_W:.4e}")
        print(f"  [Task 3 - Stress P   ] : Max Abs Diff = {max_err_P:.4e} | MAE = {mae_P:.4e}")
        print("-" * 70)

    print()
    return pred_dict


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    base_dir = r"C:\TutorialML\TutorialMLinSolidMechanics\hyperelasticity"
    calib_folder = os.path.join(base_dir, "calibration")
    test_folder = os.path.join(base_dir, "test")
    invariants_folder = os.path.join(base_dir, "data", "invariants")

    # Load datasets
    calib_data = load_dataset_folder(calib_folder)
    test_data = load_dataset_folder(test_folder)

    # Initial kinematic & 3D overview plots
    # plot_3d_constitutive_manifolds(calib_data)
    # plot_3d_calibration_vs_test(calib_data, test_data)
    # plot_stretch_trajectories(calib_data)

    # -----------------------------------------------------------------------
    # Task 1: Check Invariants against reference data
    # -----------------------------------------------------------------------
    check_invariants_against_folder(calib_data, invariants_folder)

    # -----------------------------------------------------------------------
    # Task 2 & Task 3: Potential (W) and Stress (P) Evaluation
    # -----------------------------------------------------------------------
    calib_preds = evaluate_potential_and_stress(calib_data, "Calibration Data")
    test_preds = evaluate_potential_and_stress(test_data, "Test Data")

    # -----------------------------------------------------------------------
    # Visual Stress Comparison (von Mises sigma_vM: Predicted vs. True)
    # -----------------------------------------------------------------------
    plot_vonmises_stress_comparison(calib_data, calib_preds, title_prefix="Calibration")
    plot_vonmises_stress_comparison(test_data, test_preds, title_prefix="Test")


if __name__ == "__main__":
    main()