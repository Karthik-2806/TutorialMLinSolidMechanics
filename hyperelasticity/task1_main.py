import os
import jax.numpy as jnp
from jax import grad, vmap

from data import load_hyperelastic_data, load_invariant_data
from plot_recipes import (
    plot_task1_1_raw_data,
    plot_invariants_comparison,
    plot_W_comparison,
    plot_P_comparison,
    plot_grouped_raw_data,
)

# ---------------------------------------------------------------------------
# Structural Tensor Definition (Eq. 5)
# ---------------------------------------------------------------------------
G_ti = jnp.diag(jnp.array([4.0, 0.5, 0.5]))


# ---------------------------------------------------------------------------
# Core Continuum Mechanics & JAX Formulations (Task 1.2)
# ---------------------------------------------------------------------------
def compute_invariants(F: jnp.ndarray):
    """Computes I_1, J, I_4, and I_5 from deformation gradient F (Eq. 4)."""
    C = F.T @ F
    J = jnp.linalg.det(F)
    I1 = jnp.trace(C)
    I3 = J**2

    # cof(C) = I_3 * C^{-T}
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
# Task 1.2 Evaluation: Invariants, W, and P across all datasets
# ---------------------------------------------------------------------------
def evaluate_task1_2(dataset_dict: dict, inv_ref_dict: dict, dataset_label: str):
    """
    Computes Invariants, W(F), and P = dW/dF in JAX and compares them against
    the explicit reference data dictionary.
    """
    print("=" * 75)
    print(f"TASK 1.2 VERIFICATION: [{dataset_label.upper()}]")
    print("=" * 75)

    pred_dict = {}

    for case_name, data in dataset_dict.items():
        F = data["F"]
        P_true = data["P"]
        W_true = data["W"]

        # 1. Compute Invariants [I1, J, I4, I5]
        I1_p, J_p, I4_p, I5_p = compute_invariants_batched(F)
        inv_pred = jnp.stack([I1_p, J_p, I4_p, I5_p], axis=-1)

        # 2. Compute Strain Energy W(F)
        W_pred = strain_energy_batched(F)

        # 3. Compute First Piola-Kirchhoff Stress P = dW/dF
        P_pred = compute_PK1_stress_batched(F)

        pred_dict[case_name] = {
            "inv_pred": inv_pred,
            "W_pred": W_pred,
            "P_pred": P_pred,
        }

        print(f"Case: '{case_name}' ({F.shape[0]} load steps)")

        # Check invariants against folder directly by case_name
        if case_name in inv_ref_dict:
            inv_true = inv_ref_dict[case_name]
            diff_inv = jnp.max(jnp.abs(inv_pred - inv_true[:, :4]), axis=0)
            print(
                f"  [Invariants Max Abs Diff] : I1={diff_inv[0]:.2e}, J={diff_inv[1]:.2e}, "
                f"I4={diff_inv[2]:.2e}, I5={diff_inv[3]:.2e}"
            )
        else:
            print(f"  [Invariants]              : Explicit reference array for '{case_name}' not provided.")

        # Check W and P against imported data
        max_err_W = jnp.max(jnp.abs(W_pred - W_true))
        max_err_P = jnp.max(jnp.abs(P_pred - P_true))
        print(f"  [Potential W Max Abs Diff]: {max_err_W:.4e}")
        print(f"  [Stress P Max Abs Diff   ]: {max_err_P:.4e}")
        print("-" * 75)

    print()
    return pred_dict


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    base_dir = r"C:\TutorialML\TutorialMLinSolidMechanics\hyperelasticity"
    calib_folder = os.path.join(base_dir, "calibration")
    test_folder = os.path.join(base_dir, "test")
    invariants_folder = os.path.join(base_dir, "invariants")

    # 1. Manually Load Calibration Datasets
    calib_data = {
        "biaxial": load_hyperelastic_data(os.path.join(calib_folder, "biaxial.txt")),
        "pure_shear": load_hyperelastic_data(os.path.join(calib_folder, "pure_shear.txt")),
        "uniaxial": load_hyperelastic_data(os.path.join(calib_folder, "uniaxial.txt")),
    }

    # 2. Manually Load Test Datasets
    test_data = {
        "biax_test": load_hyperelastic_data(os.path.join(test_folder, "biax_test.txt")),
        "mixed_test": load_hyperelastic_data(os.path.join(test_folder, "mixed_test.txt")),
    }

    # 3. Manually Load Invariant References mapping explicitly to the exact case_names above
    inv_ref_dict = {
        # Calibration invariant references
        "biaxial": load_invariant_data(os.path.join(invariants_folder, "I_biaxial.txt")),
        "pure_shear": load_invariant_data(os.path.join(invariants_folder, "I_pure_shear.txt")),
        "uniaxial": load_invariant_data(os.path.join(invariants_folder, "I_uniaxial.txt")),
        # Test invariant references
        "biax_test": load_invariant_data(os.path.join(invariants_folder, "I_biax_test.txt")),
        "mixed_test": load_invariant_data(os.path.join(invariants_folder, "I_mixed_test.txt")),
    }

    # -----------------------------------------------------------------------
    # TASK 1.1: Plot raw F, P, W vs. Load Steps
    # -----------------------------------------------------------------------
    plot_task1_1_raw_data(calib_data, title_prefix="Calibration Dataset")
    plot_task1_1_raw_data(test_data, title_prefix="Test Dataset")

    # -----------------------------------------------------------------------
    # TASK 1.2: Compute and Verify Invariants, W, and P in JAX
    # -----------------------------------------------------------------------
    calib_preds = evaluate_task1_2(calib_data, inv_ref_dict, "Calibration Dataset")
    test_preds = evaluate_task1_2(test_data, inv_ref_dict, "Test Dataset")

    # Combine all 5 cases (3 Calibration + 2 Test) for Task 1.2 combined plots
    all_cases_true = {**calib_data, **test_data}
    all_cases_pred = {**calib_preds, **test_preds}

    # -----------------------------------------------------------------------
    # Plot 1: Calibration Data Invariants (JAX Computed vs. Invariant Folder)
    # Plot 2: Test Data Invariants (JAX Computed vs. Invariant Folder)
    # -----------------------------------------------------------------------
    plot_invariants_comparison(
        calib_preds, inv_ref_dict, title_prefix="Plot 1 - Calibration Data"
    )
    plot_invariants_comparison(
        test_preds, inv_ref_dict, title_prefix="Plot 2 - Test Data"
    )

    # Plot W (Computed vs. True) across all 5 cases in 1 plot
    plot_W_comparison(all_cases_true, all_cases_pred)

    # Plot P (Computed vs. True) across all 5 cases
    plot_P_comparison(all_cases_true, all_cases_pred)

    # Group 1: Biaxial vs Biax_test
    group_1 = {
        "biaxial": calib_data["biaxial"],
        "biax_test": test_data["biax_test"]
    }
    plot_grouped_raw_data(group_1, title="Plot 1: Biaxial Calibration vs. Biaxial Test")

    # Group 2: All pure modes + Mixed_test
    group_2 = {
        "biaxial": calib_data["biaxial"],
        "uniaxial": calib_data["uniaxial"],
        "pure_shear": calib_data["pure_shear"],
        "mixed_test": test_data["mixed_test"]
    }
    plot_grouped_raw_data(group_2, title="Plot 2: Calibration Modes vs. Mixed Test")


if __name__ == "__main__":
    main()