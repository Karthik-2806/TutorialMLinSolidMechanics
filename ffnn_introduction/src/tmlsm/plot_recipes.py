import matplotlib.pyplot as plt
import numpy as np

def add_origin_probe(ax, z_val_at_center):
    """Marks the (0, 0, 0) origin and draws a vertical probe to the surface center."""
    # 1. Plot a black marker at the true coordinate origin (0, 0, 0)
    ax.scatter(
        [0.0], [0.0], [0.0],
        color="black",
        s=60,
        marker="o",
        zorder=10,
        label="Origin (0,0,0)",
    )
    ax.text(0.3, 0.3, 0.0, "(0,0,0)", color="black", fontsize=9, fontweight="bold")

    # 2. Plot a red probe marker on the surface at (0, 0, z_center)
    ax.scatter(
        [0.0], [0.0], [z_val_at_center],
        color="red",
        s=60,
        marker="x",
        zorder=10,
        label=f"Surface (0,0,{z_val_at_center:.2f})",
    )

    # 3. Draw a vertical dashed probe line from (0, 0, 0) to (0, 0, z_center)
    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, z_val_at_center],
        color="red",
        linestyle="--",
        linewidth=2.0,
        zorder=9,
    )
    ax.legend(loc="upper left", fontsize=8)


def plot_evaluation(coords, y_true, y_pred, grad_true, grad_pred, strategy_name="model"):
    """Plots 3D surfaces with origin probes and failing target regions."""
    X = coords[:, 0].reshape(20, 20)
    Y = coords[:, 1].reshape(20, 20)
    Z_true = np.array(y_true).reshape(20, 20)
    Z_pred = np.array(y_pred).reshape(20, 20)
    error_threshold = 0.5

    # Lock Z-limits so all plots share the exact same vertical reference
    z_min = min(float(np.min(Z_true)), float(np.min(Z_pred)), 0.0)
    z_max = max(float(np.max(Z_true)), float(np.max(Z_pred))) * 1.05

    # Estimate surface values at (0, 0) using the middle grid indices (9, 10)
    z_true_origin = float(np.mean(Z_true[9:11, 9:11]))
    z_pred_origin = float(np.mean(Z_pred[9:11, 9:11]))

    fig = plt.figure(figsize=(16, 5))

    # --- Plot 1: 3D Surface Ground Truth ---
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.65)
    add_origin_probe(ax1, z_true_origin)
    ax1.set_title("True Surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_zlim(z_min, z_max)

    # --- Plot 2: 3D Surface Prediction ---
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X, Y, Z_pred, cmap="plasma", alpha=0.65)
    add_origin_probe(ax2, z_pred_origin)
    ax2.set_title(f"Predicted Surface\n({strategy_name})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_zlim(z_min, z_max)

    # --- Plot 3: True Surface ONLY at Mismatched Points ---
    abs_error = np.abs(Z_pred - Z_true)
    Z_mismatched = np.where(abs_error > error_threshold, Z_true, np.nan)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(X, Y, Z_mismatched, color="crimson", alpha=0.8, edgecolor="darkred")
    #add_origin_probe(ax3, 0.0)
    ax3.set_title(f"3. Target Surface in Failing Regions\n(|Error| > {error_threshold})")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.show()

def plot_results(
    eps_all,
    sigma_all,
    eps_train,
    sigma_train,
    eps_dense,
    W_pred,
    sigma_pred,
    model_name,
    task_mode,
    split_boundary=None,
):
    """Plots stress-strain curve and learned strain-energy potential with automatic x-axis scaling."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Convert to flat numpy arrays
    eps_all_flat = np.array(eps_all).ravel()
    sigma_all_flat = np.array(sigma_all).ravel()
    eps_train_flat = np.array(eps_train).ravel()
    sigma_train_flat = np.array(sigma_train).ravel()
    eps_dense_flat = np.array(eps_dense).ravel()

    # Determine split boundary automatically from training data if not provided
    if split_boundary is None:
        split_boundary = float(np.max(eps_train_flat))

    # --- 1. Compute Automatic X-Axis Limits across all data ---
    min_x = min(float(np.min(eps_all_flat)), float(np.min(eps_dense_flat)))
    max_x = max(float(np.max(eps_all_flat)), float(np.max(eps_dense_flat)))
    padding = (max_x - min_x) * 0.05
    x_limits = (min_x - padding, max_x + padding)

    # --- Subplot 1: Stress vs Strain (Constitutive response) ---
    if task_mode == "extrapolation":
        ax.scatter(
            eps_train_flat,
            sigma_train_flat,
            color="black",
            s=25,
            label=f"Training Data (eps <= {split_boundary:.1f})",
            zorder=5,
        )
        # Unseen test points above the boundary
        test_mask = eps_all_flat > split_boundary
        if np.any(test_mask):
            ax.scatter(
                eps_all_flat[test_mask],
                sigma_all_flat[test_mask],
                facecolors="none",
                edgecolors="blue",
                s=35,
                label=f"Unseen Test Data (eps > {split_boundary:.1f})",
                zorder=5,
            )
        ax.axvline(
            split_boundary,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Extrapolation Boundary",
        )
    else:
        ax.scatter(
            eps_all_flat,
            sigma_all_flat,
            color="black",
            s=25,
            label="All Experimental Points",
            zorder=5,
        )

    # Predicted stress curve
    ax.plot(
        eps_dense_flat,
        np.array(sigma_pred).ravel(),
        color="crimson",
        linewidth=2.5,
        label=f"{model_name} Predicted Stress",
    )
    ax.set_title(
        f"Stress vs Strain (Task: {task_mode.capitalize()}, Model: {model_name})",
        fontsize=11,
    )
    ax.set_xlabel(r"Tensile Strain $\varepsilon$ [-]")
    ax.set_ylabel(r"Tensile Stress $\sigma$ [MPa]")
    ax.set_xlim(x_limits)  # <--- Automatically adjusted x-axis
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.show()