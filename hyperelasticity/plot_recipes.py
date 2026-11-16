import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_grouped_raw_data(group_dict: dict, title: str):
    """
    Plots F, P, and W for a specific group of load cases on a single set of shared axes.
    Calibration cases are solid lines; Test cases are thick dashed lines.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), layout="constrained")
    ax_F, ax_P, ax_W = axes

    comp_colors = plt.cm.tab10(np.linspace(0, 1, 9))
    comp_labels = [
        (0, 0, "11"), (0, 1, "12"), (0, 2, "13"),
        (1, 0, "21"), (1, 1, "22"), (1, 2, "23"),
        (2, 0, "31"), (2, 1, "32"), (2, 2, "33"),
    ]

    dataset_handles = []
    colors_W = plt.cm.tab10(np.linspace(0, 1, len(group_dict)))

    for col, (case_name, data) in enumerate(group_dict.items()):
        F = np.array(data["F"])
        P = np.array(data["P"])
        W = np.array(data["W"])
        
        # Using raw load step index for the x-axis
        steps = np.arange(len(W))

        # Distinct styling for Test vs Calibration
        is_test = "test" in case_name.lower()
        
        # ":" creates a dotted line, "-" creates a solid line
        ls = ":" if is_test else "-"       
        
        lw = 2.5 if is_test else 1.5
        alpha = 1.0  # 1.0 keeps all lines 100% bright/opaque
        
        # Add to custom dataset legend
        dataset_handles.append(
            mlines.Line2D([], [], color='black', linestyle=ls, linewidth=lw, alpha=alpha, label=case_name)
        )
        
        # Add to custom dataset legend
        dataset_handles.append(
            mlines.Line2D([], [], color='black', linestyle=ls, linewidth=lw, alpha=alpha, label=case_name)
        )

        # Plot F and P (using the 9 component colors)
        for idx, (i, j, lbl) in enumerate(comp_labels):
            ax_F.plot(steps, F[:, i, j], color=comp_colors[idx], linestyle=ls, linewidth=lw, alpha=alpha)
            ax_P.plot(steps, P[:, i, j], color=comp_colors[idx], linestyle=ls, linewidth=lw, alpha=alpha)
            
        # Plot W (using distinct colors per dataset)
        ax_W.plot(steps, W, color=colors_W[col], linestyle=ls, linewidth=lw, alpha=alpha, label=case_name)

    # Build custom legend for the 9 components
    comp_handles = [
        mlines.Line2D([], [], color=comp_colors[idx], linestyle='-', linewidth=2, label=rf"$_{{{lbl}}}$") 
        for idx, (_, _, lbl) in enumerate(comp_labels)
    ]

    # Format F subplot
    ax_F.set_title(f"Deformation Gradient F", fontsize=12, fontweight="bold")
    ax_F.set_ylabel(r"Deformation $F_{ij}$", fontsize=10)
    ax_F.grid(True, linestyle="--", alpha=0.4)
    ax_F.legend(handles=dataset_handles + comp_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=2, title="Datasets & Components")

    # Format P subplot
    ax_P.set_title(f"First Piola-Kirchhoff Stress P", fontsize=12, fontweight="bold")
    ax_P.set_ylabel(r"Stress $P_{ij}$", fontsize=10)
    ax_P.grid(True, linestyle="--", alpha=0.4)
    ax_P.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax_P.legend(handles=dataset_handles + comp_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=2)

    # Format W subplot
    ax_W.set_title(f"Strain Energy Density W", fontsize=12, fontweight="bold")
    ax_W.set_xlabel("Load Step / Time Step", fontsize=10)
    ax_W.set_ylabel(r"Energy $W$", fontsize=10)
    ax_W.grid(True, linestyle="--", alpha=0.4)
    ax_W.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Datasets")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.show()

# ===========================================================================
# TASK 1.1: Raw Data Visualization (F, P, W vs. Load Steps)
# ===========================================================================
def plot_task1_1_raw_data(dataset_dict: dict, title_prefix: str = "Calibration Data"):
    """
    Task 1.1: Plots F components, P components, and W scalar vs. Load Steps
    in a 3-row x N_cases-column grid. No von Mises or principal stretches.
    """
    n_cases = len(dataset_dict)
    fig, axes = plt.subplots(
        3, n_cases,
        figsize=(5.2 * n_cases, 10.5),
        layout="constrained",
        sharex="col"
    )

    if n_cases == 1:
        axes = np.expand_dims(axes, axis=1)

    comp_colors = plt.cm.tab10(np.linspace(0, 1, 9))
    comp_labels = [
        (0, 0, "11"), (0, 1, "12"), (0, 2, "13"),
        (1, 0, "21"), (1, 1, "22"), (1, 2, "23"),
        (2, 0, "31"), (2, 1, "32"), (2, 2, "33"),
    ]

    for col, (case_name, data) in enumerate(dataset_dict.items()):
        F = np.array(data["F"])  # (N, 3, 3)
        P = np.array(data["P"])  # (N, 3, 3)
        W = np.array(data["W"])  # (N,)
        steps = np.arange(len(W))

        ax_F = axes[0, col]
        ax_P = axes[1, col]
        ax_W = axes[2, col]

        # Row 0: Deformation Gradient F_ij vs Load Step
        for idx, (i, j, lbl) in enumerate(comp_labels):
            ax_F.plot(
                steps, F[:, i, j],
                color=comp_colors[idx], linewidth=1.8,
                marker="o", markersize=3, markevery=max(1, len(steps) // 15),
                label=rf"$F_{{{lbl}}}$"
            )
        ax_F.set_title(f"Case: {case_name}", fontsize=12, fontweight="bold")
        ax_F.set_ylabel(r"Deformation Gradient $F_{ij}$ [-]", fontsize=10)
        ax_F.grid(True, linestyle="--", alpha=0.4)
        if col == n_cases - 1:
            ax_F.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)

        # Row 1: First Piola-Kirchhoff Stress P_ij vs Load Step
        for idx, (i, j, lbl) in enumerate(comp_labels):
            ax_P.plot(
                steps, P[:, i, j],
                color=comp_colors[idx], linewidth=1.8,
                marker="o", markersize=3, markevery=max(1, len(steps) // 15),
                label=rf"$P_{{{lbl}}}$"
            )
        ax_P.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax_P.set_ylabel(r"PK1 Stress $P_{ij}$", fontsize=10)
        ax_P.grid(True, linestyle="--", alpha=0.4)
        if col == n_cases - 1:
            ax_P.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)

        # Row 2: Strain Energy Density W vs Load Step
        ax_W.plot(
            steps, W,
            color="crimson", linewidth=2.2,
            marker="o", markersize=3.5, markevery=max(1, len(steps) // 15),
            label=r"$W$"
        )
        ax_W.set_xlabel("Load Step / Time Step", fontsize=10)
        ax_W.set_ylabel(r"Strain Energy $W$", fontsize=10)
        ax_W.grid(True, linestyle="--", alpha=0.4)
        ax_W.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Task 1.1 - {title_prefix}: F, P, and W vs. Load Steps", fontsize=14, fontweight="bold")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================
# TASK 1.2 (Part 1): Invariants Validation (Plot 1: Calibration, Plot 2: Test)
# ===========================================================================
def plot_invariants_comparison(
    preds_dict: dict,
    inv_ref_dict: dict,
    title_prefix: str = "Calibration Data",
):
    """
    Task 1.2: Validates JAX-computed invariants against reference files in the
    'invariants' folder. Both True (solid) and Computed (dashed) are plotted.
    """
    n_cases = len(preds_dict)
    fig, axes = plt.subplots(
        1, n_cases,
        figsize=(5.5 * n_cases, 4.8),
        layout="constrained",
        sharey=False
    )

    if n_cases == 1:
        axes = [axes]

    inv_specs = [
        (0, r"$I_1$", "#1f77b4"),  # Blue
        (1, r"$J$",   "#2ca02c"),  # Green
        (2, r"$I_4$", "#d62728"),  # Red
        (3, r"$I_5$", "#9467bd"),  # Purple
    ]

    for ax, (case_name, pred_data) in zip(axes, preds_dict.items()):
        inv_comp = np.array(pred_data["inv_pred"])  # shape (N, 4): [I1, J, I4, I5]
        steps_comp = np.arange(len(inv_comp))

        # --- DIRECT MATCHING LOGIC ---
        # Grabs the explicitly assigned array from your manually loaded dictionary
        inv_true = inv_ref_dict.get(case_name)
        if inv_true is None:
            print(f"[Plot Warning] No invariant reference found for '{case_name}'.")

        # --- PLOTTING ---
        for idx, lbl, color in inv_specs:
            # 1. Plot Reference Invariant from folder (Thick semi-transparent solid line)
            if inv_true is not None and inv_true.shape[-1] > idx:
                # FIX: Independent x-axis prevents the (99,) vs (199,) dimension crash!
                steps_true = np.arange(len(inv_true)) 
                ax.plot(
                    steps_true, inv_true[:, idx],
                    color=color, linestyle="-", linewidth=3.5, alpha=0.4,
                    label=f"{lbl} (Folder True)"
                )

            # 2. Plot JAX Computed Invariant (Dashed line with circle markers)
            ax.plot(
                steps_comp, inv_comp[:, idx],
                color=color, linestyle="--", linewidth=1.6,
                marker="o", markersize=4, markevery=max(1, len(steps_comp) // 12),
                label=f"{lbl} (JAX Computed)"
            )

        ax.set_title(f"Case: {case_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Load Step / Time Step", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r"Invariants ($I_1, J, I_4, I_5$)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)

        if ax is axes[-1]:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5)

    fig.suptitle(
        f"Task 1.2 - {title_prefix}: Invariants Validation",
        fontsize=13,
        fontweight="bold"
    )
    plt.show()
# ===========================================================================
# TASK 1.2 (Part 2): 1 Combined Plot for Strain Energy W Across All 5 Cases
# ===========================================================================
def plot_W_comparison(all_cases_true: dict, all_cases_pred: dict):
    """
    Task 1.2: 1 combined plot showing W_true vs. W_computed across load steps
    for all 5 cases (3 calibration + 2 test).
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    case_colors = plt.cm.tab10(np.linspace(0, 1, len(all_cases_true)))

    for color, (case_name, true_data) in zip(case_colors, all_cases_true.items()):
        W_true = np.array(true_data["W"])
        W_comp = np.array(all_cases_pred[case_name]["W_pred"])
        steps = np.arange(len(W_true))

        # True W from dataset (Solid line)
        ax.plot(
            steps, W_true,
            color=color, linestyle="-", linewidth=2.4, alpha=0.85,
            label=f"{case_name} (True $W$)"
        )
        # Computed W from JAX (Dashed line with circle markers)
        ax.plot(
            steps, W_comp,
            color=color, linestyle="--", linewidth=1.8,
            marker="o", markersize=4, markevery=max(1, len(steps) // 12),
            label=f"{case_name} (Computed $W$)"
        )

    ax.set_title("Task 1.2: Strain Energy Density $W$ (Computed vs. True) Across All 5 Load Cases", fontsize=13, fontweight="bold")
    ax.set_xlabel("Load Step / Time Step", fontsize=11)
    ax.set_ylabel(r"Strain Energy Density $W$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.show()


# ===========================================================================
# TASK 1.2 (Part 3): Stress P (Computed grad W vs. True) Across All 5 Cases
# ===========================================================================
def plot_P_comparison(all_cases_true: dict, all_cases_pred: dict):
    """
    Task 1.2: Plots P_true vs. P_computed (from jax.grad(W)) vs. Load Step
    across all 5 load cases. Plots ALL 9 components of the stress tensor.
    """
    n_cases = len(all_cases_true)
    fig, axes = plt.subplots(
        1, n_cases,
        figsize=(5.5 * n_cases, 5.0), # slightly wider to fit the 2-column legend
        layout="constrained",
        sharey=False
    )

    if n_cases == 1:
        axes = [axes]

    # Generate 9 distinct colors for the 9 stress components
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    
    comp_specs = [
        (0, 0, "P_{11}", colors[0]),
        (0, 1, "P_{12}", colors[1]),
        (0, 2, "P_{13}", colors[2]),
        (1, 0, "P_{21}", colors[3]),
        (1, 1, "P_{22}", colors[4]),
        (1, 2, "P_{23}", colors[5]),
        (2, 0, "P_{31}", colors[6]),
        (2, 1, "P_{32}", colors[7]),
        (2, 2, "P_{33}", colors[8]),
    ]

    for ax, (case_name, true_data) in zip(axes, all_cases_true.items()):
        P_true = np.array(true_data["P"])
        P_comp = np.array(all_cases_pred[case_name]["P_pred"])
        steps = np.arange(len(P_true))

        for i, j, lbl, color in comp_specs:
            # True Stress (Solid semi-transparent line)
            ax.plot(
                steps, P_true[:, i, j],
                color=color, linestyle="-", linewidth=3.0, alpha=0.4,
                label=rf"${lbl}$ True"
            )
            # JAX Computed Stress (Dashed line + circle markers)
            ax.plot(
                steps, P_comp[:, i, j],
                color=color, linestyle="--", linewidth=1.6,
                marker="o", markersize=3.5, markevery=max(1, len(steps) // 12),
                label=rf"${lbl}$ JAX"
            )

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Case: {case_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Load Step / Time Step", fontsize=10)
        
        if ax is axes[0]:
            ax.set_ylabel(r"First Piola-Kirchhoff Stress $P_{ij}$", fontsize=10)
            
        ax.grid(True, linestyle="--", alpha=0.45)
        
        if ax is axes[-1]:
            # Use ncol=2 so the 18 legend entries don't run off the screen
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, ncol=2)

    fig.suptitle(
        r"Task 1.2: First Piola-Kirchhoff Stress $\mathbf{P} = \partial W / \partial \mathbf{F}$ (All 9 Components)",
        fontsize=13,
        fontweight="bold"
    )
    plt.show()

def plot_stress_parity(dataset_dict: dict, pred_dict: dict, title_prefix: str = "Parity"):
    """
    Creates a separate standalone parity plot figure for each test case.
    Within each plot, the 9 components of the Piola-Kirchhoff stress tensor 
    are plotted with distinct colors.
    """
    component_labels = [
        "P_{11}", "P_{12}", "P_{13}",
        "P_{21}", "P_{22}", "P_{23}",
        "P_{31}", "P_{32}", "P_{33}"
    ]
    # Get 9 visually distinct colors from the tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, 9)) 
    
    for case_name, data in dataset_dict.items():
        # Create a new figure for each load case
        fig, ax = plt.subplots(figsize=(8, 5))
        
        P_true = np.array(data["P"])  # shape (N, 3, 3)
        P_pred = np.array(pred_dict[case_name]["P_pred"])  # shape (N, 3, 3)
        
        # Calculate dynamic bounds for the current case
        case_min = min(np.min(P_true), np.min(P_pred))
        case_max = max(np.max(P_true), np.max(P_pred))
        margin = (case_max - case_min) * 0.05
        
        # Draw the y = x perfect fit reference line
        ax.plot([case_min - margin, case_max + margin], 
                [case_min - margin, case_max + margin], 
                color='gray', linestyle='--', zorder=0, label="Perfect Fit (y=x)")
        
        # Loop through the 3x3 tensor and plot each component separately
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                ax.scatter(
                    P_true[:, i, j], 
                    P_pred[:, i, j], 
                    alpha=0.7, 
                    s=25, 
                    color=colors[idx], 
                    label=f"${component_labels[idx]}$"
                )
        
        ax.set_xlim(case_min - margin, case_max + margin)
        ax.set_ylim(case_min - margin, case_max + margin)
        ax.set_xlabel("True Stress Components $P_{true}$")
        ax.set_ylabel("Predicted Stress Components $P_{pred}$")
        ax.set_title(f"{title_prefix}: {case_name}")
        
        # Place the legend outside the plot area to prevent overlapping data points
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def plot_componentwise_stress(dataset_dict: dict, pred_dict: dict, title_prefix: str = ""):
    """
    Plots the primary normal stress components (P11, P22, P33) against the 
    data sample index to directly compare the numerical fit over the load paths.
    """
    num_cases = len(dataset_dict)
    fig, axes = plt.subplots(nrows=num_cases, ncols=1, figsize=(10, 3 * num_cases), sharex=False)
    
    if num_cases == 1:
        axes = [axes]

    for ax, (case_name, data) in zip(axes, dataset_dict.items()):
        P_true = np.array(data["P"])
        P_pred = np.array(pred_dict[case_name]["P_pred"])
        
        samples = np.arange(P_true.shape[0])
        
        # Plot True Normal Stresses (Solid lines)
        ax.plot(samples, P_true[:, 0, 0], 'b-', linewidth=2, label="$P_{11}$ True")
        ax.plot(samples, P_true[:, 1, 1], 'g-', linewidth=2, label="$P_{22}$ True")
        ax.plot(samples, P_true[:, 2, 2], 'r-', linewidth=2, label="$P_{33}$ True")
        
        # Plot Predicted Normal Stresses (Dashed lines)
        ax.plot(samples, P_pred[:, 0, 0], 'b--', linewidth=2, label="$P_{11}$ Pred")
        ax.plot(samples, P_pred[:, 1, 1], 'g--', linewidth=2, label="$P_{22}$ Pred")
        ax.plot(samples, P_pred[:, 2, 2], 'r--', linewidth=2, label="$P_{33}$ Pred")
        
        ax.set_title(f"{title_prefix} - {case_name}")
        ax.set_xlabel("Load Step / Data Index")
        ax.set_ylabel("Stress Magnitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np


