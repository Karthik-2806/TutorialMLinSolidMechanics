import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def compute_inplane_stretches(F: jnp.ndarray):
  """Computes in-plane principal stretches (lambda_1, lambda_2) from the 2x2 in-plane

  submatrix of C = F^T @ F.
  """
  C11 = F[:, 0, 0] ** 2 + F[:, 1, 0] ** 2 + F[:, 2, 0] ** 2
  C22 = F[:, 0, 1] ** 2 + F[:, 1, 1] ** 2 + F[:, 2, 1] ** 2
  C12 = (
      F[:, 0, 0] * F[:, 0, 1]
      + F[:, 1, 0] * F[:, 1, 1]
      + F[:, 2, 0] * F[:, 2, 1]
  )

  trace_half = 0.5 * (C11 + C22)
  diff_half = 0.5 * (C11 - C22)
  R = jnp.sqrt(jnp.maximum(diff_half**2 + C12**2, 1e-12))

  eig1 = trace_half + jnp.sign(diff_half + 1e-9) * R
  eig2 = trace_half - jnp.sign(diff_half + 1e-9) * R

  lambda_1 = jnp.sqrt(jnp.maximum(eig1, 1e-12))
  lambda_2 = jnp.sqrt(jnp.maximum(eig2, 1e-12))

  return lambda_1, lambda_2


def compute_equivalent_stress(F: jnp.ndarray, P: jnp.ndarray):
  """Computes the von Mises equivalent stress sigma_vM from F and P.

  sigma = (1 / det(F)) * P @ F^T
  s = sigma - (1/3) * tr(sigma) * I
  sigma_vM = sqrt( 1.5 * s : s )
  """
  # Jacobian J = det(F) of shape (N, 1, 1)
  J = jnp.linalg.det(F)[:, None, None]

  # Cauchy stress: sigma = (1/J) * P @ F^T, shape (N, 3, 3)
  sigma = (1.0 / J) * jnp.einsum("nij,nkj->nik", P, F)

  # Enforce exact numerical symmetry: sigma = 0.5 * (sigma + sigma^T)
  sigma = 0.5 * (sigma + jnp.swapaxes(sigma, -1, -2))

  # Hydrostatic pressure: p = (1/3) * tr(sigma)
  p = (1.0 / 3.0) * jnp.trace(sigma, axis1=1, axis2=2)[:, None, None]

  # Stress deviator: s = sigma - p * I
  I = jnp.eye(3)[None, :, :]
  s = sigma - p * I

  # von Mises equivalent stress: sigma_vM = sqrt(1.5 * s_ij * s_ij)
  s_contracted = jnp.einsum("nij,nij->n", s, s)
  sigma_vM = jnp.sqrt(jnp.maximum(1.5 * s_contracted, 1e-12))

  return sigma_vM, sigma


def plot_3d_constitutive_manifolds(data_dict: dict):
  """Plots:

  1. 3D Strain energy density landscape W(lambda_1, lambda_2)
  2. 3D von Mises equivalent stress manifold sigma_vM(lambda_1, lambda_2)
  """
  fig = plt.figure(figsize=(16, 7), layout="constrained")

  ax_w = fig.add_subplot(1, 2, 1, projection="3d")
  ax_p = fig.add_subplot(1, 2, 2, projection="3d")

  for case_name, tensors in data_dict.items():
    F = tensors["F"]
    P = tensors["P"]
    W = tensors["W"]

    # In-plane principal stretches
    lam1, lam2 = compute_inplane_stretches(F)

    # Calculate von Mises equivalent stress
    sigma_vM, _ = compute_equivalent_stress(F, P)

    l1_np = np.array(lam1)
    l2_np = np.array(lam2)
    w_np = np.array(W)
    svm_np = np.array(sigma_vM)

    # Plot 1: 3D Energy Landscape Trajectory
    ax_w.plot(l1_np, l2_np, w_np, linewidth=2.5, label=case_name)
    ax_w.plot(l1_np, l2_np, np.zeros_like(l1_np), linestyle=":", alpha=0.35)

    # Plot 2: 3D von Mises Equivalent Stress Manifold Trajectory
    ax_p.plot(l1_np, l2_np, svm_np, linewidth=2.5, label=case_name)
    ax_p.plot(l1_np, l2_np, np.zeros_like(l1_np), linestyle=":", alpha=0.35)

  # Reference state points (1.0, 1.0, 0.0)
  ax_w.scatter([1.0], [1.0], [0.0], color="black", s=40, label="Reference")
  ax_p.scatter([1.0], [1.0], [0.0], color="black", s=40, label="Reference")

  # Styling Energy Plot
  ax_w.set_title(
      r"Strain Energy Landscape $W(\lambda_1, \lambda_2)$",
      fontsize=13,
      fontweight="bold",
  )
  ax_w.set_xlabel(r"Principal Stretch $\lambda_1$ [-]")
  ax_w.set_ylabel(r"Principal Stretch $\lambda_2$ [-]")
  ax_w.set_zlabel(r"Energy Density $W$ [$\mathrm{MJ/m^3}$]")
  ax_w.legend(fontsize=9)
  ax_w.view_init(elev=28, azim=-125)

  # Styling Stress Plot
  ax_p.set_title(
      r"Equivalent Stress Manifold $\sigma_{\mathrm{vM}}(\lambda_1, \lambda_2)$",
      fontsize=13,
      fontweight="bold",
  )
  ax_p.set_xlabel(r"Principal Stretch $\lambda_1$ [-]")
  ax_p.set_ylabel(r"Principal Stretch $\lambda_2$ [-]")
  ax_p.set_zlabel(r"von Mises Stress $\sigma_{\mathrm{vM}}$ [MPa]")
  ax_p.legend(fontsize=9)
  ax_p.view_init(elev=28, azim=-125)

  plt.show()


def plot_stretch_trajectories(
    data_dict: dict, title_prefix: str = "Calibration Data"
):
  """Plots a separate (lambda_1 vs lambda_2) stretch trajectory for each load case

  side-by-side, including analytical kinematics and the reference point (1, 1).
  """
  n_cases = len(data_dict)
  fig, axes = plt.subplots(
      1, n_cases, figsize=(5.2 * n_cases, 4.8), layout="constrained"
  )

  if n_cases == 1:
    axes = [axes]

  lam_ref = np.linspace(0.45, 2.05, 300)

  for ax, (case_name, tensors) in zip(axes, data_dict.items()):
    F = tensors["F"]
    lam1, lam2 = compute_inplane_stretches(F)

    l1 = np.array(lam1)
    l2 = np.array(lam2)

    ax.plot(
        l1,
        l2,
        color="crimson",
        linewidth=2.5,
        label=rf"Data: {case_name}",
        zorder=4,
    )
    ax.scatter(
        [l1[0], l1[-1]],
        [l2[0], l2[-1]],
        color="crimson",
        s=30,
        zorder=5,
        label="Start / End",
    )

    if "uniaxial" in case_name:
      ax.plot(
          lam_ref,
          lam_ref ** (-0.5),
          "k--",
          alpha=0.6,
          label=r"Theory: $\lambda_2 = \lambda_1^{-1/2}$",
      )
    elif "biaxial" in case_name:
      ax.plot(
          lam_ref,
          lam_ref,
          "k--",
          alpha=0.6,
          label=r"Theory: $\lambda_2 = \lambda_1$",
      )
    elif "pure_shear" in case_name:
      ax.plot(
          lam_ref,
          1.0 / lam_ref,
          "k--",
          alpha=0.6,
          label=r"Theory: $\lambda_2 = 1/\lambda_1$",
      )
    else:
      ax.plot(
          lam_ref,
          lam_ref ** (-0.5),
          "gray",
          linestyle=":",
          alpha=0.5,
          label=r"Uniaxial bound",
      )
      ax.plot(
          lam_ref,
          lam_ref,
          "gray",
          linestyle="--",
          alpha=0.5,
          label=r"Biaxial bound",
      )

    ax.scatter(
        [1.0],
        [1.0],
        color="black",
        s=60,
        zorder=6,
        label=r"Undeformed $(1, 1)$",
    )
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_title(f"Load Case: {case_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Principal Stretch $\lambda_1$ [-]", fontsize=10)
    ax.set_ylabel(r"Principal Stretch $\lambda_2$ [-]", fontsize=10)
    ax.set_xlim(0.4, 2.1)
    ax.set_ylim(0.4, 2.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper right")

  fig.suptitle(
      f"{title_prefix}: In-Plane Principal Stretch Paths"
      r" $(\lambda_1\text{--}\lambda_2)$",
      fontsize=14,
      fontweight="bold",
  )
  plt.show()

def plot_3d_calibration_vs_test(calib_dict: dict, test_dict: dict):
  """Plots Calibration and Test datasets together on 3D constitutive manifolds

  without duplicate legend entries.
  """
  fig = plt.figure(figsize=(16, 7.5), layout="constrained")

  ax_w = fig.add_subplot(1, 2, 1, projection="3d")
  ax_p = fig.add_subplot(1, 2, 2, projection="3d")

  # ---------------------------------------------------------
  # 1. Calibration Data Loop (Run once for calib files)
  # ---------------------------------------------------------
  for case_name, tensors in calib_dict.items():
    F = tensors["F"]
    P = tensors["P"]
    W = tensors["W"]

    lam1, lam2 = compute_inplane_stretches(F)
    sigma_vM, _ = compute_equivalent_stress(F, P)

    l1_np = np.array(lam1)
    l2_np = np.array(lam2)
    w_np = np.array(W)
    svm_np = np.array(sigma_vM)

    # 3D curve (has label)
    ax_w.plot(
        l1_np,
        l2_np,
        w_np,
        linewidth=2.5,
        alpha=0.85,
        label=f"Calib: {case_name}",
    )
    # Ground projection (no label)
    ax_w.plot(l1_np, l2_np, np.zeros_like(l1_np), linestyle=":", alpha=0.25)

    ax_p.plot(
        l1_np,
        l2_np,
        svm_np,
        linewidth=2.5,
        alpha=0.85,
        label=f"Calib: {case_name}",
    )
    ax_p.plot(l1_np, l2_np, np.zeros_like(l1_np), linestyle=":", alpha=0.25)

  # ---------------------------------------------------------
  # 2. Test Data Loop (OUTSIDE the calibration loop)
  # ---------------------------------------------------------
  for case_name, tensors in test_dict.items():
    F = tensors["F"]
    P = tensors["P"]
    W = tensors["W"]

    lam1, lam2 = compute_inplane_stretches(F)
    sigma_vM, _ = compute_equivalent_stress(F, P)

    l1_np = np.array(lam1)
    l2_np = np.array(lam2)
    w_np = np.array(W)
    svm_np = np.array(sigma_vM)

    # Styling per test case
    if "biax" in case_name:
      line_style = "--"  # Dashed for biax_test
      line_color = "darkorange"
    elif "mixed" in case_name:
      line_style = ":"  # Dotted for mixed_test
      line_color = "crimson"
    else:
      line_style = "-."
      line_color = "red"

    # 3D curve (has label)
    ax_w.plot(
        l1_np,
        l2_np,
        w_np,
        linewidth=2.8,
        linestyle=line_style,
        color=line_color,
        label=f"Test: {case_name}",
        zorder=10,
    )
    # Ground projection (no label)
    ax_w.plot(
        l1_np,
        l2_np,
        np.zeros_like(l1_np),
        linestyle=line_style,
        color=line_color,
        alpha=0.45,
    )

    ax_p.plot(
        l1_np,
        l2_np,
        svm_np,
        linewidth=2.8,
        linestyle=line_style,
        color=line_color,
        label=f"Test: {case_name}",
        zorder=10,
    )
    ax_p.plot(
        l1_np,
        l2_np,
        np.zeros_like(l1_np),
        linestyle=line_style,
        color=line_color,
        alpha=0.45,
    )

  # Reference point (1.0, 1.0, 0.0)
  ax_w.scatter([1.0], [1.0], [0.0], color="black", s=50, label="Reference")
  ax_p.scatter([1.0], [1.0], [0.0], color="black", s=50, label="Reference")

  # Axes Styling
  ax_w.set_title(
      r"Energy Landscape: Calibration vs. Test Data",
      fontsize=13,
      fontweight="bold",
  )
  ax_w.set_xlabel(r"$\lambda_1$ [-]", fontsize=10)
  ax_w.set_ylabel(r"$\lambda_2$ [-]", fontsize=10)
  ax_w.set_zlabel(r"$W$ [$\mathrm{MJ/m^3}$]", fontsize=10)
  ax_w.legend(fontsize=9, loc="upper left")
  ax_w.view_init(elev=28, azim=-125)

  ax_p.set_title(
      r"Stress Manifold: Calibration vs. Test Data",
      fontsize=13,
      fontweight="bold",
  )
  ax_p.set_xlabel(r"$\lambda_1$ [-]", fontsize=10)
  ax_p.set_ylabel(r"$\lambda_2$ [-]", fontsize=10)
  ax_p.set_zlabel(r"$\sigma_{\mathrm{vM}}$ [MPa]", fontsize=10)
  ax_p.legend(fontsize=9, loc="upper left")
  ax_p.view_init(elev=28, azim=-125)

  plt.show()

def plot_vonmises_stress_comparison(data_dict: dict, pred_dict: dict, title_prefix: str = "Calibration"):
    """
    Compares true vs. JAX-predicted von Mises equivalent stresses.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing ground truth tensors {'F', 'P', 'W'} per case.
    pred_dict : dict
        Dictionary containing {'P_pred': ...} per case.
    title_prefix : str
        Dataset label ('Calibration' or 'Test').
    """
    n_cases = len(data_dict)
    fig, axes = plt.subplots(
        2, n_cases, 
        figsize=(5.5 * n_cases, 7.5), 
        gridspec_kw={"height_ratios": [3, 1.2]},
        layout="constrained"
    )

    if n_cases == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col_idx, (case_name, tensors) in enumerate(data_dict.items()):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        F = tensors["F"]
        P_true = tensors["P"]
        P_pred = pred_dict[case_name]["P_pred"]

        # 1. Kinematics and von Mises stresses
        lam1, _ = compute_inplane_stretches(F)
        sigma_vM_true, _ = compute_equivalent_stress(F, P_true)
        sigma_vM_pred, _ = compute_equivalent_stress(F, P_pred)

        l1 = np.array(lam1)
        svm_true = np.array(sigma_vM_true)
        svm_pred = np.array(sigma_vM_pred)
        diff_svm = np.abs(svm_pred - svm_true)

        # 2. Top plot: Stress comparison
        ax_top.plot(l1, svm_true, color="black", linewidth=2.5, label=r"True $\sigma_{\mathrm{vM}}$")
        ax_top.plot(
            l1, svm_pred, 
            color="crimson", linestyle="--", linewidth=2.0, 
            label=r"JAX $\partial W/\partial\boldsymbol{F}$"
        )
        ax_top.scatter([1.0], [0.0], color="black", s=40, zorder=5, label="Undeformed")

        ax_top.set_title(f"Case: {case_name}", fontsize=12, fontweight="bold")
        ax_top.set_ylabel(r"$\sigma_{\mathrm{vM}}$ [MPa]", fontsize=10)
        ax_top.grid(True, linestyle="--", alpha=0.5)
        ax_top.legend(fontsize=9, loc="upper left")

        # 3. Bottom plot: Absolute discrepancy
        ax_bot.plot(l1, diff_svm, color="royalblue", linewidth=1.8, label=r"$|\Delta \sigma_{\mathrm{vM}}|$")
        ax_bot.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax_bot.set_xlabel(r"Principal Stretch $\lambda_1$ [-]", fontsize=10)
        ax_bot.set_ylabel(r"$|\Delta \sigma_{\mathrm{vM}}|$ [MPa]", fontsize=9)
        ax_bot.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_bot.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"{title_prefix} Dataset: von Mises Stress Validation ($\sigma_{{\mathrm{{vM}}}}^{{\mathrm{{true}}}}$ vs. $\sigma_{{\mathrm{{vM}}}}^{{\mathrm{{pred}}}}$)",
        fontsize=14,
        fontweight="bold"
    )
    plt.show()