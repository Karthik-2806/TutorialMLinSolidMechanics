"""Verification of Thermodynamic Consistency for Hyperelastic NN Models."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import klax

import models as tm


# ---------------------------------------------------------------------------
# Test 1: Closed Cycle Energy Generation / Dissipation (Path Integral)
# ---------------------------------------------------------------------------
def check_closed_cycle_work(model_fn, n_steps: int = 500):
    """
    Simulates a closed elliptical loading-unloading deformation loop:
    F(t) starts at I, deforms biaxially, and returns exactly to I.
    Computes net cycle work: W_net = oint P : dF
    """
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_steps)

    # Elliptical closed trajectory in stretch space centered around stretch = 1.2
    # Returns exactly to the starting state at theta = 2*pi
    f11 = 1.2 + 0.3 * jnp.cos(theta) - 0.3
    f22 = 1.2 + 0.2 * jnp.sin(theta)
    f33 = 1.0 / (f11 * f22)  # Incompressible / near-isochoric loop

    # Stack into batch of (N, 3, 3) deformation gradients
    zeros = jnp.zeros_like(theta)
    F_loop = jnp.stack([
        f11,   zeros, zeros,
        zeros, f22,   zeros,
        zeros, zeros, f33
    ], axis=-1).reshape(-1, 3, 3)

    # Evaluate stresses along the closed path
    batched_model = jax.vmap(model_fn)
    P_loop = batched_model(F_loop)  # shape (N, 3, 3)

    # Compute dF between consecutive steps: dF = F_{k+1} - F_k
    dF = jnp.diff(F_loop, axis=0)  # shape (N-1, 3, 3)
    P_mid = 0.5 * (P_loop[:-1] + P_loop[1:])  # midpoint stress

    # Incremental work: dW = P : dF = sum(P_ij * dF_ij)
    dW = jnp.sum(P_mid * dF, axis=(1, 2))

    # Net work around the closed cycle
    net_work = float(jnp.sum(dW))
    work_accumulated = np.cumsum(np.array(dW))

    return theta, F_loop, P_loop, work_accumulated, net_work


# ---------------------------------------------------------------------------
# Execution Function
# ---------------------------------------------------------------------------
def run_thermodynamic_assessment(model_):
    print("=" * 75)
    print("THERMODYNAMIC CONSISTENCY NUMERICAL EVALUATION")
    print("=" * 75)

    # 1. Closed Cycle Work
    theta, F_loop, P_loop, work_acc, net_work = check_closed_cycle_work(model_)
    print(f"\nClosed Deformation Loop Path Integral:")
    print(f"  Net Cycle Work (oint P : dF)               = {net_work:.4e} [J/m^3]")

    if abs(net_work) > 1e-3:
        status = "VIOLATED (Energy is created or dissipated over closed elastic loop)"
    else:
        status = "FULFILLED (Conservative potential exists)"
    print(f"  Thermodynamic Consistency Status           = {status}")
    print("=" * 75)

    # Plot cycle work accumulated vs path step
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # Plot closed path in deformation space
    ax1.plot(np.array(F_loop[:, 0, 0]), np.array(F_loop[:, 1, 1]), 'b-', linewidth=2.0)
    ax1.scatter([F_loop[0, 0, 0]], [F_loop[0, 1, 1]], color='red', s=40, zorder=5, label="Start / End")
    ax1.set_title("Prescribed Closed Deformation Path", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"$F_{11}$ [-]", fontsize=10)
    ax1.set_ylabel(r"$F_{22}$ [-]", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Plot accumulated energy around the cycle
    ax2.plot(np.arange(len(work_acc)), work_acc, color='crimson', linewidth=2.0)
    ax2.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)
    ax2.scatter([len(work_acc) - 1], [work_acc[-1]], color='black', s=50, zorder=5,
                label=f"Net $\Delta W$ = {net_work:.2e}")
    ax2.set_title(r"Accumulated Work $\oint \mathbf{P} : d\mathbf{F}$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Loop Step Index", fontsize=10)
    ax2.set_ylabel(r"Work $\mathcal{W}$ [$\mathrm{MJ/m^3}$]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.show()

def check_stress_free_reference(model_fn, atol: float = 1e-4):
    """
    Evaluates the stress-free reference configuration condition:
    P(F = I) == 0.
    """
    # 1. Undeformed reference state: F = Identity
    F_ref = jnp.eye(3)

    # 2. Evaluate model prediction
    P_ref = model_fn(F_ref)  # shape (3, 3)

    # 3. Calculate error metrics
    frobenius_norm = float(jnp.linalg.norm(P_ref))
    max_abs_component = float(jnp.max(jnp.abs(P_ref)))

    print("=" * 75)
    print("STRESS-FREE REFERENCE STATE CHECK: P(F = I) == 0")
    print("=" * 75)
    print("Predicted PK1 Stress Tensor at F = I:\n")
    print(np.array2string(np.array(P_ref), precision=4, suppress_small=True, floatmode="fixed"))
    print(f"\n  Frobenius Norm ||P(I)||      : {frobenius_norm:.4e}")
    print(f"  Max Absolute Component |P_ij|: {max_abs_component:.4e}")

    if frobenius_norm <= atol:
        status = "FULFILLED (Model predicts zero stress in reference state)"
    else:
        status = "VIOLATED (Model predicts spurious residual stress at zero strain)"

    print(f"  Condition Status             : {status}")
    print("=" * 75)

    return P_ref, frobenius_norm

def plot_reference_state_normalization(model_fn):
  """Reproduces the paper's schematic (h):

  Plots P_11 vs F_11 across compression (F < 1) and tension (F > 1).
  """
  # Range of stretch spanning compression (0.75) to tension (1.35)
  lambdas = np.linspace(0.75, 1.35, 150)

  # Uniaxial isochoric deformation path
  F_path = np.zeros((len(lambdas), 3, 3), dtype=np.float32)
  for k, lam in enumerate(lambdas):
    F_path[k] = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])

  # Model evaluation
  P_pred = jax.vmap(model_fn)(F_path)  # shape (N, 3, 3)
  P11_pred = P_pred[:, 0, 0]

  # Stress at F = I (lambda = 1)
  idx_ref = np.argmin(np.abs(lambdas - 1.0))
  P11_residual = P11_pred[idx_ref]

  fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")

  # Predicted curve (Dashed red)
  ax.plot(
      lambdas,
      P11_pred,
      color="#a00000",
      linestyle="--",
      linewidth=2.2,
      label=rf"Naïve FFNN: $P_{{11}}(\mathbf{{I}}) = {P11_residual:.2e}$",
  )

  # Coordinate axes passing through (1, 0)
  ax.axhline(0.0, color="black", linewidth=1.0)
  ax.axvline(1.0, color="black", linestyle=":", linewidth=1.0)

  # Mark the reference crossing
  ax.scatter(
      [1.0],
      [0.0],
      color="navy",
      s=40,
      zorder=5,
      label=r"Physical: $\mathbf{P}(\mathbf{I}) = \mathbf{0}$",
  )
  ax.scatter(
      [1.0],
      [P11_residual],
      color="#a00000",
      s=40,
      zorder=5,
      label="Residual Offset",
  )

  ax.set_title(
      r"(h) Normalization condition stress: $\mathbf{P}(\mathbf{I}) = \mathbf{0}$",
      fontsize=12,
      fontweight="bold",
  )
  ax.set_xlabel(r"Deformation $F_{11} = \lambda$ [-]", fontsize=11)
  ax.set_ylabel(r"PK1 Stress $P_{11}$", fontsize=11)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper left", fontsize=9)
  plt.show()


def plot_predicted_strain_energy_non_negativity(model_fn):
  """Evaluates and plots the predicted strain energy psi(F) along a 1D

  uniaxial stretch path spanning compression (F11 < 1) to tension (F11 > 1).

  Numerically integrates P_pred : dF from the reference state F = I (lambda = 1)
  to demonstrate:
    1. The curve dips below zero (psi < 0 under compression).
    2. The minimum does not lie at the reference state F = I.
  """
  # 1. Define stretch range across compression (0.75) and tension (1.35)
  n_points = 200
  lambdas = np.linspace(0.75, 1.35, n_points)

  # Isochoric uniaxial deformation path: F = diag(lam, 1/sqrt(lam), 1/sqrt(lam))
  F_path = np.zeros((n_points, 3, 3), dtype=np.float32)
  for k, lam in enumerate(lambdas):
    F_path[k] = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])

  # 2. Evaluate model stresses along the path
  P_pred = jax.vmap(model_fn)(F_path)  # shape (N, 3, 3)

  # 3. Locate the reference index where lambda = 1.0
  idx_ref = int(np.argmin(np.abs(lambdas - 1.0)))

  # 4. Integrate work: psi(lam) = \int_{1}^{lam} P : dF
  # Compute incremental work between adjacent steps: dW = P_mid : dF
  dF = np.diff(F_path, axis=0)  # (N-1, 3, 3)
  P_mid = 0.5 * (P_pred[:-1] + P_pred[1:])
  dW_steps = np.sum(P_mid * dF, axis=(1, 2))  # (N-1,)

  psi_pred = np.zeros(n_points)

  # Forward integration for tension (lambda >= 1)
  psi_pred[idx_ref + 1 :] = np.cumsum(dW_steps[idx_ref:])

  # Backward integration for compression (lambda < 1)
  # psi(lam) = - \int_{lam}^{1} P : dF
  psi_pred[:idx_ref] = -np.cumsum(dW_steps[:idx_ref][::-1])[::-1]

  min_psi = float(np.min(psi_pred))
  psi_ref = float(psi_pred[idx_ref])
  lam_at_min = float(lambdas[np.argmin(psi_pred)])

  print("=" * 75)
  print("NON-NEGATIVITY OF STRAIN ENERGY CHECK: psi(F) >= 0")
  print("=" * 75)
  print(f"  Reference State Energy psi(I)  : {psi_ref:.4e} MJ/m^3")
  print(f"  Minimum Strain Energy Value    : {min_psi:.4e} MJ/m^3")
  print(f"  Stretch at Minimum Energy      : lambda = {lam_at_min:.4f}")

  if min_psi < -1e-4:
    print(
        "  Condition Status               : VIOLATED (psi < 0 detected in"
        " compression)"
    )
  else:
    print("  Condition Status               : FULFILLED (psi >= 0 everywhere)")
  print("=" * 75)

  # 5. Plotting
  fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")

  # Predicted strain energy curve (Dashed red)
  ax.plot(
      lambdas,
      psi_pred,
      color="#a00000",
      linestyle="--",
      linewidth=2.2,
      label=r"Naïve FFNN: $\psi_{\mathrm{pred}}(\mathbf{F})$",
  )

  # Zero reference lines
  ax.axhline(0.0, color="black", linewidth=1.0)
  ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0)

  # Highlight the unphysical negative energy region
  ax.fill_between(
      lambdas,
      psi_pred,
      0.0,
      where=(psi_pred < 0.0),
      color="red",
      alpha=0.15,
      label=r"Unphysical Zone ($\psi < 0$)",
  )

  # Reference state marker (lambda = 1, psi = 0)
  ax.scatter(
      [1.0],
      [0.0],
      color="navy",
      s=50,
      zorder=5,
      label=r"Reference State: $\mathbf{F} = \mathbf{I}$",
  )

  # Minimum energy marker showing the dip into negative values
  ax.scatter(
      [lam_at_min],
      [min_psi],
      color="#a00000",
      s=50,
      zorder=5,
      label=rf"Minimum: $\psi = {min_psi:.2e}$ $\mathrm{{MJ/m^3}}$",
  )

  ax.set_title(
      r"(i) Non-negativity of strain energy: $\psi(\mathbf{F}) \geq 0$",
      fontsize=12,
      fontweight="bold",
  )
  ax.set_xlabel(r"Deformation $F_{11} = \lambda$ [-]", fontsize=11)
  ax.set_ylabel(r"Strain Energy Density $\psi$ [$\mathrm{MJ/m^3}$]", fontsize=11)
  ax.grid(True, linestyle="--", alpha=0.45)
  ax.legend(loc="upper left", fontsize=8.8)

  plt.show()

def plot_angular_momentum_parity(
    dataset_dict: dict,
    pred_dict: dict,
    title_prefix: str = "Angular Momentum Parity",
):
  """Evaluates P_pred @ F^T = F @ P_pred^T for each load case across ALL 9 components.

  Plots a parity plot comparing components of (P @ F^T)_ij vs (F @ P^T)_ij.
  Diagonal terms (11, 22, 33) are mathematically identical and will lie exactly
  on y=x. Off-diagonal terms will show scatter if angular momentum balance is
  violated.
  """
  # All 9 tensor indices
  comp_specs = [
      ((0, 0), "11"),
      ((0, 1), "12"),
      ((0, 2), "13"),
      ((1, 0), "21"),
      ((1, 1), "22"),
      ((1, 2), "23"),
      ((2, 0), "31"),
      ((2, 1), "32"),
      ((2, 2), "33"),
  ]

  # 9 distinct colors from tab10
  colors = plt.cm.tab10(np.linspace(0, 1, 9))

  for case_name, data in dataset_dict.items():
    F = np.array(data["F"])  # (N, 3, 3)
    P_pred = np.array(pred_dict[case_name]["P_pred"])  # (N, 3, 3)

    # Compute PF^T and FP^T: shape (N, 3, 3)
    PF_T = np.einsum("nij,nkj->nik", P_pred, F)
    FP_T = np.einsum("nij,nkj->nik", F, P_pred)

    # Skew-symmetric residual: Omega = PF^T - FP^T
    Omega = PF_T - FP_T
    max_asym = np.max(np.abs(Omega))
    mean_asym = np.mean(np.abs(Omega))

    print(f"[{title_prefix}] Case '{case_name}':")
    print(f"  Max Residual |(PF^T - FP^T)_ij|  : {max_asym:.4e}")
    print(f"  Mean Residual |(PF^T - FP^T)_ij| : {mean_asym:.4e}")

    # Determine dynamic plot bounds across all 9 components
    all_vals = np.concatenate([PF_T.ravel(), FP_T.ravel()])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    margin = (vmax - vmin) * 0.05 if vmax != vmin else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 6.0), layout="constrained")

    # Reference diagonal y = x
    ax.plot(
        [vmin - margin, vmax + margin],
        [vmin - margin, vmax + margin],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Exact Balance ($y=x$)",
        zorder=1,
    )

    # Plot all 9 component pairs
    for ((i, j), lbl), color in zip(comp_specs, colors):
      x_vals = FP_T[:, i, j]
      y_vals = PF_T[:, i, j]

      # Differentiate diagonals with a distinct marker style
      is_diagonal = i == j
      marker = "s" if is_diagonal else "o"
      size = 35 if is_diagonal else 25
      alpha = 0.85 if is_diagonal else 0.70

      ax.scatter(
        x_vals,
        y_vals,
        color=color,
        marker=marker,
        s=size,
        alpha=alpha,
        label=(
            rf"$(\mathbf{{P}}\mathbf{{F}}^T)_{{{lbl}}}$ vs"
            rf" $(\mathbf{{F}}\mathbf{{P}}^T)_{{{lbl}}}$"
        ),
        zorder=3 if is_diagonal else 2,
      )

    ax.set_xlim(vmin - margin, vmax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_xlabel(r"$(\mathbf{F}\mathbf{P}_{\mathrm{pred}}^T)_{ij}$", fontsize=11)
    ax.set_ylabel(r"$(\mathbf{P}_{\mathrm{pred}}\mathbf{F}^T)_{ij}$", fontsize=11)
    ax.set_title(
        f"{title_prefix}: {case_name}\n"
        rf"$\max |\mathbf{{P}}\mathbf{{F}}^T - \mathbf{{F}}\mathbf{{P}}^T| ="
        f" {max_asym:.2e}$",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.45)

    # 2-column legend outside the plot box to accommodate all 9 components
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, ncol=1
    )
    plt.show()

# Structural Tensor Definition from Eq. (5)
G_ti = jnp.diag(jnp.array([4.0, 0.5, 0.5]))


def analytical_strain_energy(F: jnp.ndarray) -> jnp.ndarray:
  """Exact hyperelastic energy density W(F) from Task 1 (Eq. 3)."""
  C = F.T @ F
  J = jnp.linalg.det(F)
  I1 = jnp.trace(C)
  I3 = J**2
  C_inv = jnp.linalg.inv(C)
  cof_C = I3 * C_inv.T
  I4 = jnp.trace(C @ G_ti)
  I5 = jnp.trace(cof_C @ G_ti)
  return 8.0 * I1 + 10.0 * (J**2) - 56.0 * jnp.log(J) + 0.2 * (I4**2 + I5**2) - 44.0


def plot_growth_condition(model_fn):
  """Evaluates and compares the asymptotic growth condition psi(F) as J -> 0+ and J -> infty

  along a pure volumetric deformation path F = J^(1/3) * I.
  """
  # 1. Define J-span approaching 0+ (compression) and expanding up to 2.5
  # Avoid J=0 exactly to allow analytical log(J) evaluation
  n_points = 300
  J_vals = np.linspace(0.12, 2.5, n_points)

  # Pure volumetric deformation: F(J) = J^(1/3) * I
  F_path = np.zeros((n_points, 3, 3), dtype=np.float32)
  for k, J in enumerate(J_vals):
    scale = J ** (1.0 / 3.0)
    F_path[k] = np.diag([scale, scale, scale])

  # 2. Exact True Energy Density
  batched_true_W = jax.vmap(analytical_strain_energy)
  psi_true = np.array(batched_true_W(jnp.array(F_path)))

  # 3. Model Predicted Work by integrating P : dF from J = 1.0
  batched_model = jax.vmap(model_fn)
  P_pred = np.array(batched_model(jnp.array(F_path)))  # (N, 3, 3)

  idx_ref = int(np.argmin(np.abs(J_vals - 1.0)))

  # Incremental work: dW = P_mid : dF
  dF = np.diff(F_path, axis=0)  # (N-1, 3, 3)
  P_mid = 0.5 * (P_pred[:-1] + P_pred[1:])
  dW_steps = np.sum(P_mid * dF, axis=(1, 2))  # (N-1,)

  psi_pred = np.zeros(n_points)
  # Forward integration for expansion (J >= 1)
  psi_pred[idx_ref + 1 :] = np.cumsum(dW_steps[idx_ref:])
  # Backward integration for compression (J < 1)
  psi_pred[:idx_ref] = -np.cumsum(dW_steps[:idx_ref][::-1])[::-1]

  # 4. Console diagnostics
  print("=" * 75)
  print(
      "GROWTH CONDITION CHECK: psi(F) -> infty as (J -> 0+ or J -> infty)"
  )
  print("=" * 75)
  print(f"At J = {J_vals[0]:.2f} (Extreme Compression):")
  print(f"  Exact Energy W_true          : {psi_true[0]:.4e} MJ/m^3")
  print(f"  Model Integrated psi_pred    : {psi_pred[0]:.4e} MJ/m^3")
  print(f"\nAt J = {J_vals[-1]:.2f} (Extreme Expansion):")
  print(f"  Exact Energy W_true          : {psi_true[-1]:.4e} MJ/m^3")
  print(f"  Model Integrated psi_pred    : {psi_pred[-1]:.4e} MJ/m^3")

  if psi_pred[0] < psi_true[0] * 0.1 or psi_pred[-1] < psi_true[-1] * 0.1:
    print(
        "Condition Status               : VIOLATED (Model saturates to finite"
        " values)"
    )
  else:
    print("Condition Status               : FULFILLED")
  print("=" * 75)

  # 5. Visual Plot matching the schematic
  fig, ax = plt.subplots(figsize=(6.8, 5.0), layout="constrained")

  # Physical / Analytical curve (Solid Navy)
  ax.plot(
      J_vals,
      psi_true,
      color="#0b3c5d",
      linestyle="-",
      linewidth=2.4,
      label=r"Physical: $\psi \to \infty$ as $J \to 0^+ \vee J \to \infty$",
  )

  # Naive FFNN curve (Dashed Crimson)
  ax.plot(
      J_vals,
      psi_pred,
      color="#b82601",
      linestyle="--",
      linewidth=2.2,
      label=r"Naïve FFNN: $\psi_{\mathrm{pred}}(J)$ (Saturating/Bounded)",
  )

  # Coordinate axes and reference state J = 1
  ax.axhline(0.0, color="black", linewidth=0.9)
  ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
  ax.scatter(
      [1.0],
      [0.0],
      color="#0b3c5d",
      s=45,
      zorder=5,
      label=r"Reference: $J = 1$",
  )

  # Add schematic asymptotic infinity arrows on the physical curve
  ax.annotate(
      r"$\infty$",
      xy=(J_vals[2], psi_true[2]),
      xytext=(J_vals[2] - 0.04, psi_true[2] * 0.95),
      fontsize=12,
      fontweight="bold",
      color="#0b3c5d",
  )
  ax.annotate(
      r"$\infty$",
      xy=(J_vals[-5], psi_true[-5]),
      xytext=(J_vals[-5], psi_true[-5] * 0.95),
      fontsize=12,
      fontweight="bold",
      color="#0b3c5d",
  )

  # Limit vertical view so the finite saturation of the NN is clearly visible
  ax.set_ylim(-10.0, min(120.0, float(np.max(psi_true) * 0.75)))
  ax.set_xlim(0.0, 2.6)

  ax.set_title(
      r"Growth condition: $\psi(\mathbf{F}) \to \infty$ as $(J \to \infty \vee"
      r" J \to 0^+)$",
      fontsize=12,
      fontweight="bold",
  )
  ax.set_xlabel(r"Volume Ratio $J = \det(\mathbf{F})$ [-]", fontsize=11)
  ax.set_ylabel(
      r"Strain Energy Density $\psi$ [$\mathrm{MJ/m^3}$]", fontsize=11
  )
  ax.grid(True, linestyle="--", alpha=0.45)
  ax.legend(loc="upper center", fontsize=8.8)

  plt.show()


# Structural Tensor Definition from Eq. (5)
G_ti = np.diag([4.0, 0.5, 0.5]).astype(np.float64)
sqrt_G = np.diag(np.sqrt([4.0, 0.5, 0.5])).astype(np.float64)


def polyconvex_auxiliary_P(A: np.ndarray, B: np.ndarray, c: float) -> float:
  """Evaluates the analytical polyconvex auxiliary energy:

  P(A, B, c) = 8||A||^2 + (10c^2 - 56 ln(c)) + 0.2||A G^(1/2)||^4 + 0.2||B G^(1/2)||^4 - 44
  """
  term1 = 8.0 * np.sum(A**2)
  term2 = 10.0 * (c**2) - 56.0 * np.log(np.maximum(c, 1e-12))
  term3 = 0.2 * (np.linalg.norm(A @ sqrt_G) ** 4)
  term4 = 0.2 * (np.linalg.norm(B @ sqrt_G) ** 4)
  return float(term1 + term2 + term3 + term4 - 44.0)


def plot_polyconvex_parity(
    dataset_dict: dict,
    model_fn,
    title_prefix: str = "Polyconvex Potential Parity",
):
  """Plots Polyconvex Potential P(F, cof F, det F) on the x-axis vs.

  Predicted Strain Energy psi_pred on the y-axis for each load path.
  """
  batched_model = jax.vmap(model_fn)

  fig, ax = plt.subplots(figsize=(7.2, 5.8), layout="constrained")
  colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_dict), 3)))

  all_x, all_y = [], []

  for idx, (case_name, data) in enumerate(dataset_dict.items()):
    F = np.array(data["F"], dtype=np.float64)  # (N, 3, 3)
    n_samples = F.shape[0]

    # 1. Compute Polyconvex Potential P(A, B, c) for the X-axis
    P_aux_exact = np.zeros(n_samples)
    for k in range(n_samples):
      F_k = F[k]
      c_k = float(np.linalg.det(F_k))
      B_k = c_k * np.linalg.inv(F_k).T  # cof(F) = det(F) * F^(-T)
      P_aux_exact[k] = polyconvex_auxiliary_P(F_k, B_k, c_k)

    # 2. Predict Stress using trained model
    P_pred = np.array(batched_model(jnp.array(F)))  # (N, 3, 3)

    # 3. Predict Strain Energy by integrating P_pred : dF for the Y-axis
    dF = np.diff(F, axis=0)  # (N-1, 3, 3)
    P_mid = 0.5 * (P_pred[:-1] + P_pred[1:])
    dW_steps = np.sum(P_mid * dF, axis=(1, 2))  # (N-1,)

    psi_pred = np.zeros(n_samples)
    psi_pred[0] = P_aux_exact[0]
    psi_pred[1:] = P_aux_exact[0] + np.cumsum(dW_steps)

    all_x.extend(P_aux_exact)
    all_y.extend(psi_pred)

    # Line style: solid for calibration, dashed for test
    is_test = "test" in case_name.lower()
    ls = "--" if is_test else "-"

    # Plot curve and discrete markers
    ax.plot(
        P_aux_exact,
        psi_pred,
        color=colors[idx],
        linestyle=ls,
        linewidth=2.0,
        label=f"{case_name}",
    )
    ax.scatter(
        P_aux_exact,
        psi_pred,
        color=colors[idx],
        s=18,
        alpha=0.6,
    )

    # Console diagnostics
    mae = np.mean(np.abs(psi_pred - P_aux_exact))
    max_err = np.max(np.abs(psi_pred - P_aux_exact))
    print(f"[{title_prefix}] Case '{case_name}':")
    print(f"  Mean Absolute Error |psi_pred - P_exact| : {mae:.4e} MJ/m^3")
    print(f"  Max Absolute Error  |psi_pred - P_exact| : {max_err:.4e} MJ/m^3")
    print("-" * 75)

  # Determine plot bounds and reference line y = x
  min_val = min(min(all_x), min(all_y))
  max_val = max(max(all_x), max(all_y))
  margin = 0.05 * (max_val - min_val) if max_val != min_val else 1.0
  lims = [min_val - margin, max_val + margin]

  ax.plot(
      lims,
      lims,
      color="black",
      linestyle=":",
      linewidth=1.5,
      label="Exact Match ($y = x$)",
      zorder=1,
  )

  ax.set_xlim(lims)
  ax.set_ylim(lims)
  ax.set_title(
      f"{title_prefix}\n"
      r"Polyconvex Potential $\mathcal{P}(\mathbf{F}, \mathrm{cof}\,\mathbf{F}, \det\mathbf{F})$ vs. Predicted $\psi_{\mathrm{pred}}$",
      fontsize=11,
      fontweight="bold",
  )
  ax.set_xlabel(
      r"Polyconvex Potential $\mathcal{P}(\mathbf{F}, \mathrm{cof}\,\mathbf{F}, \det\mathbf{F})$ [$\mathrm{MJ/m^3}$]",
      fontsize=11,
  )
  ax.set_ylabel(
      r"Predicted Strain Energy $\psi_{\mathrm{pred}} = \int \mathbf{P}_{\mathrm{pred}} : d\mathbf{F}$ [$\mathrm{MJ/m^3}$]",
      fontsize=11,
  )
  ax.grid(True, linestyle="--", alpha=0.45)
  ax.legend(loc="upper left", fontsize=9)

  plt.show()