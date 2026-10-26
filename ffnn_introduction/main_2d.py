"""Main example script."""

import datetime

import jax
import jax.random as jrandom
import klax
from matplotlib import pyplot as plt
import time
import numpy as np

import src.tmlsm.losses_2d as tl2d
import src.tmlsm.models_2d as tm2d
import src.tmlsm.data_2d as td2d

now = datetime.datetime.now


def main():
    # Create random key for reproducible weight initialization, and
    # batch splits. The call to `time.time_ns()` may be replaced with
    # a constant seed if exactly reproductible results ought to be
    # produced.
    key = jrandom.PRNGKey(42) #time.time_ns()
    keys = jrandom.split(key, 2)

    # 1. Configuration: Select model and target function
    # Options:
    #   model_choice: "ffnn" or "icnn"[cite: 8]
    #   target_choice: "f1" (saddle: x^2 - y^2) or "f2" (convex: x^2 + 0.5y^2)
    model_choice = "icnn2d" # ffnn2d or icnn2d
    target_choice = "f2"

    # 2D Data
    coords, f1, f2, grad_f2 = td2d.generate_2d_data()
    y_target = f1 if target_choice == "f1" else f2

    # Build model instance
    model = tm2d.build(model_type=model_choice, key=keys[0]) #ffnn or icnn
    print(model)

    # Calibrate model
    t1 = now()
    print(t1)

    model, history = klax.fit(
        model,
        (coords, y_target),
        batch_size=32,
        steps=1_0000,
        loss_fn=tl2d.MSE(),
        history=klax.HistoryCallback(log_every=1),
        key=keys[1],
    )

    t2 = now()
    print("it took", t2 - t1, "(sec) to calibrate the model")

    history.plot()

    # Evaluation
    # First the model need to be finalized to unwrap and apply all
    # wrappers and constraints (if present).
    model_ = klax.finalize(model)

    y_pred = jax.vmap(model)(coords)

    """Visualize true 2D surface versus model prediction in 3D."""
    X = coords[:, 0].reshape(20, 20)
    Y = coords[:, 1].reshape(20, 20)
    Z_true = np.array(y_target).reshape(20, 20)
    Z_pred = np.array(y_pred).reshape(20, 20)

    fig = plt.figure(figsize=(12, 5))

    # Ground Truth Surface
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.85)
    ax1.set_title(f"{f"{model_choice.upper()} fitted to {target_choice}"} - Ground Truth")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")

    # Model Prediction Surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X, Y, Z_pred, cmap="plasma", alpha=0.85)
    ax2.set_title(f"{f"{model_choice.upper()} fitted to {target_choice}"} - Model Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f_pred(x, y)")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
