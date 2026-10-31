"""Main example script."""

import datetime

import jax
import jax.random as jrandom
import klax
from matplotlib import pyplot as plt
import time
import numpy as np
import tmlsm.plot_recipes as plot_results

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
    model_choice = "icnn2d" # ffnn2d or icnn2d or sobolev_icnn2d
    target_choice = "f2"
    strategy_name = None

    # 2D Data
    coords, f1, f2, grad_f2 = td2d.generate_2d_data()
    y_target = f1 if target_choice == "f1" else f2

    # Build model instance
    model = tm2d.build(model_type=model_choice, key=keys[0]) #ffnn or icnn
    print(model)

    # Calibrate model
    t1 = now()
    print(t1)

    #select loss function according to the strategy
    loss_choice = "strategy_3"

    if loss_choice == "strategy_1":
        loss_fn = tl2d.LossStrategy1_valueOnly()
    elif loss_choice == "strategy_2":
        loss_fn = tl2d.LossStrategy2_ValueAndGradient()
    elif loss_choice == "strategy_3":
        loss_fn = tl2d.LossStrategy3_GradientOnly()
    else:
        raise ValueError("Invalid strategy name")

    if model_choice == "sobolev_icnn2d":
        model, history = klax.fit(
            model,
            (coords, f1, f2, grad_f2),
            batch_size=32,
            steps=1_0000,
            loss_fn=loss_fn,
            history=klax.HistoryCallback(log_every=100),
            key=keys[1],
        )
    else:
        model, history = klax.fit(
            model,
            (coords, y_target),
            batch_size=32,
            steps=1_0000,
            loss_fn=tl2d.MSE(),
            history=klax.HistoryCallback(log_every=100),
            key=keys[1],
        )
    t2 = now()
    print("it took", t2 - t1, "(sec) to calibrate the model")

    history.plot()

    # Evaluation
    # First the model need to be finalized to unwrap and apply all
    # wrappers and constraints (if present).
    model_ = klax.finalize(model)

    if model_choice == "sobolev_icnn2d":
        y_pred, grad_pred = jax.vmap(model)(coords)
    else:
        y_pred = jax.vmap(model)(coords)
        loss_choice = "strategy_1"
        strategy_name = model_choice

    if loss_choice == "strategy_1":
        plot_results.plot_evaluation(coords, f2, y_pred, grad_f2, None, 
                                     strategy_name="values only" if strategy_name is None else strategy_name)
    elif loss_choice == "strategy_2":
        plot_results.plot_evaluation(coords, f2, y_pred, grad_f2, grad_pred, strategy_name="values and gradients")
    elif loss_choice == "strategy_3":
        plot_results.plot_evaluation(coords, f2, y_pred, grad_f2, grad_pred, strategy_name="gradients only")


if __name__ == "__main__":
    main()
