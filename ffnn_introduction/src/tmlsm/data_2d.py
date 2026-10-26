"""2D Data generation module"""

import jax
import numpy

def generate_2d_data():
    """Generate Grid coordinates, f1, f2, and gradient of f2"""

    # 20 Equispaced points in [-4,4] each direction
    x_1d = jax.numpy.linspace(-4.0, 4.0, 20)
    y_1d = jax.numpy.linspace(-4.0, 4.0, 20)

    # Create a 2D grid of points
    X, Y = jax.numpy.meshgrid(x_1d, y_1d)

    # Flatten coordinates into pairs of (x,y) for model input: shape (400, 2)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    coords = jax.numpy.column_stack([x_flat, y_flat])

    # f1(x) = x^2 - y^2: shape (400,) # pure 1D array
    f1 = x_flat**2 - y_flat**2

    # f2(x) = x^2 + 0.5 * y^2: shape (400,)
    f2 = x_flat**2 + 0.5 * (y_flat**2)

    # df2/dx = [2x, y]^T: shape (400, 2)
    df2_dx = 2.0 * x_flat
    df2_dy = 1.0 * y_flat
    grad_f2 = jax.numpy.column_stack([df2_dx, df2_dy])

    return coords, f1, f2, grad_f2

