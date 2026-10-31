import matplotlib.pyplot as plt
import numpy as np

def plot_evaluation(coords, y_true, y_pred, grad_true, grad_pred, strategy_name="model"):
    """Plots 3D surfaces and 2D gradient arrow vector fields."""
    X = coords[:, 0].reshape(20, 20)
    Y = coords[:, 1].reshape(20, 20)
    Z_true = np.array(y_true).reshape(20, 20)
    Z_pred = np.array(y_pred).reshape(20, 20)

    fig = plt.figure(figsize=(14, 6))

    if grad_pred == None:
    # Plot 1: 3D Surface Ground Truth
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.8)
        ax1.set_title("True f2 Surface")
    
        # Plot 2: 3D Surface Prediction
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(X, Y, Z_pred, cmap="plasma", alpha=0.8)
        ax2.set_title(f"Predicted Surface\n({strategy_name})")
        plt.tight_layout()
        plt.show()

    else:
    # Plot 1: 3D Surface Ground Truth
        ax1 = fig.add_subplot(1, 4, 1, projection="3d")
        ax1.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.8)
        ax1.set_title("True f2 Surface")
    
        # Plot 2: 3D Surface Prediction
        ax2 = fig.add_subplot(1, 4, 2, projection="3d")
        ax2.plot_surface(X, Y, Z_pred, cmap="plasma", alpha=0.8)
        ax2.set_title(f"Predicted Surface\n({strategy_name})")

        # Plot 3: True Gradient Vector Field (Quiver Plot)
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.quiver(
            coords[:, 0], coords[:, 1], grad_true[:, 0], grad_true[:, 1], color="blue"
        )
        ax3.set_title("True Gradient Vectors")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.grid(True)

        # Plot 4: Predicted Gradient Vector Field
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.quiver(
            coords[:, 0], coords[:, 1], grad_pred[:, 0], grad_pred[:, 1], color="red"
        )
        ax4.set_title(f"Predicted Gradient\n({strategy_name})")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.grid(True)
        plt.tight_layout()
        plt.show()