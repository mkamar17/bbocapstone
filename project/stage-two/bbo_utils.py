import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.exceptions import ConvergenceWarning
import warnings

from query_history import query_history

def load_functions():
    functions = {}

    for i in range(1, 9):
        X = np.load(f"../initial_data/function_{i}/initial_inputs.npy")
        y = np.load(f"../initial_data/function_{i}/initial_outputs.npy")

        for query_X, query_y in query_history[i]:
            X = np.vstack([X, query_X])
            y = np.append(y, query_y)

        functions[i] = {
            "X": X,
            "y": y
        }

    return functions

def check_result(i, functions):
    X = functions[i]["X"]
    y = functions[i]["y"]

    best = np.argmax(y)

    print(f"Function {i}")
    print(f"Latest: X={X[-1]}, y={y[-1]}")
    print(f"Best:   X={X[best]}, y={y[best]}")


def plot_scores(i, functions):
    y = functions[i]["y"]

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(y) + 1), y, marker="o")
    plt.xlabel("Observation")
    plt.ylabel("Output")
    plt.title(f"Function {i}: observed outputs")
    plt.grid(alpha=0.3)
    plt.show()

# ============================================================
# VISUALISATION
# ============================================================

def plot_2d(i, functions, y=None):
    X = functions[i]["X"]
    y = functions[i]["y"] if y is None else y

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=80)
    plt.colorbar(label="Output")

    # Last round's query
    plt.scatter(
        X[-1, 0], X[-1, 1],
        marker="*", s=250,
        edgecolors="red",
        facecolors="none",
        linewidths=2,
        label="Last query"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Function {i}")
    plt.legend()
    plt.show()

def scale_scores(y):
    qt = QuantileTransformer(
        n_quantiles=len(y),
        output_distribution="uniform"
    )

    return qt.fit_transform(y.reshape(-1, 1)).ravel()

def plot_3d(i, functions, y=None):
    X = functions[i]["X"]
    y = functions[i]["y"] if y is None else y

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=y,
        cmap="viridis",
        s=80
    )

    fig.colorbar(scatter, ax=ax, label="Output", shrink=0.7)

    # Last round's query
    ax.scatter(
        X[-1, 0],
        X[-1, 1],
        X[-1, 2],
        marker="*",
        s=300,
        edgecolors="red",
        facecolors="none",
        linewidths=2,
        label="Last query"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(f"Function {i}")
    ax.legend()

    plt.show()

# ============================================================
# GP KERNEL VALIDATION
# ============================================================
def validate_kernels(i, functions):
    X = functions[i]["X"]
    y = functions[i]["y"]

    length_scales = [0.05, 0.1, 0.2, 0.5]
    results = {}

    for ls in length_scales:
        kernels = {
            "RBF": RBF(ls),
            "Matern": Matern(ls, nu=2.5),
            "RQ": RationalQuadratic(
                length_scale=ls,
                alpha=1.0
            )
        }

        for name, kernel in kernels.items():
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                optimizer=None
            )

            mae = -cross_val_score(
                gp,
                X,
                y,
                cv=LeaveOneOut(),
                scoring="neg_mean_absolute_error"
            ).mean()

            results[(name, ls)] = mae

    return sorted(results.items(), key=lambda x: x[1])


# ============================================================
# GP FITTING
# ============================================================

def fit_gp(i, functions, kernel):
    X = functions[i]["X"]
    y = functions[i]["y"]

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=None
    )

    gp.fit(X, y)

    return gp

def check_uncertainty(gp, suggestions):
    for name, point in suggestions.items():
        mean, std = gp.predict(point.reshape(1, -1), return_std=True)

        print(
            f"{name}: mean={mean[0]:.4f}, "
            f"uncertainty={std[0]:.4f}"
        )

def plot_input_sensitivity(gp, best_x, n_points=100):
    values = np.linspace(0, 1, n_points)

    for j in range(len(best_x)):
        X_test = np.tile(best_x, (n_points, 1))
        X_test[:, j] = values

        mean, std = gp.predict(X_test, return_std=True)

        plt.figure(figsize=(7, 4))
        plt.plot(values, mean)
        plt.fill_between(
            values,
            mean - std,
            mean + std,
            alpha=0.2
        )

        plt.axvline(best_x[j], linestyle="--")
        plt.xlabel(f"x{j + 1}")
        plt.ylabel("GP predicted output")
        plt.title(f"Input sensitivity to x{j + 1}")
        plt.show()

# ============================================================
# ACQUISITION FUNCTIONS
# ============================================================

def acquisitions(gp, X_candidates, best_y, beta=1.96):
    mean, std = gp.predict(X_candidates, return_std=True)

    # Upper Confidence Bound
    ucb = mean + beta * std

    # Expected Improvement
    improvement = mean - best_y
    z = improvement / (std + 1e-9)

    ei = (
        improvement * norm.cdf(z)
        + std * norm.pdf(z)
    )

    # Probability of Improvement
    pi = norm.cdf(z)

    return {
        "UCB": X_candidates[np.argmax(ucb)],
        "EI": X_candidates[np.argmax(ei)],
        "PI": X_candidates[np.argmax(pi)]
    }


# ============================================================
# QUERY PREDICTION
# ============================================================

def check_prediction(gp, point):
    point = np.asarray(point).reshape(1, -1)

    mean, std = gp.predict(point, return_std=True)

    print(f"Predicted mean: {mean[0]:.6f}")
    print(f"Uncertainty:    {std[0]:.6f}")

    return mean[0], std[0]


# ============================================================
# 2D GP VISUALISATION
# ============================================================

def plot_gp_2d(i, gp, functions, suggestions=None):
    X = functions[i]["X"]

    grid = np.linspace(0, 1, 100)
    x1, x2 = np.meshgrid(grid, grid)

    X_grid = np.c_[x1.ravel(), x2.ravel()]

    mean = gp.predict(X_grid)

    plt.figure(figsize=(7, 5))

    plt.contourf(
        x1,
        x2,
        mean.reshape(x1.shape),
        levels=30
    )

    plt.colorbar(label="Predicted output")

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c="white",
        edgecolors="black"
    )

    if suggestions:
        for name, point in suggestions.items():
            plt.scatter(
                *point,
                marker="X",
                s=120,
                label=name
            )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Function {i}: GP predicted mean")

    if suggestions:
        plt.legend()

    plt.show()


def plot_gp_uncertainty_2d(i, gp, functions):
    X = functions[i]["X"]

    grid = np.linspace(0, 1, 100)
    x1, x2 = np.meshgrid(grid, grid)

    X_grid = np.c_[x1.ravel(), x2.ravel()]

    _, std = gp.predict(
        X_grid,
        return_std=True
    )

    plt.figure(figsize=(7, 5))

    plt.contourf(
        x1,
        x2,
        std.reshape(x1.shape),
        levels=30
    )

    plt.colorbar(label="Predictive uncertainty (std)")

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c="white",
        edgecolors="black"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Function {i}: GP uncertainty")

    plt.show()


# ============================================================
# QUERY FORMATTING
# ============================================================

def format_query(point):
    return "-".join(f"{x:.6f}" for x in point)