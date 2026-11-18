import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
# GP KERNEL VALIDATION
# ============================================================

def validate_kernels(i, functions):
    X = functions[i]["X"]
    y = functions[i]["y"]

    kernels = {
        "RBF 0.1": RBF(length_scale=0.1),
        "RBF 0.5": RBF(length_scale=0.5),
        "Matern 0.1": Matern(length_scale=0.1),
        "Matern 0.5": Matern(length_scale=0.5),
        "RQ 0.1": RationalQuadratic(length_scale=0.1),
        "RQ 0.5": RationalQuadratic(length_scale=0.5),
    }

    loo = LeaveOneOut()
    results = {}

    for name, kernel in kernels.items():
        predictions = []
        actual = []

        for train_idx, test_idx in loo.split(X):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=42
            )

            gp.fit(X[train_idx], y[train_idx])

            pred = gp.predict(X[test_idx])[0]

            predictions.append(pred)
            actual.append(y[test_idx][0])

        mae = mean_absolute_error(actual, predictions)
        results[name] = mae

    return dict(sorted(results.items(), key=lambda x: x[1]))


# ============================================================
# GP FITTING
# ============================================================

def fit_gp(i, functions, kernel):
    X = functions[i]["X"]
    y = functions[i]["y"]

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42
    )

    gp.fit(X, y)

    return gp


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