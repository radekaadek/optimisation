from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.io import loadmat


def load_isoperimetric_data(filepath: str) -> dict[str, Any]:
    """Loads and extracts variables from the MATLAB data file."""
    data = loadmat(filepath)

    n_nodes = int(data["N"].item())
    a_val = float(data["a"].item())
    length_limit = float(data["L"].item())
    max_curvature = float(data["C"].item())

    fixed_indices = data["F"].flatten() - 1
    y_fixed = data["y_fixed"].flatten()

    return {
        "n_nodes": n_nodes,
        "a_val": a_val,
        "length_limit": length_limit,
        "max_curvature": max_curvature,
        "fixed_indices": fixed_indices,
        "y_fixed": y_fixed,
    }


def get_base_constraints(
    y: cp.Variable,
    n_nodes: int,
    h_step: float,
    length_limit: float,
    fixed_indices: np.ndarray,
    y_fixed: np.ndarray,
    max_curvature: float,
    *,
    include_curvature: bool = True,
) -> list[cp.Constraint]:
    """Generates the base constraints for the isoperimetric problem."""
    constraints: list[cp.Constraint] = []

    dy = cp.diff(y)
    dx = np.full(n_nodes, h_step)
    segment_lengths = cp.norm(cp.vstack([dx, dy]), axis=0)
    constraints.append(cp.sum(segment_lengths) <= length_limit)

    constraints.append(y[0] == 0)
    constraints.append(y[n_nodes] == 0)

    # Fixed points constraints
    constraints.append(y[fixed_indices] == y_fixed[fixed_indices])

    # Curvature constraint
    if include_curvature:
        d2y = cp.diff(y, 2)
        constraints.append(cp.abs(d2y) / (h_step**2) <= max_curvature)

    return constraints


def solve_problem(
    objective_type: str,
    n_nodes: int,
    h_step: float,
    length_limit: float,
    fixed_indices: np.ndarray,
    y_fixed: np.ndarray,
    max_curvature: float,
    *,
    include_curvature: bool = True,
    non_negative: bool = False,
) -> tuple[cp.Variable, float]:
    """Builds and solves a single variation of the optimization problem."""
    y = cp.Variable(n_nodes + 1)
    area = h_step * cp.sum(y[:-1])

    constraints = get_base_constraints(
        y,
        n_nodes,
        h_step,
        length_limit,
        fixed_indices,
        y_fixed,
        max_curvature,
        include_curvature=include_curvature,
    )

    if non_negative:
        constraints.append(y >= 0)

    if objective_type == "maximize":
        prob = cp.Problem(cp.Maximize(area), constraints)
    else:
        prob = cp.Problem(cp.Minimize(area), constraints)

    prob.solve()

    # Handle the case where the solver might return None for the value
    solved_area = float(area.value) if area.value is not None else 0.0
    return y, solved_area


def plot_task(
    ax: Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    fixed_indices: np.ndarray,
    y_fixed: np.ndarray,
    title: str,
) -> None:
    """Plots a single task result."""
    ax.plot(x_vals, y_vals, "k-")
    ax.plot(x_vals[fixed_indices], y_fixed[fixed_indices], "ro")
    ax.set_title(title)
    ax.set_xlabel("x / a")
    ax.set_ylabel("y(x)")
    ax.grid(visible=True, linestyle="--", alpha=0.5)


def run_isoperimetric_tasks() -> None:
    """Main execution function."""
    data = load_isoperimetric_data("isoPerimData.mat")

    n_nodes = data["n_nodes"]
    a_val = data["a_val"]
    length_limit = data["length_limit"]
    max_curvature = data["max_curvature"]
    fixed_indices = data["fixed_indices"]
    y_fixed = data["y_fixed"]

    h_step = a_val / n_nodes
    x_vals = np.linspace(0, a_val, n_nodes + 1) / a_val

    y1, area1 = solve_problem(
        "maximize", n_nodes, h_step, length_limit, fixed_indices, y_fixed, max_curvature
    )
    print(f"Task 1 (Base Maximize) Area: {area1:.4f}")

    y_a, area_a = solve_problem(
        "minimize", n_nodes, h_step, length_limit, fixed_indices, y_fixed, max_curvature
    )
    print(f"Task (a) Minimum Area: {area_a:.4f}")

    y_b, area_b = solve_problem(
        "minimize",
        n_nodes,
        h_step,
        length_limit,
        fixed_indices,
        y_fixed,
        max_curvature,
        non_negative=True,
    )
    print(f"Task (b) Minimum Area (y >= 0): {area_b:.4f}")

    y_c, area_c = solve_problem(
        "maximize",
        n_nodes,
        h_step,
        length_limit,
        fixed_indices,
        y_fixed,
        max_curvature,
        include_curvature=False,
    )
    print(f"Task (c) Maximum Area (No max curvature): {area_c:.4f}")

    _, axs = plt.subplots(2, 2, figsize=(12, 8))

    plot_task(
        axs[0, 0],
        x_vals,
        y1.value,
        fixed_indices,
        y_fixed,
        "Task 1: Base Maximization",
    )
    plot_task(
        axs[0, 1],
        x_vals,
        y_a.value,
        fixed_indices,
        y_fixed,
        "Task (a): Minimization",
    )
    plot_task(
        axs[1, 0],
        x_vals,
        y_b.value,
        fixed_indices,
        y_fixed,
        "Task (b): Minimization (y >= 0)",
    )
    plot_task(
        axs[1, 1],
        x_vals,
        y_c.value,
        fixed_indices,
        y_fixed,
        "Task (c): Maximization (No curvature constraint)",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_isoperimetric_tasks()
