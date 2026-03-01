from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

P: np.ndarray = (1 / 8) * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])
xc: np.ndarray = np.array([[1.0], [1.0]])


x0_2: np.ndarray = xc.copy()
epsilon_2: float = 1e-4
alpha_2: float = 0.3
beta_2: float = 0.8
t_values: list[float] = [0.1, 1.0, 10.0]


def f2(x: np.ndarray, t: float) -> float:
    x1: float = float(x[0, 0])
    x2: float = float(x[1, 0])
    term1: float = float(t * (np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1)))
    diff: np.ndarray = x - xc
    arg: float = 1.0 - (diff.T @ P @ diff).item()

    if arg <= 0:
        return float(np.inf)
    return float(term1 - np.log(arg))


def grad_f2(x: np.ndarray, t: float) -> np.ndarray:
    x1: float = float(x[0, 0])
    x2: float = float(x[1, 0])
    term1: np.ndarray = t * np.array(
        [
            [np.exp(x1 + 3 * x2 - 0.1) - np.exp(-x1 - 0.1)],
            [3 * np.exp(x1 + 3 * x2 - 0.1)],
        ]
    )
    diff: np.ndarray = x - xc
    arg: float = 1.0 - (diff.T @ P @ diff).item()
    term2: np.ndarray = (2 * P @ diff) / arg
    return term1 + term2


def hess_f2(x: np.ndarray, t: float) -> np.ndarray:
    x1: float = float(x[0, 0])
    x2: float = float(x[1, 0])
    h1_mat: np.ndarray = t * np.array(
        [
            [
                np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1),
                3 * np.exp(x1 + 3 * x2 - 0.1),
            ],
            [3 * np.exp(x1 + 3 * x2 - 0.1), 9 * np.exp(x1 + 3 * x2 - 0.1)],
        ]
    )
    diff: np.ndarray = x - xc
    arg: float = 1.0 - (diff.T @ P @ diff).item()

    term2: np.ndarray = (4 * (P @ diff) @ (diff.T @ P)) / (arg**2)
    term3: np.ndarray = (2 * P) / arg
    return h1_mat + term2 + term3


def newton_damped(
    x0: np.ndarray,
    eps: float,
    alpha: float,
    beta: float,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    x: np.ndarray = x0.copy()
    history: list[np.ndarray] = [x.copy()]
    while True:
        g = grad_f(x)
        hessian = hess_f(x)
        v = -np.linalg.inv(hessian) @ g
        delta = (-g.T @ v).item()
        if delta < eps:
            break

        s = 1.0
        while f(x + s * v) > f(x) + s * alpha * (g.T @ v).item():
            s *= beta

        x = x + s * v
        history.append(x.copy())
    return x, history


def make_t_funcs(
    t_val: float,
) -> tuple[
    Callable[[np.ndarray], float],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    def f_wrap(x: np.ndarray) -> float:
        return f2(x, t_val)

    def grad_wrap(x: np.ndarray) -> np.ndarray:
        return grad_f2(x, t_val)

    def hess_wrap(x: np.ndarray) -> np.ndarray:
        return hess_f2(x, t_val)

    return f_wrap, grad_wrap, hess_wrap


for t_param in t_values:
    f2_t, grad_f2_t, hess_f2_t = make_t_funcs(t_param)

    x_opt_t, hist_t = newton_damped(
        x0_2, epsilon_2, alpha_2, beta_2, f2_t, grad_f2_t, hess_f2_t
    )
    print(
        f"t = {t_param} | Metoda z tłumieniem - Minimum:\n{x_opt_t.flatten()} | Iteracje: {len(hist_t) - 1}"
    )

    def f2_scipy_wrapper(x_flat: np.ndarray) -> float:
        return f2_t(x_flat.reshape(-1, 1))

    res_scipy_t: OptimizeResult = minimize(
        f2_scipy_wrapper, x0_2.flatten(), method="Nelder-Mead"
    )
    print(f"t = {t_param} | Scipy (Nelder-Mead) - Minimum:\n{res_scipy_t.x}\n")
