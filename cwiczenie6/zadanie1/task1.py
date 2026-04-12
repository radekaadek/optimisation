from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

P = (1 / 8) * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])
xc = np.array([[1.0], [1.0]])

x0_1 = np.array([[2.0], [-2.0]])
epsilon_1 = 1e-4
alpha_1 = 0.5
beta_1 = 0.5


def f1(x: np.ndarray) -> float:
    x1 = float(x[0, 0])
    x2 = float(x[1, 0])
    term1 = float(np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1))
    diff = x - xc
    term2 = (diff.T @ P @ diff).item()
    return term1 + term2


def grad_f1(x: np.ndarray) -> np.ndarray:
    x1 = float(x[0, 0])
    x2 = float(x[1, 0])
    term1 = np.array(
        [
            [np.exp(x1 + 3 * x2 - 0.1) - np.exp(-x1 - 0.1)],
            [3 * np.exp(x1 + 3 * x2 - 0.1)],
        ]
    )
    term2 = 2 * P @ (x - xc)
    return term1 + term2


def hess_f1(x: np.ndarray) -> np.ndarray:
    x1 = float(x[0, 0])
    x2 = float(x[1, 0])
    hessian = np.array(
        [
            [
                np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1),
                3 * np.exp(x1 + 3 * x2 - 0.1),
            ],
            [3 * np.exp(x1 + 3 * x2 - 0.1), 9 * np.exp(x1 + 3 * x2 - 0.1)],
        ]
    )
    return hessian + 2 * P


def newton_classic(
    x0: np.ndarray,
    eps: float,
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    x: np.ndarray = x0.copy()
    history: list[np.ndarray] = [x.copy()]
    while True:
        g: np.ndarray = grad_f(x)
        hessian: np.ndarray = hess_f(x)
        v: np.ndarray = -np.linalg.inv(hessian) @ g
        delta = (-g.T @ v).item()
        if delta < eps:
            break
        x = x + v
        history.append(x.copy())
    return x, history


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


x_opt_classic, hist_classic = newton_classic(x0_1, epsilon_1, grad_f1, hess_f1)
print(
    f"Klasyczny Newton - Minimum: \n{x_opt_classic}\nLiczba iteracji: {len(hist_classic) - 1}"
)

x_opt_damped, hist_damped = newton_damped(
    x0_1, epsilon_1, alpha_1, beta_1, f1, grad_f1, hess_f1
)
print(
    f"Newton z tłumieniem - Minimum: \n{x_opt_damped}\nLiczba iteracji: {len(hist_damped) - 1}"
)


def f1_wrapper(x_flat: np.ndarray) -> float:
    return f1(x_flat.reshape(-1, 1))


res_scipy: OptimizeResult = minimize(f1_wrapper, x0_1.flatten(), method="Nelder-Mead")
print(f"Scipy fminsearch (Nelder-Mead) - Minimum: \n{res_scipy.x.reshape(-1, 1)}\n")


x_vals: np.ndarray = np.linspace(-3.5, 3.5, 400)
y_vals: np.ndarray = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z: np.ndarray = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f1(np.array([[X[i, j]], [Y[i, j]]]))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
levels: list[float] = [2.47, 3.62, 5.34, 7.59, 19.2, 50, 200, 600]
cs1 = plt.contour(X, Y, Z, levels=levels, colors="k")
plt.clabel(cs1, inline=True, fontsize=8)
hist_c_np: np.ndarray = np.array(hist_classic).squeeze()
plt.plot(hist_c_np[:, 0], hist_c_np[:, 1], "bo-", label="Iteracje (Klasyczny)")
plt.title("Klasyczna Metoda Newtona")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(1, 2, 2)
cs2 = plt.contour(X, Y, Z, levels=levels, colors="k")
plt.clabel(cs2, inline=True, fontsize=8)
hist_d_np: np.ndarray = np.array(hist_damped).squeeze()
plt.plot(hist_d_np[:, 0], hist_d_np[:, 1], "ro-", label="Iteracje (Z tłumieniem)")
plt.title("Metoda Newtona z Tłumieniem")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig("results.png")
