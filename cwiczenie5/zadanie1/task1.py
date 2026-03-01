from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def backtracking_search(
    phi: Callable[[Any], Any],
    phi_0: float,
    phi_prime_0: float,
    alpha: float,
    beta: float,
    s_init: float = 1.0,
) -> tuple[float, list[float]]:
    s: float = s_init
    s_history: list[float] = [s]

    while phi(s) > phi_0 + alpha * phi_prime_0 * s:
        s = beta * s
        s_history.append(s)

    return s, s_history


def plot_on_axis(
    ax: Axes,
    phi_func: Callable[[Any], Any],
    title: str,
    alpha: float,
    beta: float,
    alphas_to_draw: list[float],
) -> None:
    phi_0: float = 29.0
    phi_prime_0: float = -44.0

    s_vals: np.ndarray = np.linspace(0, 2.5, 200)

    ax.plot(s_vals, phi_func(s_vals), "k-", linewidth=2.5, label=r"$\phi(s)$")

    for a in alphas_to_draw:
        y_vals: np.ndarray = phi_0 + a * phi_prime_0 * s_vals
        ax.plot(s_vals, y_vals, "b-", linewidth=1)
        ax.text(
            2.1,
            phi_0 + a * phi_prime_0 * 2.1 + 1,
            rf"$\alpha={a}$",
            fontsize=10,
            color="darkblue",
        )

    s_opt, s_hist_list = backtracking_search(phi_func, phi_0, phi_prime_0, alpha, beta)
    s_hist: np.ndarray = np.array(s_hist_list)

    ax.plot(s_hist, phi_func(s_hist), "ks-", label=r"Kroki na $\phi(s)$")
    ax.plot(
        s_hist,
        phi_0 + alpha * phi_prime_0 * s_hist,
        "rs-",
        label="Kroki na warunku Armijo",
    )

    ax.plot(s_opt, phi_func(s_opt), "go", markersize=8, label="Zaakceptowany krok")

    ax.set_ylim(-30, 50)
    ax.set_xlim(0, 2.5)

    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xlabel("s")
    ax.set_ylabel(r"$\phi(s), y(s)$")
    ax.set_title(title)
    ax.legend(loc="upper right")


def phi1(s: Any) -> Any:
    return 20 * s**2 - 44 * s + 29


def phi2(s: Any) -> Any:
    return 40 * s**3 + 20 * s**2 - 44 * s + 29


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

plot_on_axis(
    ax=ax1,
    phi_func=phi1,
    title=r"Rys. 3: Backtracking search dla $\phi(s)=20s^2-44s+29, \alpha=0.3, \beta=0.8$",
    alpha=0.3,
    beta=0.8,
    alphas_to_draw=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
)

plot_on_axis(
    ax=ax2,
    phi_func=phi2,
    title=r"Rys. 4: Backtracking search dla $\phi(s)=40s^3+20s^2-44s+29, \alpha=0.4, \beta=0.9$",
    alpha=0.4,
    beta=0.9,
    alphas_to_draw=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
)

plt.tight_layout()

plt.show()
