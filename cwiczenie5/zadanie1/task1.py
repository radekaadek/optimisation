from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def backtracking_search(
    phi: Callable[[Any], Any],
    phi_0: float,           # Wartość funkcji w punkcie początkowym s=0, czyli phi(0)
    phi_prime_0: float,     # Wartość pochodnej w punkcie s=0, czyli phi'(0)
    alpha: float,           # Parametr alfa z przedziału (0, 1) do warunku Armijo
    beta: float,            # Parametr beta z przedziału (0, 1) określający tempo zmniejszania kroku s
    s_init: float = 1.0,    # Początkowa długość kroku (najczęściej s=1)
) -> tuple[float, list[float]]:
    """
    Funkcja implementująca algorytm dokładnego poszukiwania w kierunku (backtracking line search).
    Jej celem jest znalezienie takiej długości kroku 's', która zagwarantuje wystarczający spadek 
    wartości funkcji (spełnienie warunku Armijo).
    """
    s: float = s_init
    s_history: list[float] = [s] # Zapisujemy historię kroków do późniejszego wyrysowania

    # Pętla działa dopóki NIE jest spełniony warunek Armijo.
    # Warunek Armijo: phi(s) <= phi(0) + s * alpha * phi'(0)
    # Zatem jeśli phi(s) JEST WIĘKSZE, krok s jest zbyt duży i nie daje wystarczającego spadku.
    while phi(s) > phi_0 + alpha * phi_prime_0 * s:
        s = beta * s # Zmniejszamy krok s iteracyjnie z ustaloną prędkością beta
        s_history.append(s)

    # Zwracamy ostateczny, zaakceptowany krok 's' oraz całą historię poszukiwań
    return s, s_history


def plot_on_axis(
    ax: Axes,
    phi_func: Callable[[Any], Any],
    title: str,
    alpha: float,
    beta: float,
    alphas_to_draw: list[float],
) -> None:
    # Parametry brzegowe obu badanych funkcji (z zadania):
    # Dla phi1(s) = 20s^2 - 44s + 29 oraz phi2(s) = 40s^3 + 20s^2 - 44s + 29
    # Wartość dla s=0 to w obu przypadkach 29.
    phi_0: float = 29.0
    # Pochodna dla s=0 (współczynnik przy potędze 1) to w obu przypadkach -44.
    phi_prime_0: float = -44.0

    # Generowanie wartości s do narysowania gładkiego wykresu funkcji
    s_vals: np.ndarray = np.linspace(0, 2.5, 200)

    # Rysowanie głównego wykresu badanej funkcji phi(s)
    ax.plot(s_vals, phi_func(s_vals), "k-", linewidth=2.5, label=r"$\phi(s)$")

    # Rysowanie prostych l(s) dla różnych wartości parametru alpha
    # Prosta ma równanie: y(s) = phi(0) + alpha * phi'(0) * s
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

    # Wywołanie algorytmu backtracking search dla zadanego alpha i beta
    s_opt, s_hist_list = backtracking_search(phi_func, phi_0, phi_prime_0, alpha, beta)
    s_hist: np.ndarray = np.array(s_hist_list)

    # Zaznaczanie punktów (kroków) sprawdzanych przez algorytm na krzywej funkcji phi(s)
    ax.plot(s_hist, phi_func(s_hist), "ks-", label=r"Kroki na $\phi(s)$")
    # Zaznaczanie odpowiadających im punktów na prostej warunku Armijo
    ax.plot(
        s_hist,
        phi_0 + alpha * phi_prime_0 * s_hist,
        "rs-",
        label="Kroki na warunku Armijo",
    )

    # Zaznaczenie ostatecznie zaakceptowanego punktu (zielona kropka)
    ax.plot(s_opt, phi_func(s_opt), "go", markersize=8, label="Zaakceptowany krok")

    # Ustawienia estetyczne wykresu (limity osi, siatka, etykiety)
    ax.set_ylim(-30, 50)
    ax.set_xlim(0, 2.5)
    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xlabel("s")
    ax.set_ylabel(r"$\phi(s), y(s)$")
    ax.set_title(title)
    ax.legend(loc="upper right")

# Definicja pierwszej funkcji z zadania (wykorzystywana na Rys. 3)
def phi1(s: Any) -> Any:
    return 20 * s**2 - 44 * s + 29

# Definicja drugiej funkcji z zadania (wykorzystywana na Rys. 4)
def phi2(s: Any) -> Any:
    return 40 * s**3 + 20 * s**2 - 44 * s + 29

# Tworzenie obszaru roboczego z dwoma wykresami (1 wiersz, 2 kolumny)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Generowanie wykresu lewego (Rys. 3 z zadania)
plot_on_axis(
    ax=ax1,
    phi_func=phi1,
    title=r"Rys. 3: Backtracking search dla $\phi(s)=20s^2-44s+29, \alpha=0.3, \beta=0.8$",
    alpha=0.3,
    beta=0.8,
    alphas_to_draw=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
)

# Generowanie wykresu prawego (Rys. 4 z zadania)
plot_on_axis(
    ax=ax2,
    phi_func=phi2,
    title=r"Rys. 4: Backtracking search dla $\phi(s)=40s^3+20s^2-44s+29, \alpha=0.4, \beta=0.9$",
    alpha=0.4,
    beta=0.9,
    alphas_to_draw=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
)

plt.tight_layout()
plt.savefig("results.png")
