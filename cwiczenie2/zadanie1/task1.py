from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.io import loadmat


def load_isoperimetric_data(filepath: str) -> dict[str, Any]:
    """Wczytuje i wyciąga zmienne z pliku danych MATLABa."""
    data = loadmat(filepath)

    # N to liczba przedziałów, a_val to długość całkowitego przedziału na osi X
    n_nodes = int(data["N"].item())
    a_val = float(data["a"].item())

    # L to maksymalna dozwolona długość krzywej
    length_limit = float(data["L"].item())

    # C to maksymalna dozwolona krzywizna (ograniczenie na drugą pochodną)
    max_curvature = float(data["C"].item())

    # F to indeksy punktów przez które krzywa musi przejść
    # Odejmujemy 1, ponieważ w MATLABie indeksuje się od 1, a w Pythonie od 0.
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
    """Generuje podstawowe ograniczenia (constraints) dla problemu izoperymetrycznego."""
    constraints: list[cp.Constraint] = []

    # Ograniczenie 1: Maksymalna długość krzywej <= L
    # dy to wektor różnic y_{i+1} - y_i
    dy = cp.diff(y)
    # dx to wektor stałych kroków dyskretyzacji h
    dx = np.full(n_nodes, h_step)

    # Obliczamy normy wektorów [h, y_{i+1}-y_i] za pomocą cvxpy
    # Używamy cp.norm(..., axis=0), aby policzyć normę (długość przeciwprostokątnej) dla każdego segmentu.
    segment_lengths = cp.norm(cp.vstack([dx, dy]), axis=0)

    # Suma długości segmentów nie może przekroczyć zadanej długości L
    constraints.append(cp.sum(segment_lengths) <= length_limit)

    # Ograniczenie 2: Warunki brzegowe. Krzywa zaczyna się i kończy w 0 na osi Y.
    constraints.append(y[0] == 0)
    constraints.append(y[n_nodes] == 0)

    # Ograniczenie 3: Punkty stałe. W z góry zadanych węzłach F, wartość y musi być równa y_fixed
    constraints.append(y[fixed_indices] == y_fixed[fixed_indices])

    # Ograniczenie 4: Ograniczenie krzywizny
    if include_curvature:
        # cp.diff(y, 2) oblicza różnicę drugiego rzędu: y_{i+2} - 2y_{i+1} + y_i
        d2y = cp.diff(y, 2)
        # Przybliżenie drugiej pochodnej nie może przekroczyć wartości C w obu kierunkach (|f''(x)| <= C)
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
    """Konstruuje i rozwiązuje pojedynczą odmianę problemu optymalizacyjnego."""

    # Definiujemy zmienną optymalizacyjną y, to wektor punktów y_1, ..., y_{N+1}
    y = cp.Variable(n_nodes + 1)

    # Funkcja celu to pole pod krzywą, przybliżone metodą prostokątów: h * suma(y_i)
    # y[:-1] bierze punkty od 0 do N-1, co odpowiada i=1...N w notacji matematycznej.
    area = h_step * cp.sum(y[:-1])

    # Pobieramy zestaw podstawowych ograniczeń opisanych wyżej
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

    # Wariant do punktu (b) instrukcji: dodatkowe ograniczenie, że y >= 0
    if non_negative:
        constraints.append(y >= 0)

    # Wybór kierunku optymalizacji w zależności od zadania
    if objective_type == "maximize":
        prob = cp.Problem(cp.Maximize(area), constraints)
    else:
        # Wykorzystywane w podpunkcie (a) - minimalizacja pola pod krzywą
        prob = cp.Problem(cp.Minimize(area), constraints)

    # Rozwiązujemy zdefiniowany problem optymalizacyjny (wypukły)
    prob.solve()

    # Obsługa przypadku, gdy solver zwróci None dla danej wartości area
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
    """Funkcja pomocnicza do tworzenia wykresów."""
    ax.plot(x_vals, y_vals, "k-")
    ax.plot(x_vals[fixed_indices], y_fixed[fixed_indices], "ro")
    ax.set_title(title)
    ax.set_xlabel("x / a")  # Znormalizowana oś X zgodnie z instrukcją
    ax.set_ylabel("y(x)")
    ax.grid(visible=True, linestyle="--", alpha=0.5)


def run_isoperimetric_tasks() -> None:
    """Główna funkcja wykonująca poszczególne podpunkty zadania."""
    data = load_isoperimetric_data("isoPerimData.mat")

    n_nodes = data["n_nodes"]
    a_val = data["a_val"]
    length_limit = data["length_limit"]
    max_curvature = data["max_curvature"]
    fixed_indices = data["fixed_indices"]
    y_fixed = data["y_fixed"]

    # Obliczamy krok dyskretyzacji h = a / N
    h_step = a_val / n_nodes
    # Generujemy zmienną z i od razu ją normalizujemy z/a do celów wyświetlania
    x_vals = np.linspace(0, a_val, n_nodes + 1) / a_val

    # Zadanie 1: Wymagana podstawowa optymalizacja (maksymalizacja)
    y1, area1 = solve_problem(
        "maximize", n_nodes, h_step, length_limit, fixed_indices, y_fixed, max_curvature
    )
    print(f"Task 1 (Base Maximize) Area: {area1:.4f}")

    # Zadanie 2(a): Minimalizacja pola pod krzywą (Zadanie modyfikujące a)
    y_a, area_a = solve_problem(
        "minimize", n_nodes, h_step, length_limit, fixed_indices, y_fixed, max_curvature
    )
    print(f"Task (a) Minimum Area: {area_a:.4f}")

    # Zadanie 2(b): Minimalizacja pola przy nieujemnych zmiennych optymalizacyjnych (Zadanie modyfikujące b)
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

    # Zadanie 2(c): Maksymalizacja pola bez ograniczenia maksymalnej krzywizny (Zadanie modyfikujące c)
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

    # Rysowanie wykresów, podzielone na 4 panele dla poszczególnych modyfikacji
    _, axs = plt.subplots(2, 2, figsize=(12, 8))

    plot_task(
        axs[0, 0], x_vals, y1.value, fixed_indices, y_fixed, "Task 1: Base Maximization"
    )
    plot_task(
        axs[0, 1], x_vals, y_a.value, fixed_indices, y_fixed, "Task (a): Minimization"
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
    plt.savefig("isoperimetric_results.png") # Saves to your current folder
    print("Plot saved as isoperimetric_results.png")


if __name__ == "__main__":
    run_isoperimetric_tasks()
