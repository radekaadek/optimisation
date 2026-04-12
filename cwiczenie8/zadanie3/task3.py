import itertools

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares


# Funkcja generująca sztuczne dane do problemu (macierz A oraz wektor b z nałożonym szumem v)
def generate_problem(
    m: int, n: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(42)  # Ustawione ziarno ułatwia odtwarzalność
    a_matrix = rng.standard_normal((m, n))
    x_true = np.sign(
        rng.standard_normal(n)
    )  # Dokładne wektory binarne (złożone z 1 i -1)
    v = rng.standard_normal(m) * 0.1  # Szum wektora
    b = a_matrix @ x_true + v
    return a_matrix, b


# Metoda Brute Force (siłowa). Generuje wszystkie 2^n możliwe kombinacje i znajduje absolutne minimum.
# Działa tylko dla małych 'n' (np. około n=10, max n~30), dla większych zajęłoby to lata.
def brute_force_boolean_ls(
    a_matrix: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64] | None, float]:
    n = a_matrix.shape[1]
    best_x = None
    best_val = float("inf")

    # Użycie itertools do wygenerowania każdej możliwej sekwencji z (-1, 1)
    for x_tuple in itertools.product([-1, 1], repeat=n):
        x = np.array(x_tuple, dtype=np.float64)
        val = float(np.linalg.norm(a_matrix @ x - b) ** 2)
        if val < best_val:
            best_val = val
            best_x = x

    return best_x, best_val


# Funkcja celu ALA zaadaptowana dla logiki binarnej
def boolean_ls_ala_objective(
    x: npt.NDArray[np.float64],
    a_matrix: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    mu: float,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # Część optymalizowana ||Ax - b||
    f_val = a_matrix @ x - b
    # Więzy: chcemy aby x_i^2 = 1, zatem g(x) = x^2 - 1 = 0 (aby x było +/- 1)
    g_val = x**2 - 1.0

    # Standardowe połączenie funkcji i członu kary ALA
    penalty = np.sqrt(mu) * g_val + (1.0 / (2 * np.sqrt(mu))) * z
    return np.concatenate((f_val, penalty))


# Heurystyczne zastosowanie ALA dla Boolean Least Squares.
# "Heurystyczne", ponieważ ALA natywnie daje ułamkowe wyniki, ale my wymuszamy na każdym
# etapie zaokrąglenie do 1 lub -1, po czym ewaluujemy jak dobrze taki punkt realizuje nasze zadanie.
def solve_boolean_ls_ala(
    a_matrix: npt.NDArray[np.float64], b: npt.NDArray[np.float64], max_iter: int = 20
) -> tuple[npt.NDArray[np.float64] | None, float]:
    n = a_matrix.shape[1]

    # Zgodnie ze wskazówką z polecenia, jako punkt startowy bierzemy zrelaksowane rozwiązanie
    # najmniejszych kwadratów bez więzów (funkcja lstsq znajduje x dla min ||Ax - b||^2).
    x, _, _, _ = np.linalg.lstsq(a_matrix, b, rcond=None)
    x = np.array(x, dtype=np.float64)
    mu = 1.0
    z = np.zeros(n, dtype=np.float64)

    best_rounded_x = None
    best_rounded_val = float("inf")

    for _ in range(max_iter):
        g_x = x**2 - 1.0

        # Heurystyka: Na każdym etapie zaokrąglamy zrelaksowany wynik ALA do najbliższego +/- 1
        x_rounded = np.sign(x)
        x_rounded[x_rounded == 0] = 1  # Zabezpieczenie przed x=0
        rounded_val = float(np.linalg.norm(a_matrix @ x_rounded - b) ** 2)

        # Zachowanie najlepszego wyniku po zaokrągleniu z całego przebiegu iteracji
        if rounded_val < best_rounded_val:
            best_rounded_val = rounded_val
            best_rounded_x = x_rounded.copy()

        # Uruchomienie kroku minimalizacji przy pomocy algorytmu LM
        res = least_squares(
            boolean_ls_ala_objective, x, args=(a_matrix, b, mu, z), method="lm"
        )
        x_new = np.array(res.x, dtype=np.float64)
        g_new = x_new**2 - 1.0

        # Aktualizacja mnożników z
        z_new = z + 2 * mu * g_new

        # Aktualizacja parametru mu (według standardowej zasady oceny postępu dopuszczalności)
        mu_new = (
            mu
            if float(np.linalg.norm(g_new)) < 0.25 * float(np.linalg.norm(g_x))
            else 2 * mu
        )

        x = x_new
        z = z_new
        mu = mu_new

    return best_rounded_x, best_rounded_val


def solve_zadanie3() -> None:
    print("--- Mały problem (m=n=10) ---")
    # Dla małych macierzy (10x10) jesteśmy w stanie policzyć wynik idealny
    a_small, b_small = generate_problem(10, 10)

    _, val_bf = brute_force_boolean_ls(a_small, b_small)
    print(f"Rozwiązanie Brute Force: {val_bf:.4f}")

    _, val_ala = solve_boolean_ls_ala(a_small, b_small)
    print(f"Rozwiązanie ALA (heurystyka): {val_ala:.4f}")
    # Oczekujemy, że wynik z ALA będzie bliski, a wręcz identyczny jak z metody Brute Force

    print("\n--- Duży problem (m=n=500) ---")
    # Z macierzą 500x500 metoda brute force szukałaby wariacji z 2^500 opcji. Nie ma sensu tego robić.
    a_large, b_large = generate_problem(500, 500)
    # Odpalamy tylko ALA (jako szybką, inteligentną heurystykę)
    _, val_ala_large = solve_boolean_ls_ala(a_large, b_large, max_iter=30)
    print(f"Wartość funkcji celu (zaokrąglone x): {val_ala_large:.4f}")


if __name__ == "__main__":
    solve_zadanie3()
