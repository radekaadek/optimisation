import numpy as np


def linprog_sbm_solver(
    c: np.ndarray,
    a_mat: np.ndarray,
    b: np.ndarray,
    t_init: float = 1.0,
    gamma: float = 2.5,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Uniwersalny solver LP bazujący na Sekwencyjnej Metodzie Bariery.
    Automatycznie rozwiązuje zadanie pierwszej fazy i przechodzi do optymalizacji.
    """
    m, n = a_mat.shape

    # ============ Faza 1: Punkt dopuszczalny ============
    # Cel: Sprowadzenie problemu poszukiwania startowego X
    # do problemu optymalizacyjnego z dodatkową zmienną pomocniczą.
    c_tilde = np.zeros(n + 1)
    c_tilde[-1] = 1.0
    a_tilde = np.hstack([a_mat, -np.ones((m, 1))])
    x_tilde = np.zeros(n + 1)
    x_tilde[-1] = 1.0 + float(
        np.max(-b)
    )  # Gwarancja dopuszczalności w pomocniczym problemie

    t = t_init
    while True:
        # Pętla Newtona
        for _ in range(100):
            d = b - np.dot(a_tilde, x_tilde)
            g = c_tilde + (1 / t) * np.dot(a_tilde.T, 1 / d)
            hess_mat = (1 / t) * np.dot(a_tilde.T * (1 / d**2), a_tilde)
            try:
                dx = np.linalg.solve(hess_mat, -g)
            except np.linalg.LinAlgError:
                break

            # Kryterium stopu wyliczania kroku
            if -np.dot(g, dx) / 2 < 1e-5:
                break

            # Line search ograniczający krok Newtona
            step = 1.0
            while np.any(b - np.dot(a_tilde, x_tilde + step * dx) <= 0):
                step *= 0.5
            x_tilde += step * dx

        # PrzerwijFazę 1 jeżeli pomocnicza zmienna 's' przyjmie wartość mniejszą od zera
        if x_tilde[-1] < 0:
            break

        # Zabezpieczenie: jeśli t urosło za bardzo, a s nadal > 0, to
        # problem oryginalny może nie mieć rozwiązań (infeasible).
        if (m / t) <= eps:
            raise ValueError("Zadanie jest sprzeczne (infeasible).")
        t *= gamma

    x0 = x_tilde[:-1]  # Odcięcie zmiennej 's' – otrzymujemy punkt startowy w R^n

    # ============ Faza 2: Optymalizacja docelowa ============
    # Cel: Mając x0 z dziedziny zadania, stosujemy oryginalne zadanie optymalizacji
    # SBM (Metoda Punktu Wewnętrznego) w celu zminimalizowania funkcji celu.
    x = np.copy(x0)
    t = t_init

    # Pętla iteracji zewnętrznej (kolejne kroki centrujące)
    while (m / t) > eps:
        # Metoda Newtona z logarytmiczną funkcją bariery
        for _ in range(100):
            d = b - np.dot(a_mat, x)
            # Obliczenie analityczne gradientu i hesjanu
            g = c + (1 / t) * np.dot(a_mat.T, 1 / d)
            hess_mat = (1 / t) * np.dot(a_mat.T * (1 / d**2), a_mat)
            dx = np.linalg.solve(hess_mat, -g)

            if -np.dot(g, dx) / 2 < 1e-5:
                break

            # Nawrót w ramach wyszukiwania kierunku
            step = 1.0
            while np.any(b - np.dot(a_mat, x + step * dx) <= 0):
                step *= 0.5
            x += step * dx

        t *= gamma  # Aktualizacja współczynnika bariery

    return x  # Zwracamy optymalny wektor z rozwiązania problemu z zadaną precyzją


if __name__ == "__main__":
    a_test = np.array(
        [
            [0.4873, -0.8732],
            [0.6072, 0.7946],
            [0.9880, -0.1546],
            [-0.9768, -0.2142],
            [-0.1601, -0.9871],
            [0.9124, 0.4093],
        ]
    )
    b_test = np.ones(6)
    c_test = np.array([-0.5, 0.5])

    x_opt = linprog_sbm_solver(c_test, a_test, b_test)
    print("Ostateczny wynik z uniwersalnego solvera:", x_opt)  # noqa: T201
