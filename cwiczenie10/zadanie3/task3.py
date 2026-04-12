import numpy as np


def solve_task3() -> None:
    # Definicje danych
    a_mat = np.array(
        [
            [0.4873, -0.8732],
            [0.6072, 0.7946],
            [0.9880, -0.1546],
            [-0.9768, -0.2142],
            [-0.1601, -0.9871],
            [0.9124, 0.4093],
        ]
    )
    b = np.ones(6)
    c = np.array([-0.5, 0.5])

    # Krok 1: Metoda pierwszej fazy
    m, n = a_mat.shape
    c_tilde = np.zeros(n + 1)
    c_tilde[-1] = 1.0
    a_tilde = np.hstack([a_mat, -np.ones((m, 1))])
    x_tilde = np.zeros(n + 1)
    x_tilde[-1] = 1.0 + float(np.max(-b))

    t, gamma, eps = 1.0, 2.5, 1e-5
    found_x0 = None

    while True:
        for _ in range(100):
            d = b - np.dot(a_tilde, x_tilde)
            g = c_tilde + (1 / t) * np.dot(a_tilde.T, 1 / d)
            hess_mat = (1 / t) * np.dot(a_tilde.T * (1 / d**2), a_tilde)
            dx = np.linalg.solve(hess_mat, -g)
            if -np.dot(g, dx) / 2 < 1e-5:
                break
            step = 1.0
            while np.any(b - np.dot(a_tilde, x_tilde + step * dx) <= 0):
                step *= 0.5
            x_tilde += step * dx
        if x_tilde[-1] < 0:
            found_x0 = x_tilde[:-1]
            break
        t *= gamma

    print(f"Automatycznie wyznaczony punkt startowy: {found_x0}")  # noqa: T201

    if found_x0 is None:
        return

    # Krok 2: Uruchomienie optymalizacji
    t = 1.0
    x = np.copy(found_x0)

    while (m / t) > eps:
        for _ in range(100):
            d = b - np.dot(a_mat, x)
            g = c + (1 / t) * np.dot(a_mat.T, 1 / d)
            hess_mat = (1 / t) * np.dot(a_mat.T * (1 / d**2), a_mat)
            dx = np.linalg.solve(hess_mat, -g)
            if -np.dot(g, dx) / 2 < 1e-5:
                break
            step = 1.0
            while np.any(b - np.dot(a_mat, x + step * dx) <= 0):
                step *= 0.5
            x += step * dx
        t *= gamma

    print(f"Końcowe optimum SBM: {x}")  # noqa: T201


if __name__ == "__main__":
    solve_task3()
