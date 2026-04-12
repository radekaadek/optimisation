import numpy as np


def find_feasible_point(
    a_mat: np.ndarray,
    b: np.ndarray,
    t_init: float = 1.0,
    gamma: float = 2.5,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Znajduje ściśle dopuszczalny punkt startowy spełniający a_mat*x < b.
    Korzysta z zadania pierwszej fazy rozwiązywanego metodą SBM.
    """
    m, n = a_mat.shape

    # Przekształcenie do zadania: min s, s.t. a_mat*x - s <= b
    # x_tilde = [x, s]
    c_tilde = np.zeros(n + 1)
    c_tilde[-1] = 1.0  # minimalizujemy tylko zmienną s
    a_tilde = np.hstack([a_mat, -np.ones((m, 1))])

    # Ściśle dopuszczalny punkt startowy dla pierwszej fazy
    x_tilde = np.zeros(n + 1)
    x_tilde[-1] = 1.0 + float(
        np.max(-b)
    )  # s0 tak dobrane, aby a_mat*x - s0 < b dla x=0

    t = t_init

    def func(xt: np.ndarray) -> float:
        d = b - np.dot(a_tilde, xt)
        return float(np.dot(c_tilde, xt) - (1 / t) * np.sum(np.log(d)))

    def grad(xt: np.ndarray) -> np.ndarray:
        d = b - np.dot(a_tilde, xt)
        return c_tilde + (1 / t) * np.dot(a_tilde.T, 1 / d)

    def hess(xt: np.ndarray) -> np.ndarray:
        d = b - np.dot(a_tilde, xt)
        return (1 / t) * np.dot(a_tilde.T * (1 / d**2), a_tilde)

    while True:
        # Metoda Newtona z tłumieniem dla zadania z danym t
        for _ in range(100):
            g = grad(x_tilde)
            hess_mat = hess(x_tilde)
            try:
                dx = np.linalg.solve(hess_mat, -g)
            except np.linalg.LinAlgError:
                break

            # Warunek stopu Newtona
            if -np.dot(g, dx) / 2 < 1e-5:
                break

            # Line search (tłumienie) - warunek przynależności do dziedziny
            step = 1.0
            while np.any(b - np.dot(a_tilde, x_tilde + step * dx) <= 0):
                step *= 0.5

            # Line search - warunek spadku funkcji
            f_cur = func(x_tilde)
            while func(x_tilde + step * dx) > f_cur + 0.01 * step * np.dot(g, dx):
                step *= 0.5

            x_tilde = x_tilde + step * dx

        # Jeśli osiągnęliśmy s < 0, to x_tilde[:-1] jest ściśle dopuszczalne
        if x_tilde[-1] < 0:
            return x_tilde[:-1]

        if (m / t) <= eps:
            raise ValueError(
                "Zadanie nie posiada punktu ściśle dopuszczalnego (infeasible)."
            )
        t *= gamma


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
    x_feas = find_feasible_point(a_test, b_test)
    print("Znaleziony punkt ściśle dopuszczalny:", x_feas)  # noqa: T201
