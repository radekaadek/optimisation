import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def sbm_lp(
    c: np.ndarray,
    a_mat: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    t_init: float = 1.0,
    gamma: float = 2.5,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rozwiązuje zadanie LP metodą SBM z podanym ściśle dopuszczalnym x0."""
    m, _ = a_mat.shape
    x = np.copy(x0)
    t = t_init
    path = [
        np.copy(x)
    ]  # Zmienna do przechowywania kolejnych punktów ze ścieżki centralnej

    # Definicja funkcji, gradientu i hesjanu z uwzględnieniem logarytmicznej funkcji bariery
    def func(x_val: np.ndarray, t_val: float) -> float:
        d = b - np.dot(a_mat, x_val)
        return float(np.dot(c, x_val) - (1 / t_val) * np.sum(np.log(d)))

    def grad(x_val: np.ndarray, t_val: float) -> np.ndarray:
        d = b - np.dot(a_mat, x_val)
        return c + (1 / t_val) * np.dot(a_mat.T, 1 / d)

    def hess(x_val: np.ndarray, t_val: float) -> np.ndarray:
        d = b - np.dot(a_mat, x_val)
        return (1 / t_val) * np.dot(a_mat.T * (1 / d**2), a_mat)

    # Krok centrujący (iteracja zewnętrzna algorytmu SBM)
    # Wykonujemy aż do osiągnięcia żądanej dokładności: m/t <= eps
    while (m / t) > eps:
        # Metoda Newtona (iteracje wewnętrzne) dla konkretnego t
        for _ in range(100):
            g = grad(x, t)
            hess_mat = hess(x, t)

            # Wyznaczenie kierunku Newtona
            dx = np.linalg.solve(hess_mat, -g)

            # Kryterium stopu metody Newtona oparte na dekrementu Newtona
            if -np.dot(g, dx) / 2 < 1e-5:
                break

            # Line search (wyszukiwanie długości kroku)
            step = 1.0
            # Zapobiegamy wyjściu poza obszar dopuszczalny (argumenty logarytmu > 0)
            while np.any(b - np.dot(a_mat, x + step * dx) <= 0):
                step *= 0.5

            # Sprawdzamy czy zrobienie kroku zmniejszy wartość funkcji
            f_cur = func(x, t)
            while func(x + step * dx, t) > f_cur + 0.01 * step * np.dot(g, dx):
                step *= 0.5

            # Aktualizacja wektora x
            x = x + step * dx

        # Zapisanie zaktualizowanego punktu centralnego (punktu na ścieżce centralnej)
        path.append(np.copy(x))
        # Zwiększenie parametru bariery t o współczynnik gamma
        t *= gamma

    return x, np.array(path)


if __name__ == "__main__":
    # Dane do zadania (macierz ograniczeń A i wektor b)
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
    c_test = np.array([-0.5, 0.5])  # Wektor współczynników funkcji celu
    x0_test = np.array(
        [0.0, 0.0]
    )  # Znamy ściśle dop. punkt startowy (x=0, bo wszystkie b=1 > 0)

    # Uruchomienie własnego algorytmu SBM
    x_opt, opt_path = sbm_lp(c_test, a_test, b_test, x0_test)
    # Porównanie z gotową funkcją optymalizacyjną ze Scipy
    res = linprog(c_test, A_ub=a_test, b_ub=b_test, bounds=(None, None))

    print(f"Wynik SBM: {x_opt}")  # noqa: T201
    print(f"Wynik linprog: {res.x}")  # noqa: T201

    # Rysowanie wykresu obszaru dopuszczalnego i ścieżki
    # v_mat zawiera współrzędne wierzchołków wielokomórki (obliczone zewnętrznie)
    v_mat = np.array(
        [
            [0.1562, 0.9127, 1.0338, 0.8086, -1.3895, -0.8782],
            [-1.0580, -0.6358, 0.1386, 0.6406, 2.3203, -0.8311],
        ]
    )

    plt.figure(figsize=(8, 6))
    # Rysowanie szarego obszaru wielokomórki (obszar dopuszczalny)
    plt.fill(v_mat[0, :], v_mat[1, :], "lightgray", edgecolor="black")

    # Tworzenie siatki i poziomicy funkcji celu
    X1, X2 = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-1.5, 3, 100))
    Z = c_test[0] * X1 + c_test[1] * X2
    plt.contour(X1, X2, Z, 20, linestyles="dashed", cmap="jet")
    plt.colorbar()

    # Zaznaczanie punktów ze ścieżki centralnej wyliczonych przez metodę
    plt.plot(opt_path[:, 0], opt_path[:, 1], "ko", markersize=4)
    plt.plot(x0_test[0], x0_test[1], "ks")  # Start - czarny kwadrat
    plt.plot(x_opt[0], x_opt[1], "ro")  # Optimum - czerwona kropka
    plt.title("Ścieżka centralna SBM")
    plt.savefig("zadanie2_wykres.png")
