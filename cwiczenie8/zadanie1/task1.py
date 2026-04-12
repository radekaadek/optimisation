import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares


# Funkcja celu f(x), którą chcemy zminimalizować w sensie najmniejszych kwadratów: ||f(x)||^2
def f(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([x[0] + np.exp(-x[1]), x[0] ** 2 + 2 * x[1] + 1], dtype=np.float64)


# Funkcja więzów g(x). Szukamy rozwiązania, dla którego g(x) = 0
def g(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([x[0] + x[0] ** 3 + x[1] + x[1] ** 2], dtype=np.float64)


# Macierz Jacobiego funkcji f(x) (pochodne cząstkowe f względem x1 i x2)
def Df(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [[1.0, -np.exp(-x[1])], [2 * x[0], 2.0]],
        dtype=np.float64,
    )


# Macierz Jacobiego funkcji więzów g(x)
def Dg(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [[1 + 3 * x[0] ** 2, 1 + 2 * x[1]]],
        dtype=np.float64,
    )


# Funkcja celu dla algorytmu ALA przekazywana do solvera Levenberga-Marquardta (LM).
# Zgodnie ze wzorem (28) z instrukcji, rozszerzony lagranżjan jest minimalizowany
# poprzez minimalizację normy z wektora połączonego z f(x) oraz składnika kary.
def ala_objective(
    x: npt.NDArray[np.float64], mu: float, z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    f_val = f(x)
    g_val = g(x)
    # Składnik kary: sqrt(mu)*g(x) + 1/(2*sqrt(mu))*z
    penalty_term = np.sqrt(mu) * g_val + (1.0 / (2 * np.sqrt(mu))) * z
    # Funkcja zwraca połączony wektor: [f(x); penalty_term]
    return np.concatenate((f_val, penalty_term))


# Funkcja pomocnicza do rysowania poziomic i wykresów zbieżności
def plot_results(
    x_history: npt.NDArray[np.float64],
    fr_history: list[float],
    or_history: list[float],
) -> None:
    # Tworzenie siatki do narysowania poziomic
    x1_mesh, x2_mesh = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    z_f = np.zeros_like(x1_mesh)
    z_g = np.zeros_like(x1_mesh)

    # Obliczanie wartości ||f(x)||^2 oraz g(x) dla każdego punktu siatki
    for i in range(x1_mesh.shape[0]):
        for j in range(x1_mesh.shape[1]):
            pt = np.array([x1_mesh[i, j], x2_mesh[i, j]], dtype=np.float64)
            z_f[i, j] = np.linalg.norm(f(pt)) ** 2
            z_g[i, j] = g(pt)[0]

    plt.figure(figsize=(10, 5))

    # Wykres poziomic
    plt.subplot(1, 2, 1)
    # Poziomice funkcji celu (czarne)
    cp = plt.contour(
        x1_mesh, x2_mesh, z_f, levels=[2, 4, 6, 8, 10, 12, 14], colors="black"
    )
    plt.clabel(cp, inline=True, fontsize=8)
    # Poziomica g(x) = 0 (czerwona linia - zbiór rozwiązań dopuszczalnych)
    plt.contour(x1_mesh, x2_mesh, z_g, levels=[0], colors="red")
    # Zaznaczenie kolejnych iteracji punktu x (niebieskie kropki)
    plt.plot(x_history[:, 0], x_history[:, 1], "bo")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Poziomice i iteracje ALA")

    # Wykres zbieżności residuów w skali logarytmicznej
    plt.subplot(1, 2, 2)
    iters = list(range(len(fr_history)))
    plt.step(iters, fr_history, "bo-", label="feasibility (FR)", where="post")
    plt.step(iters, or_history, "rs-", label="opt. cond. (OR)", where="post")
    plt.yscale("log")  # Skala logarytmiczna
    plt.xlabel("kolejne iteracje ALA")
    plt.ylabel("residual")
    plt.legend()
    plt.grid(visible=True)

    plt.tight_layout()
    plt.savefig("results.png")


# Główna funkcja rozwiązująca Zadanie 1
def solve_zadanie1() -> None:
    # Punkt startowy, początkowa kara (mu) i początkowe mnożniki Lagrange'a (z)
    x = np.array([0.5, -0.5], dtype=np.float64)
    mu = 1.0
    z = np.array([0.0], dtype=np.float64)

    max_iter = 10

    fr_history: list[float] = []
    or_history: list[float] = []
    x_history_list = [x.copy()]
    total_lm_iters = 0

    print("Iter\tlog10(FR)\tlog10(OR)\tx1\tx2")

    # Główna pętla algorytmu Rozszerzonego Lagranżjanu (ALA)
    for k in range(max_iter + 1):
        # 1. Obliczenie residuów:
        # FR (Feasibility Residual) - błąd spełnienia więzów: ||g(x)||
        fr = float(np.linalg.norm(g(x)))
        # OR (Optimality Residual) - błąd warunku optymalności: ||2*Df(x)^T*f(x) + Dg(x)^T*z||
        or_val = float(np.linalg.norm(2 * Df(x).T @ f(x) + Dg(x).T @ z))

        fr_history.append(fr)
        or_history.append(or_val)

        # Wypisanie aktualnych statystyk z dodanym małym epsilonem (1e-16), aby uniknąć log(0)
        print(
            f"{k}\t{np.log10(fr + 1e-16):.4f}\t\t{np.log10(or_val + 1e-16):.4f}\t\t{x[0]:.4f}\t{x[1]:.4f}"
        )

        if k == max_iter:
            break

        # KROK 1 ALGORYTMU: Minimalizacja funkcji celu metodą Levenberga-Marquardta (LM)
        res = least_squares(ala_objective, x, args=(mu, z), method="lm")
        x_new = np.array(res.x, dtype=np.float64)
        total_lm_iters += int(res.nfev)

        g_new = g(x_new)

        # KROK 2 ALGORYTMU: Aktualizacja wektora z (mnożników Lagrange'a)
        # Zgodnie ze wzorem: z^{(k+1)} = z^{(k)} + 2*mu^{(k)}*g(x^{(k+1)})
        z_new = z + 2 * mu * g_new

        # KROK 3 ALGORYTMU: Aktualizacja parametru kary mu
        # Jeśli warunek dopuszczalności poprawił się wystarczająco (jest mniejszy niż 25% poprzedniego),
        # mu zostaje takie samo. W przeciwnym razie jest podwajane.
        mu_new = (
            mu
            if float(np.linalg.norm(g_new)) < 0.25 * float(np.linalg.norm(g(x)))
            else 2 * mu
        )

        # Zapisanie nowych wartości na następną iterację
        x = x_new
        z = z_new
        mu = mu_new
        x_history_list.append(x.copy())

    x_history = np.array(x_history_list, dtype=np.float64)
    plot_results(x_history, fr_history, or_history)


if __name__ == "__main__":
    solve_zadanie1()
