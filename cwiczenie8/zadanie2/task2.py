import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares


# Zadanie polega na znalezieniu punktu na krzywej (określonej przez dwa równania),
# który jest najbliżej punktu (1, 1, 1). Minimalizujemy więc dystans kwadratowy.
def f(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([x[0] - 1.0, x[1] - 1.0, x[2] - 1.0], dtype=np.float64)


# Mamy tu układ dwóch nieliniowych równań (więzów równościowych)
def g(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    g1 = x[0] ** 2 + 0.5 * x[1] ** 2 + x[2] ** 2 - 1.0
    g2 = (
        0.8 * x[0] ** 2
        + 2.5 * x[1] ** 2
        + x[2] ** 2
        + 2 * x[0] * x[2]
        - x[0]
        - x[1]
        - x[2]
        - 1.0
    )
    return np.array([g1, g2], dtype=np.float64)


# Macierz Jacobiego (pochodne) dla f(x). Pochodna z (x_i - 1) to po prostu 1.
def Df(_x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Zmienna _x jest oznaczona podkreślnikiem, ponieważ macierz Df jest stała (macierz jednostkowa)
    return np.eye(3, dtype=np.float64)


# Macierz Jacobiego dla dwóch funkcji więzów
def Dg(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    dg1 = [2.0 * x[0], x[1], 2.0 * x[2]]
    dg2 = [
        1.6 * x[0] + 2.0 * x[2] - 1.0,
        5.0 * x[1] - 1.0,
        2.0 * x[2] + 2.0 * x[0] - 1.0,
    ]
    return np.array([dg1, dg2], dtype=np.float64)


# Uniwersalna funkcja celu obsługująca zarówno metodę ALA jak i klasyczną Metodę Kary (Penalty)
def objective(
    x: npt.NDArray[np.float64],
    mu: float,
    z: npt.NDArray[np.float64],
    method_type: str = "ala",
) -> npt.NDArray[np.float64]:
    if method_type == "ala":
        # Składnik kary dla ALA posiada też mnożniki Lagrange'a (z)
        penalty = np.sqrt(mu) * g(x) + (1.0 / (2.0 * np.sqrt(mu))) * z
    else:  # penalty
        # Klasyczna metoda kary ma w składniku kary tylko parametr mu
        penalty = np.sqrt(mu) * g(x)
    return np.concatenate((f(x), penalty))


# Główna funkcja wykonująca optymalizację w zależności od wybranego algorytmu
def solve_optimization(
    method: str = "ala",
) -> tuple[npt.NDArray[np.float64], list[int], list[float], list[float], list[float]]:
    # Punkty startowe (wymagane w treści zadania: x=0, mu=1, z=0)
    x = np.zeros(3, dtype=np.float64)
    mu = 1.0
    z = np.zeros(2, dtype=np.float64)

    fr_list: list[float] = []
    or_list: list[float] = []
    mu_list: list[float] = []
    lm_iters: list[int] = []
    cum_lm = 0

    for _ in range(100):  # Maksymalnie 100 iteracji zewnętrznych
        # Obliczenie Feasibility Residual (błąd spełnienia więzów)
        fr_val = float(np.linalg.norm(g(x)))

        # Obliczenie Optimality Condition Residual
        if method == "ala":
            or_val = float(np.linalg.norm(2 * Df(x).T @ f(x) + Dg(x).T @ z))
        else:
            # Dla metody kary nie mamy mnożników z, ale teoretycznie ich rolę pełni wyrażenie 2*mu*g(x).
            # Zatem postępujemy zgodnie ze wzorem zastępując z.
            or_val = float(
                np.linalg.norm(2 * Df(x).T @ f(x) + Dg(x).T @ (2 * mu * g(x)))
            )

        # Zbieranie statystyk do wykresów
        fr_list.append(fr_val)
        or_list.append(or_val)
        mu_list.append(mu)
        lm_iters.append(cum_lm)

        # Warunek stopu (zgodnie z poleceniem w zadaniu oba błędy mają być mniejsze niż 1e-5)
        if fr_val < 1e-5 and or_val < 1e-5:
            break

        # Uruchomienie solvera Levenberga-Marquardta dla zdefiniowanej wcześniej funkcji
        res = least_squares(objective, x, args=(mu, z, method), method="lm")
        cum_lm += int(res.nfev)  # Aktualizacja łącznej liczby ewaluacji/iteracji LM
        x_new = np.array(res.x, dtype=np.float64)
        g_new = g(x_new)

        # Zasady aktualizacji w zależności od metody
        if method == "ala":
            # W ALA aktualizujemy mnożniki z
            z_new = z + 2 * mu * g_new
            # Sprytna aktualizacja mu - rośnie tylko gdy warunek FR nie poprawił się o czynnik 0.25
            if float(np.linalg.norm(g_new)) < 0.25 * float(np.linalg.norm(g(x))):
                mu_new = mu
            else:
                mu_new = 2.0 * mu
            z = z_new
        else:  # Metoda kary
            # W metodzie kary po prostu naiwnie zwiększamy parametr mu w każdej iteracji
            # Co może prowadzić do problemów numerycznych dla bardzo dużych wartości
            mu_new = 2.0 * mu

        x = x_new
        mu = mu_new

    return x, lm_iters, fr_list, or_list, mu_list


def solve_zadanie2() -> None:
    # 1. Rozwiązanie przy pomocy ALA
    x_ala, lm_ala, fr_ala, or_ala, mu_ala = solve_optimization("ala")
    print(f"Rozwiązanie ALA: x* = {x_ala}")

    # 2. Rozwiązanie przy pomocy Metody Kary
    x_pen, lm_pen, fr_pen, or_pen, mu_pen = solve_optimization("penalty")
    print(f"Rozwiązanie Kary: x* = {x_pen}")

    # Generowanie wykresów porównujących obie metody
    plt.figure(figsize=(12, 5))

    # Wykres residuów (błędów) względem liczby iteracji w LM (sposób na rzetelne porównanie wysiłku obl.)
    plt.subplot(1, 2, 1)
    plt.step(lm_ala, fr_ala, label="FR (ALA)", where="post")
    plt.step(lm_ala, or_ala, label="OR (ALA)", where="post")
    plt.step(lm_pen, fr_pen, "--", label="FR (Penalty)", where="post")
    plt.step(lm_pen, or_pen, "--", label="OR (Penalty)", where="post")
    plt.yscale("log")
    plt.xlabel("Skumulowane iteracje LM")
    plt.ylabel("Residua")
    plt.legend()
    plt.title("Porównanie zbieżności: ALA vs Penalty")

    # Wykres parametru mu względem iteracji. W metodzie kary mu rośnie wykładniczo bez przerw.
    plt.subplot(1, 2, 2)
    plt.step(lm_ala, mu_ala, label="mu (ALA)", where="post")
    plt.step(lm_pen, mu_pen, "--", label="mu (Penalty)", where="post")
    plt.yscale("log")
    plt.xlabel("Skumulowane iteracje LM")
    plt.ylabel("Wartość parametru mu")
    plt.legend()
    plt.title("Zmiana parametru kary")

    plt.tight_layout()
    plt.savefig("results.png")


if __name__ == "__main__":
    solve_zadanie2()
