import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

data = loadmat("inertialData.mat")
t = data["t"].flatten()
y = data["y"].flatten()


def h(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Funkcja odpowiedzi skokowej obiektu inercyjnego.
    x[0] = k (wzmocnienie)
    x[1] = T (stała czasowa)
    """
    return x[0] * (1 - np.exp(-t / x[1]))


def f(x: np.ndarray) -> np.ndarray:
    """Wektor reszt (błędów)"""
    return h(x, t) - y


def J(x: np.ndarray) -> np.ndarray:
    """Macierz Jakobianu"""
    k, T = x[0], x[1]
    df_dk = 1 - np.exp(-t / T)
    df_dT = -k * (t / T**2) * np.exp(-t / T)
    return np.column_stack((df_dk, df_dT))


# 3. Inicjalizacja parametrów algorytmu Levenberga-Marquardta
x0 = np.array([1.0, 1.0])  # Wartości początkowe parametrów k i T (first guess)
L = 1.0  # Początkowa wartość parametru ufności lambda
k_max = 25  # Maksymalna liczba iteracji algorytmu
n = len(x0)

# Tablice do logowania historii wartości dla wykresów
X_hist = np.zeros((n, k_max + 1))
X_hist[:, 0] = x0
L_hist = np.zeros(k_max + 1)
L_hist[0] = L
f_norm_hist = np.zeros(k_max + 1)
f_norm_hist[0] = np.linalg.norm(f(x0)) ** 2

x = x0.copy()

# 4. Główna pętla algorytmu
for i in range(k_max):
    Jac = J(x)
    fx = f(x)

    # Obliczanie przyrostu: delta = - (J^T J + lambda * I)^-1 * J^T * f
    I = np.eye(n)
    # Rozwiązywanie układu równań jest wydajniejsze numerycznie niż odwracanie macierzy
    delta = np.linalg.solve(Jac.T @ Jac + L * I, -Jac.T @ fx)
    x_new = x + delta

    # Warunek modyfikacji parametru lambda
    if np.linalg.norm(f(x_new)) < np.linalg.norm(fx):
        L = 0.8 * L
        x = x_new
    else:
        L = 2.0 * L

    # Zapis logów do wykresów
    X_hist[:, i + 1] = x
    L_hist[i + 1] = L
    f_norm_hist[i + 1] = np.linalg.norm(f(x)) ** 2

# Wyniki końcowe
k_opt, T_opt = x
print(f"Zoptymalizowane parametry: k = {k_opt:.4f}, T = {T_opt:.4f}")
print("Oczekiwane odpowiedzi wg skryptu: k = 1.45, T = 1.25")

# 5. Generowanie odpowiednich wykresów

plt.figure(figsize=(10, 12))

# Wykres dopasowania modelu
plt.subplot(4, 1, 1)
plt.plot(t, y, "rs", label="measurement")
plt.plot(t, h(x0, t), "b-", label="first guess")
plt.plot(t, h(x, t), "k-", linewidth=2, label="final fit")
plt.title("Step response fitting")
plt.xlabel("t [s]")
plt.ylabel("h(t)")
plt.grid(visible=True)
plt.legend()

# Wykres ewolucji parametrów k oraz T
plt.subplot(4, 1, 2)
plt.plot(range(k_max + 1), X_hist[0, :], "k-", label="k (gain)", linewidth=2)
plt.plot(range(k_max + 1), X_hist[1, :], "b-", label="T (time constant)", linewidth=2)
plt.title("Parameters evolution")
plt.xlabel("iteration number")
plt.ylabel("Parameter value")
plt.grid(visible=True)
plt.legend()

# Wykres ewolucji parametru ufności Lambda
plt.subplot(4, 1, 3)
plt.plot(range(k_max + 1), L_hist, "ks")
plt.title(r"Trust parameter $\lambda^{(k)}$ change")
plt.xlabel("k (iteration)")
plt.ylabel(r"$\lambda$")
plt.grid(visible=True)

# Wykres wartości funkcji celu (błędu)
plt.subplot(4, 1, 4)
plt.plot(range(k_max + 1), f_norm_hist, "ks")
plt.title(r"Objective function $||f(x^{(k)})||^2$")
plt.xlabel("k (iteration)")
plt.ylabel("Error squared sum")
plt.yscale("log")  # Skala logarytmiczna dla lepszej czytelności
plt.grid(visible=True)

plt.tight_layout()
plt.savefig("task3.png")
