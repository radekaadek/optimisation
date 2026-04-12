import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


# --- 1. Definicja funkcji modelu i Jakobianu ---
def h_model(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Odpowiedź skokowa układu dwuinercyjnego."""
    k, T1, T2 = x[0], x[1], x[2]
    # Zabezpieczenie przed dzieleniem przez zero, gdy T1 == T2
    if np.isclose(T1, T2):
        T1 += 1e-5
    return k * (1 - 1 / (T1 - T2) * (T1 * np.exp(-t / T1) - T2 * np.exp(-t / T2)))


def jacobian(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Analityczny Jakobian modelu."""
    k, T1, T2 = x[0], x[1], x[2]
    if np.isclose(T1, T2):
        T1 += 1e-5

    df_dk = 1 - 1 / (T1 - T2) * (T1 * np.exp(-t / T1) - T2 * np.exp(-t / T2))
    df_dT1 = (k / (T2 - T1)) * (
        (t / T1) * np.exp(-t / T1)
        + (T2 / (T1 - T2)) * (np.exp(-t / T2) - np.exp(-t / T1))
    )
    df_dT2 = (k / (T1 - T2)) * (
        (t / T2) * np.exp(-t / T2)
        + (T1 / (T2 - T1)) * (np.exp(-t / T1) - np.exp(-t / T2))
    )

    return np.column_stack((df_dk, df_dT1, df_dT2))


# --- 2. Wczytanie lub wygenerowanie danych ---
file_name = "twoInertialData.mat"
if Path.exists(file_name):
    data = sio.loadmat(file_name)
    t = data["t"].flatten()
    y = data["y"].flatten()

# --- 3. Inicjalizacja algorytmu LM ---
# Wartości początkowe parametrów (pierwsze przybliżenie)
# Ustawiamy T1 różniące się od T2, aby uniknąć błędu Dzielenia przez zero we wzorach
x0 = np.array([1.0, 1.0, 2.0])
n = len(x0)
k_max = 35

X_history = np.zeros((n, k_max + 1))
X_history[:, 0] = x0
lambda_param = 1.0
lambda_history = [lambda_param]
obj_history = []

x_current = x0

# --- 4. Główna pętla algorytmu Levenberga-Marquardta ---
for k_iter in range(k_max):
    # Aktualne błędy i Jakobian
    f_current = h_model(x_current, t) - y
    J_current = jacobian(x_current, t)

    obj_history.append(np.linalg.norm(f_current) ** 2)

    # Krok zgodnie ze wzorem (26)
    I = np.eye(n)
    delta_x = np.linalg.solve(
        J_current.T @ J_current + lambda_param * I, -J_current.T @ f_current
    )
    x_new = x_current + delta_x

    # Ocena nowego punktu
    f_new = h_model(x_new, t) - y

    if np.linalg.norm(f_new) ** 2 < np.linalg.norm(f_current) ** 2:
        x_current = x_new
        lambda_param *= 0.8
    else:
        lambda_param *= 2.0

    X_history[:, k_iter + 1] = x_current
    lambda_history.append(lambda_param)

# Ostatni punkt do historii błędów
obj_history.append(np.linalg.norm(h_model(x_current, t) - y) ** 2)

# --- 5. Rysowanie wykresów ---
x_optimal = X_history[:, -1]
print(
    f"Oszacowane parametry: k = {x_optimal[0]:.4f}, T1 = {x_optimal[1]:.4f}, T2 = {x_optimal[2]:.4f}"
)

plt.figure(figsize=(10, 10))

# Wykres 1: Dopasowanie krzywej
plt.subplot(3, 1, 1)
plt.plot(t, y, "rs", label="Pomiary", markersize=4)
plt.plot(t, h_model(x0, t), "b-", label="Pierwsze przybliżenie", linewidth=2)
plt.plot(t, h_model(x_optimal, t), "k-", label="Dopasowanie LM", linewidth=2)
plt.xlabel("Czas t [s]")
plt.ylabel("h(t)")
plt.legend()
plt.grid(True)
plt.title("Dopasowanie modelu do układu dwuinercyjnego")

# Wykres 2: Ewolucja parametrów
plt.subplot(3, 1, 2)
plt.plot(range(k_max + 1), X_history[0, :], "k-", label="k", linewidth=2)
plt.plot(range(k_max + 1), X_history[1, :], "b-", label="T1", linewidth=2)
plt.plot(range(k_max + 1), X_history[2, :], "g-", label="T2", linewidth=2)
plt.xlabel("Numer iteracji k")
plt.ylabel("Wartość parametrów")
plt.legend()
plt.grid(True)

# Wykres 3: Funkcja celu
plt.subplot(3, 1, 3)
plt.semilogy(range(k_max + 1), obj_history, "ks")
plt.xlabel("Numer iteracji k")
plt.ylabel("||f(x_k)||^2 (skala log)")
plt.grid(True)

plt.tight_layout()
plt.savefig("task4.png")
