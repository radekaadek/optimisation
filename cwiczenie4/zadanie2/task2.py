import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

# 1. Definicja i generacja oryginalnego sygnału y(t)
t = np.linspace(0, 1, 501)
a_t = 1 + 0.5 * np.sin(11 * t)
theta_t = 30 * np.sin(5 * t)
y_sig = a_t * np.sin(theta_t)

# Chwilowa częstotliwość do wykresu (analityczna)
omega_t = 150 * np.abs(np.cos(5 * t))

# 2. Generowanie nadkompletnej bazy sygnałów Gabora
sigma = 0.05
tau_vals = np.linspace(0, 1, 501)
omega_vals = np.arange(0, 155, 5)

A_cols = []
dict_params = []

for tau in tau_vals:
    envelope = np.exp(-((t - tau) ** 2) / (2 * sigma**2))
    for omega in omega_vals:
        if omega == 0:
            basis_c = envelope * np.cos(omega * t)
            A_cols.append(basis_c)
            dict_params.append((tau, omega, "cos"))
        else:
            basis_c = envelope * np.cos(omega * t)
            basis_s = envelope * np.sin(omega * t)
            A_cols.append(basis_c)
            A_cols.append(basis_s)
            dict_params.append((tau, omega, "cos"))
            dict_params.append((tau, omega, "sin"))

A_mat = np.column_stack(A_cols)

# 3. Poszukiwanie bazy korzystając z scikit-learn (Lasso)
# Funkcja celu w sklearn to: (1 / (2 * n_samples)) * ||Xw - y||^2_2 + alpha * ||w||_1
# Instrukcja z ćwiczenia zakłada: ||Xw - y||^2_2 + gamma * ||w||_1
# Aby algorytmy były równoważne, musimy przeskalować parametr gamma (gdzie gamma = 1)
gamma = 1.0
n_samples = len(y_sig)
alpha = gamma / (2 * n_samples)

print("Uruchamianie algorytmu Coordinate Descent (Lasso)...")
lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
lasso.fit(A_mat, y_sig)
x_val = lasso.coef_

# 4. Wyznaczenie rzadkiej bazy B i finalnego dopasowania LS
threshold = 1e-4
B_indices = np.where(np.abs(x_val) > threshold)[0]
print(f"Liczba niezerowych elementów w bazie: {len(B_indices)} (oczekiwane około 42)")

# Estymacja metodą najmniejszych kwadratów tylko dla wybranych elementów bazy
A_B = A_mat[:, B_indices]
x_ls, _, _, _ = np.linalg.lstsq(A_B, y_sig, rcond=None)
y_hat = A_B @ x_ls

# Błąd dopasowania
error_wzgledny = np.mean((y_sig - y_hat) ** 2) / np.mean(y_sig**2)
print(f"Błąd względny aproksymacji: {error_wzgledny:.2e}")

# 5. Wykresy
plt.figure(figsize=(10, 8))

# Panel 1: Porównanie y(t) oraz y_hat(t)
plt.subplot(3, 1, 1)
plt.plot(t, y_sig, "b-", label="Sygnał oryginalny y(t)")
plt.plot(t, y_hat, "r--", label=r"Aproksymacja $\hat{y}(t)$")
plt.title("Porównanie sygnału oryginalnego z aproksymacją rzadką")
plt.legend(loc="lower right")

# Panel 2: Różnica dopasowania
plt.subplot(3, 1, 2)
plt.plot(t, y_sig - y_hat, "k-")
plt.title(r"Błąd aproksymacji $y(t) - \hat{y}(t)$")

# Panel 3: Wykres czasowo-częstotliwościowy
plt.subplot(3, 1, 3)
plt.plot(t, omega_t, "k--", label=r"Chwilowa częstość $\omega(t)$")
tau_plot = [dict_params[i][0] for i in B_indices]
omega_plot = [dict_params[i][1] for i in B_indices]
plt.scatter(
    tau_plot,
    omega_plot,
    edgecolors="r",
    facecolors="none",
    s=50,
    label="Wybrane funkcje bazowe",
)
plt.xlabel("Czas (t)")
plt.ylabel(r"Częstość ($\omega$)")
plt.title("Wykres czasowo-częstotliwościowy")
plt.legend(loc="lower center")

plt.tight_layout()
plt.show()
