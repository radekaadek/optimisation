import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import signal

# Parametry transmitancji 4-go rzędu
L = 197e-6
C = 100e-9
Vin = 14.0
Vout = 1.95
omega_s = 40 * 2 * np.pi * 1e3

# Obliczenia współczynników pomocniczych
omega_0_sq = 1 / (L * C)
M = (2 * np.sqrt(Vin**2 - Vout**2)) / (np.pi * np.abs(1 - omega_s**2 * L * C))
K = (2 * omega_0_sq * Vout) / (np.pi * M * omega_s)
Sigma = omega_s**2 + omega_0_sq
Delta = omega_s**2 - omega_0_sq

# Licznik i mianownik transmitancji G(s)
num = [-M * omega_s * K, -M * omega_s * 2 * Delta]
den = [1, K, 2 * Sigma, K * Sigma, Delta**2]
system_4th = signal.TransferFunction(num, den)

# Symulacja odpowiedzi skokowej w czasie t od 0 do 3.5 ms
t = np.linspace(0, 0.0035, 1000)
t, y = signal.step(system_4th, T=t)

# Skalowanie o 10^3 ze względu na małe wartości
t_scale = t * 1000
y_scale = y * 1000


# Model odpowiedzi członu oscylacyjnego
def h_osc(x, t_val):
    k, gamma, beta = x
    return k * (
        1
        - np.exp(-gamma * t_val)
        * (np.cos(beta * t_val) + (gamma / beta) * np.sin(beta * t_val))
    )


# Wektor błędu f(x)
def f_err(x):
    return h_osc(x, t_scale) - y_scale


# Analityczny Jakobian J(x)
def jacobian(x):
    k, gamma, beta = x
    J = np.zeros((len(t_scale), 3))

    exp_term = np.exp(-gamma * t_scale)
    sin_term = np.sin(beta * t_scale)
    cos_term = np.cos(beta * t_scale)

    # Pochodne cząstkowe wg wzorów (67a, 67b, 67c)
    J[:, 0] = 1 - exp_term * (cos_term + (gamma / beta) * sin_term)
    J[:, 1] = (
        k * exp_term * (t_scale * cos_term - ((1 - t_scale * gamma) / beta) * sin_term)
    )
    J[:, 2] = (
        k
        * exp_term
        * (
            (t_scale + gamma / (beta**2)) * sin_term
            - (gamma / beta) * t_scale * cos_term
        )
    )

    return J


# ==========================================
# 4. ALGORYTM LEVENBERGA-MARQUARDTA
# ==========================================
k_max = 35
x_vals = np.zeros((3, k_max + 1))
x0 = np.array([1.0, 1.0, 1.0])  # Wartości początkowe
x_vals[:, 0] = x0

lam = 1.0  # Początkowy parametr ufności (trust parameter)
I = np.eye(3)

x_current = x0
for k_iter in range(k_max):
    J = jacobian(x_current)
    f_val = f_err(x_current)

    # Rozwiązanie układu równań LM: delta = -(J^T*J + lam*I)^(-1) * J^T * f
    delta = inv(J.T @ J + lam * I) @ (-J.T @ f_val)
    x_new = x_current + delta

    # Decyzja o zmianie kroku i lambdy (wzorowane na algorytmie B z pliku)
    if np.linalg.norm(f_err(x_new)) ** 2 < np.linalg.norm(f_val) ** 2:
        lam = 0.8 * lam
        x_current = x_new
    else:
        lam = 2.0 * lam

    x_vals[:, k_iter + 1] = x_current

x_opt_scaled = x_current

# Odzyskanie prawdziwych wartości mnożąc przez 10^(-3)
k_opt = x_opt_scaled[0] * 1e-3
gamma_opt = x_opt_scaled[1] * 1e-3
beta_opt = x_opt_scaled[2] * 1e-3

# Obliczenie współczynnika tłumienia (xi) i stałej czasowej (T)
T_opt = 1.0 / np.sqrt(beta_opt**2 + gamma_opt**2)
xi_opt = gamma_opt * T_opt

print("Zidentyfikowane parametry:")
print(f"k     = {k_opt:.6e}")
print(f"gamma = {gamma_opt:.6e}")
print(f"beta  = {beta_opt:.6e}")
print(f"T     = {T_opt:.6e}")
print(f"xi    = {xi_opt:.6e}")

# Rekonstrukcja zoptymalizowanej odpowiedzi
y_fit_scaled = h_osc(x_opt_scaled, t_scale)
y_fit = y_fit_scaled / 1000  # Powrót do oryginalnej skali do wykresu

y_guess_scaled = h_osc(x0, t_scale)
y_guess = y_guess_scaled / 1000

plt.figure(figsize=(10, 6))
plt.plot(t, y, "r-", linewidth=2, label="h (Oryginał 4-go rzędu)")
plt.plot(t, y_guess, "b-", linewidth=1.5, label="Pierwsze przybliżenie (x0)")
plt.plot(t, y_fit, "k-", linewidth=2, label=f"h_osc (Zidentyfikowany model, k={k_max})")
plt.title("Zadanie 5: Porównanie odpowiedzi skokowych")
plt.xlabel("t [s]")
plt.ylabel("Amplituda")
plt.legend()
plt.grid(True)
plt.savefig("task5_fig1.png")

# Wykres ewolucji parametrów (podobnie jak na Rysunku 14)
plt.figure(figsize=(10, 5))
plt.plot(range(k_max + 1), x_vals[0, :], "k-", linewidth=2, label="k (skalowane)")
plt.plot(range(k_max + 1), x_vals[1, :], "r-", linewidth=2, label="gamma (skalowane)")
plt.plot(range(k_max + 1), x_vals[2, :], "b-", linewidth=2, label="beta (skalowane)")
plt.xlabel("Numer iteracji k")
plt.ylabel("Wartość parametru")
plt.title("Ewolucja parametrów k, gamma i beta (w przeskalowanej dziedzinie)")
plt.legend()
plt.grid(True)
plt.savefig("task5_fig2.png")
