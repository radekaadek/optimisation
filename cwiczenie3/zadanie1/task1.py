import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Wczytanie danych pomiarowych w formacie .mat
data = loadmat("Data01.mat")

# y_tilde to wektor zaszumionych pomiarów (tylda y z instrukcji)
y_tilde = data["y"].flatten()
t = data["t"].flatten()
n = len(y_tilde)

# ==========================================================
# CZĘŚĆ 1: Ograniczenie na normę l1 (Równanie 3 z instrukcji)
# minimize ||y_tilde - v||_2^2 subject to ||Dv||_1 <= q
# ==========================================================
q_val = 5.0
# v_constr to wektor zmiennych optymalizacyjnych o długości n (szukana estymata v)
v_constr = cp.Variable(n)

# Funkcja celu: minimalizacja sumy kwadratów błędów dopasowania (norma l2 do kwadratu)
objective_constr = cp.Minimize(cp.sum_squares(y_tilde - v_constr))

# Ograniczenia: cp.diff(v_constr) działa jak mnożenie wektora v przez macierz D z instrukcji
# (czyli liczy różnice v[i+1] - v[i]). Narzucamy ograniczenie, by suma modułów tych różnic (norma l1) była <= q
constraints = [cp.norm(cp.diff(v_constr), 1) <= q_val]

# Definicja problemu i jego rozwiązanie
prob_constr = cp.Problem(objective_constr, constraints)
prob_constr.solve()
v_est_constr = v_constr.value  # Zapisanie wyliczonej estymaty do zmiennej

# ==========================================================
# CZĘŚĆ 2: Problem LASSO (Równanie 4 z instrukcji)
# minimize ||y_tilde - v||_2^2 + tau * ||Dv||_1
# ==========================================================
tau_val = 2.0
v_lasso = cp.Variable(n)

# W LASSO nie mamy warunku ograniczającego "subject to", ale nakładamy karę na funkcję celu.
# Drugi człon to współczynnik tau mnożony przez normę l1 z różnic (norma l1 z Dv)
objective_lasso = cp.Minimize(
    cp.sum_squares(y_tilde - v_lasso) + tau_val * cp.norm(cp.diff(v_lasso), 1)
)

prob_lasso = cp.Problem(objective_lasso)
prob_lasso.solve()
v_est_lasso = v_lasso.value  # Zapisanie estymaty LASSO

# ==========================================================
# RYSOWANIE WYKRESÓW
# ==========================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(t, y_tilde, color="gray", s=5, alpha=0.5, label=r"$\tilde{y}$ (pomiar)")
plt.plot(
    t,
    v_est_constr,
    color="red",
    linewidth=1.5,
    label=rf"$\hat{{y}}$ (estymata, q={q_val})",
)
plt.title("Ograniczenie $l_1$")
plt.xlabel("t")
plt.ylabel(r"$y, \tilde{y}, \hat{y}$")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(t, y_tilde, color="gray", s=5, alpha=0.5, label=r"$\tilde{y}$ (pomiar)")
plt.plot(
    t,
    v_est_lasso,
    color="blue",
    linewidth=1.5,
    label=rf"$\hat{{y}}$ (estymata, $\tau$={tau_val})",
)
plt.title("LASSO")
plt.xlabel("t")
plt.ylabel(r"$y, \tilde{y}, \hat{y}$")
plt.legend()

plt.tight_layout()
plt.savefig("results.png") # Saves to your current folder
