import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.optimize import linprog

data = loadmat("Data01.mat")
y_tilde = data["y"].flatten()
t = data["t"].flatten()

n = len(y_tilde)
m = n - 1  # Liczba różnic między próbkami

# Macierz D operująca na różnicach sygnału (v_i - v_{i-1}). D ma wymiary (m, n)
D = sp.diags([-1, 1], [0, 1], shape=(m, n), dtype=None)

# Zmienne pomocnicze: macierze jednostkowe (I) i zerowe (Z) do budowy blokowej macierzy A_ub
I_n = sp.eye(n)
I_m = sp.eye(m)
Z_nm = sp.csr_matrix((n, m))
Z_mn = sp.csr_matrix((m, n))

# ==========================================================
# CZĘŚĆ 1: Równanie 7 z instrukcji przerobione na programowanie liniowe
# Oczekiwany format do linprog: minimize c^T*x subject to A*x <= b
# Wektor x zawiera teraz trzy zestawy zmiennych połączonych w jeden długi wektor:
# x = [v_1...v_n,  xi_1...xi_n,  delta_1...delta_m]^T
# ==========================================================
q_val = 5.0

# Wektor c funkcji celu. Cel to minimize suma xi_i, a v i delta mają mnożnik 0 (nie ma ich w funkcji celu z Równania 17a)
# Zapis z instrukcji: c = [0_n, 1_n, 0_m]
c7 = np.concatenate([np.zeros(n), np.ones(n), np.zeros(m)])

# Macierz A zbudowana blokowo dla układu nierówności (równania 20 i 21 z instrukcji):
row1 = sp.hstack([I_n, -I_n, Z_nm])  # y_tilde - v <= xi  -->  v - xi <= y_tilde
row2 = sp.hstack([-I_n, -I_n, Z_nm])  # -(y_tilde - v) <= xi --> -v - xi <= -y_tilde
row3 = sp.csr_matrix(np.concatenate([np.zeros(2 * n), np.ones(m)]))  # suma(delta) <= q
row4 = sp.hstack([D, Z_mn, -I_m])  # Dv <= delta --> Dv - delta <= 0
row5 = sp.hstack([-D, Z_mn, -I_m])  # -Dv <= delta --> -Dv - delta <= 0

# Złożenie wszystkich wierszy w jedną dużą macierz ograniczeń
A7 = sp.vstack([row1, row2, row3, row4, row5])

# Wektor ograniczeń b (prawa strona nierówności macierzy A)
b7 = np.concatenate([y_tilde, -y_tilde, [q_val], np.zeros(m), np.zeros(m)])

# Granice dla poszczególnych zmiennych:
# v: może przyjmować dowolne wartości (-inf do inf)
# xi: musi być >= 0 (bo jest ograniczeniem dla modułu)
# delta: musi być >= 0 (ograniczenie dla modułu różnic)
bounds = [(None, None)] * n + [(0, None)] * n + [(0, None)] * m

# Uruchomienie solvera programowania liniowego dla problemu z ograniczeniem na q
res7 = linprog(c7, A_ub=A7, b_ub=b7, bounds=bounds, method="highs")
v_est_7 = res7.x[
    :n
]  # Wyciągamy z wektora x tylko pierwsze n zmiennych (samo v, ignorujemy xi i delty)

# ==========================================================
# CZĘŚĆ 2: Problem 8 (LASSO z normą l1) przerobiony na programowanie liniowe
# minimize suma(xi) + tau * suma(delta)
# ==========================================================
tau_val = 2.0

# Wektor c funkcji celu (Równanie 28a z instrukcji).
# Teraz delta też wchodzi do funkcji celu pomnożona przez tau.
c8 = np.concatenate([np.zeros(n), np.ones(n), tau_val * np.ones(m)])

# Macierz ograniczeń dla problemu LASSO - jest taka sama, bez ograniczenia sum(delta) <= q (usuwamy row3)
A8 = sp.vstack([row1, row2, row4, row5])
# Analogicznie aktualizujemy wektor ograniczeń b
b8 = np.concatenate([y_tilde, -y_tilde, np.zeros(m), np.zeros(m)])

# Rozwiązanie LASSO metodą Highs
res8 = linprog(c8, A_ub=A8, b_ub=b8, bounds=bounds, method="highs")
v_est_8 = res8.x[:n]

# ==========================================================
# RYSOWANIE WYKRESÓW
# ==========================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(t, y_tilde, color="gray", s=5, alpha=0.5, label=r"$\tilde{y}$ (pomiar)")
plt.plot(
    t,
    v_est_7,
    color="red",
    linewidth=1.5,
    label=rf"$\hat{{y}}$ (estymata, $q={q_val}$)",
)
plt.title(r"Minimalizacja $\|\| \tilde{y} - v \|\|_1$ (z ograniczeniem $q$)")
plt.xlabel("t")
plt.ylabel(r"$y, \tilde{y}, \hat{y}$")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(t, y_tilde, color="gray", s=5, alpha=0.5, label=r"$\tilde{y}$ (pomiar)")
plt.plot(
    t,
    v_est_8,
    color="blue",
    linewidth=1.5,
    label=rf"$\hat{{y}}$ (estymata, $\tau={tau_val}$)",
)
plt.title(r"LASSO z normą $l_1$")
plt.xlabel("t")
plt.ylabel(r"$y, \tilde{y}, \hat{y}$")
plt.legend()

plt.tight_layout()
plt.savefig("results.png")
