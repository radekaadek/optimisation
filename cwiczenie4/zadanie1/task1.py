import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

# 1. Dane i inicjalizacja parametrów
y = np.array([[1.8, 2.5], [2.0, 1.7], [1.5, 1.5], [1.5, 2.0], [2.5, 1.5]])
d = np.array([2.00, 1.24, 0.59, 1.31, 1.44])
m = len(d)

A = np.zeros((m, 3))
A[:, :2] = -2 * y
A[:, 2] = 1

b = np.zeros(m)
for k in range(m):
    b[k] = d[k] ** 2 - np.linalg.norm(y[k]) ** 2

Q = np.diag([1, 1, 0])
c = np.array([0, 0, -0.5])

# 2. Rozwiązanie zadania dualnego SDP z użyciem CVXPY
mu = cp.Variable()
t = cp.Variable()

M_tl = A.T @ A + mu * Q
v_tr = A.T @ b - mu * c

# Warunek LMI (Linear Matrix Inequality)
LMI = cp.bmat(
    [
        [M_tl, cp.reshape(v_tr, (3, 1), order="F")],
        [cp.reshape(v_tr, (1, 3), order="F"), cp.reshape(t, (1, 1), order="F")],
    ]
)

constraints = [LMI >> 0, M_tl >> 0]

prob = cp.Problem(cp.Minimize(t - np.linalg.norm(b) ** 2), constraints)
prob.solve()

mu_opt = mu.value

# 3. Wyznaczenie z* i estymaty położenia (x*)
z_opt = np.linalg.solve(A.T @ A + mu_opt * Q, A.T @ b - mu_opt * c)
x_star = z_opt[:2]

print(f"Optymalne mu: {mu_opt:.4f}")
print(f"Rozwiązanie - położenie źródła (x*): [{x_star[0]:.2f}, {x_star[1]:.2f}]")

# 4. Sprawdzenie warunku dla z* (wynik bliski zera oznacza spełnienie warunku)
warunek = np.linalg.norm((A.T @ A + mu_opt * Q) @ z_opt - (A.T @ b - mu_opt * c))
print(f"Wartość warunku (oczekiwane bliskie 0): {warunek:.2e}")

# 5. Wykres poziomic funkcji f0 oraz lokalizacji
x1_vals = np.linspace(0, 3, 200)
x2_vals = np.linspace(0, 3, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x_vec = np.array([X1[i, j], X2[i, j]])
        # Funkcja celu f0(x)
        Z[i, j] = sum(
            (np.linalg.norm(x_vec - y[k]) ** 2 - d[k] ** 2) ** 2 for k in range(m)
        )

plt.figure(figsize=(8, 6))
# Rysowanie poziomic
plt.contour(X1, X2, Z, levels=np.linspace(0, 50, 40), cmap="viridis", linewidths=1)
# Rysowanie sensorów i źródła
plt.plot(y[:, 0], y[:, 1], "r*", markersize=8, label="Sensory")
plt.plot(
    x_star[0],
    x_star[1],
    "ko",
    markersize=8,
    markerfacecolor="none",
    label="Źródło sygnału",
)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Poziomice funkcji $f_0(x)$ wraz z lokalizacją sensorów i źródła")
plt.legend()
plt.grid(visible=True, linestyle="--", alpha=0.5)
plt.xlim([np.min(x1_vals), np.max(x1_vals)])
plt.ylim([np.min(x2_vals), np.max(x2_vals)])
plt.show()
