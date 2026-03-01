import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.optimize import linprog

data = loadmat("Data01.mat")
y_tilde = data["y"].flatten()
t = data["t"].flatten()

n = len(y_tilde)
m = n - 1

D = sp.diags([-1, 1], [0, 1], shape=(m, n))

I_n = sp.eye(n)
I_m = sp.eye(m)
Z_nm = sp.csr_matrix((n, m))
Z_mn = sp.csr_matrix((m, n))

q_val = 5.0
c7 = np.concatenate([np.zeros(n), np.ones(n), np.zeros(m)])

row1 = sp.hstack([I_n, -I_n, Z_nm])
row2 = sp.hstack([-I_n, -I_n, Z_nm])
row3 = sp.csr_matrix(np.concatenate([np.zeros(2 * n), np.ones(m)]))
row4 = sp.hstack([D, Z_mn, -I_m])
row5 = sp.hstack([-D, Z_mn, -I_m])
A7 = sp.vstack([row1, row2, row3, row4, row5])

b7 = np.concatenate([y_tilde, -y_tilde, [q_val], np.zeros(m), np.zeros(m)])

bounds = [(None, None)] * n + [(0, None)] * n + [(0, None)] * m

res7 = linprog(c7, A_ub=A7, b_ub=b7, bounds=bounds, method="highs")
v_est_7 = res7.x[:n]

tau_val = 2.0
c8 = np.concatenate([np.zeros(n), np.ones(n), tau_val * np.ones(m)])

A8 = sp.vstack([row1, row2, row4, row5])
b8 = np.concatenate([y_tilde, -y_tilde, np.zeros(m), np.zeros(m)])

res8 = linprog(c8, A_ub=A8, b_ub=b8, bounds=bounds, method="highs")
v_est_8 = res8.x[:n]

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
plt.show()
