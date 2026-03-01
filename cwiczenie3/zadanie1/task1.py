import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("Data01.mat")

y_tilde = data["y"].flatten()
t = data["t"].flatten()
n = len(y_tilde)

q_val = 5.0
v_constr = cp.Variable(n)

objective_constr = cp.Minimize(cp.sum_squares(y_tilde - v_constr))
constraints = [cp.norm(cp.diff(v_constr), 1) <= q_val]

prob_constr = cp.Problem(objective_constr, constraints)

prob_constr.solve()
v_est_constr = v_constr.value

tau_val = 2.0
v_lasso = cp.Variable(n)

objective_lasso = cp.Minimize(
    cp.sum_squares(y_tilde - v_lasso) + tau_val * cp.norm(cp.diff(v_lasso), 1)
)

prob_lasso = cp.Problem(objective_lasso)
prob_lasso.solve()
v_est_lasso = v_lasso.value

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
plt.show()
