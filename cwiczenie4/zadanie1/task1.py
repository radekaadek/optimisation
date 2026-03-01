import cvxpy as cp
import numpy as np

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


mu = cp.Variable()
t = cp.Variable()

M_tl = A.T @ A + mu * Q
v_tr = A.T @ b - mu * c

LMI = cp.bmat(
    [
        [M_tl, cp.reshape(v_tr, (3, 1), order="F")],
        [cp.reshape(v_tr, (1, 3), order="F"), cp.reshape(t, (1, 1), order="F")],
    ]
)

constraints = [LMI >> 0]
constraints += [M_tl >> 0]

prob = cp.Problem(cp.Minimize(t - np.linalg.norm(b) ** 2), constraints)
prob.solve()

mu_opt = mu.value
z_opt = np.linalg.solve(A.T @ A + mu_opt * Q, A.T @ b - mu_opt * c)

x_star = z_opt[:2]

print(f"{mu_opt:.4f}")
print(f"[{x_star[0]:.2f}, {x_star[1]:.2f}]")
