import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def h(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t + x[3])


def f_err(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return h(x, t) - y


def jacobian(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    A, a, w, phi = x
    J = np.zeros((len(t), 4))
    exp_term = np.exp(-a * t)
    sin_term = np.sin(w * t + phi)
    cos_term = np.cos(w * t + phi)

    J[:, 0] = exp_term * sin_term
    J[:, 1] = -A * t * exp_term * sin_term
    J[:, 2] = A * t * exp_term * cos_term
    J[:, 3] = A * exp_term * cos_term
    return J


data = scipy.io.loadmat("LM04Data.mat")
t = data["t"].flatten()
y = data["y"].flatten()

x0 = np.array([1.0, 1.0, 20.0, 0.0])
x = x0.copy()
n = len(x)
k_max = 25
l_val = 1.0

X = np.zeros((n, k_max + 1))
X[:, 0] = x

L_vals = [l_val]
f_norms = []

for k in range(k_max):
    f_k = f_err(x, t, y)
    J_k = jacobian(x, t)
    f_norms.append(np.linalg.norm(f_k) ** 2)

    try:
        delta = np.linalg.solve(J_k.T @ J_k + l_val * np.eye(n), -J_k.T @ f_k)
    except np.linalg.LinAlgError:
        delta = np.zeros(n)

    x_new = x + delta

    if np.linalg.norm(f_err(x_new, t, y)) ** 2 < np.linalg.norm(f_err(x, t, y)) ** 2:
        l_val *= 0.8
        x = x_new
    else:
        l_val *= 2.0

    L_vals.append(l_val)
    X[:, k + 1] = x

f_norms.append(np.linalg.norm(f_err(x, t, y)) ** 2)

print("Parametry końcowe:")
print(f"  A   = {x[0]:.4f}")
print(f"  a   = {x[1]:.4f}")
print(f"  w   = {x[2]:.4f}")
print(f"  phi = {x[3]:.4f}")
print(f"  ||f||^2 końcowe = {f_norms[-1]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(t, y, "rs", label="measurement")
t_plot = np.linspace(t[0], t[-1], 1000)
plt.plot(t_plot, h(x0, t_plot), "b-", label="first guess")
plt.plot(t_plot, h(x, t_plot), "k-", label="final fit", linewidth=2)
plt.legend()
plt.title("Zadanie 2 - Dopasowanie tlumionej sinusoidy")
plt.xlabel("t [s]")
plt.ylabel("y [a.u.]")
plt.grid()
plt.tight_layout()
plt.savefig("task2_fig5.png")

plt.figure(figsize=(8, 4))
plt.plot(L_vals, "ks")
plt.ylabel("lambda")
plt.xlabel("k")
plt.grid()
plt.tight_layout()
plt.savefig("task2_fig6.png")

plt.figure(figsize=(8, 4))
plt.semilogy(f_norms, "ks")
plt.ylabel("||f(x)||^2")
plt.xlabel("k")
plt.grid()
plt.tight_layout()
plt.savefig("task2_fig7.png")
