import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def h(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x[0] * np.sin(x[1] * t + x[2])


def f(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return h(x, t) - y


def J(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.sin(x[1] * t + x[2]),
            x[0] * t * np.cos(x[1] * t + x[2]),
            x[0] * np.cos(x[1] * t + x[2]),
        ]
    )


data = scipy.io.loadmat("LM01Data.mat")
t = data["t"].flatten()
y = data["y"].flatten()

x0 = np.array([1.0, 100 * np.pi, 0.0])
x = x0.copy()
n = len(x)
k_max = 35
l_val = 1.0

X = np.zeros((n, k_max + 1))
X[:, 0] = x
L_vals = []
f_norms = []

for k in range(k_max):
    f_k = f(x, t, y)
    J_k = J(x, t)
    f_norms.append(np.linalg.norm(f_k) ** 2)
    L_vals.append(l_val)

    try:
        delta = np.linalg.solve(J_k.T @ J_k + l_val * np.eye(n), -J_k.T @ f_k)
    except np.linalg.LinAlgError:
        delta = np.zeros(n)

    x_new = x + delta
    if np.linalg.norm(f(x_new, t, y)) ** 2 < np.linalg.norm(f_k) ** 2:
        l_val *= 0.8
        x = x_new
    else:
        l_val *= 2.0
    X[:, k + 1] = x

f_norms.append(np.linalg.norm(f(x, t, y)) ** 2)

print("Parametry końcowe:")
print(f"  A   = {x[0]:.4f}")
print(f"  w   = {x[1]:.4f}")
print(f"  phi = {x[2]:.4f}")
print(f"  ||f||^2 końcowe = {f_norms[-1]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(t, y, "rs", label="measurement")
plt.plot(t, h(X[:, 0], t), "b-", label="first guess")
plt.plot(t, h(x, t), "k-", label="final fit", linewidth=2)
plt.legend()
plt.title("Zadanie 1 - Dopasowanie sinusoidy")
plt.xlabel("t [s]")
plt.ylabel("y [a.u.]")
plt.grid()
plt.tight_layout()
plt.savefig("task1_fig1.png")

plt.figure(figsize=(8, 8))
plt.subplot(311)
plt.plot(X[0, :], "k-", linewidth=2)
plt.ylabel("A [a.u.]")
plt.grid()
plt.subplot(312)
plt.plot(X[1, :], "k-", linewidth=2)
plt.ylabel("omega [rad/s]")
plt.grid()
plt.subplot(313)
plt.plot(X[2, :], "k-", linewidth=2)
plt.ylabel("phi [rad]")
plt.grid()
plt.xlabel("iteration number")
plt.tight_layout()
plt.savefig("task1_fig2.png")

plt.figure(figsize=(8, 4))
plt.plot(L_vals, "ks")
plt.ylabel("lambda")
plt.xlabel("k")
plt.grid()
plt.tight_layout()
plt.savefig("task1_fig3.png")

plt.figure(figsize=(8, 4))
plt.plot(f_norms, "ks")
plt.ylabel("||f(x)||^2")
plt.xlabel("k")
plt.grid()
plt.tight_layout()
plt.savefig("task1_fig4.png")
