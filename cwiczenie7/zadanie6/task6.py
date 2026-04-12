import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.optimize import least_squares

print("--- ZADANIE 1 ---")
data1 = sio.loadmat("LM01Data.mat")
t1, y1 = data1["t"].flatten(), data1["y"].flatten()


def fun1(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[0] * np.sin(x[1] * t + x[2]) - y


def jac1(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    J = np.zeros((len(t), 3))
    J[:, 0] = np.sin(x[1] * t + x[2])
    J[:, 1] = x[0] * t * np.cos(x[1] * t + x[2])
    J[:, 2] = x[0] * np.cos(x[1] * t + x[2])
    return J


x0_1 = [1.0, 100 * np.pi, 0.0]
res1 = least_squares(fun1, x0_1, jac=jac1, method="lm", args=(t1, y1))
print(f"Wynik Zadanie 1: A={res1.x[0]:.4f}, w={res1.x[1]:.4f}, phi={res1.x[2]:.4f}")


print("\n--- ZADANIE 2 ---")
data2 = sio.loadmat("LM04Data.mat")
t2, y2 = data2["t"].flatten(), data2["y"].flatten()


def fun2(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t + x[3]) - y


def jac2(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
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


x0_2 = [1.0, 1.0, 20.0, 0.0]
res2 = least_squares(fun2, x0_2, jac=jac2, method="lm", args=(t2, y2))
print(
    f"Wynik Zadanie 2: A={res2.x[0]:.4f}, a={res2.x[1]:.4f}, w={res2.x[2]:.4f}, phi={res2.x[3]:.4f}"
)


# ==========================================
# ZADANIE 3
# ==========================================
print("\n--- ZADANIE 3 ---")
data3 = sio.loadmat("inertialData.mat")
t3, y3 = data3["t"].flatten(), data3["y"].flatten()


def fun3(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[0] * (1 - np.exp(-t / x[1])) - y


def jac3(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    k, T = x
    J = np.zeros((len(t), 2))
    J[:, 0] = 1 - np.exp(-t / T)
    J[:, 1] = -k * (t / T**2) * np.exp(-t / T)
    return J


x0_3 = [1.0, 1.0]
res3 = least_squares(fun3, x0_3, jac=jac3, method="lm", args=(t3, y3))
print(f"Wynik Zadanie 3: k={res3.x[0]:.4f}, T={res3.x[1]:.4f}")


print("\n--- ZADANIE 4 ---")
data4 = sio.loadmat("twoInertialData.mat")
t4, y4 = data4["t"].flatten(), data4["y"].flatten()


def fun4(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    k, T1, T2 = x
    if np.isclose(T1, T2):
        T1 += 1e-5
    return k * (1 - 1 / (T1 - T2) * (T1 * np.exp(-t / T1) - T2 * np.exp(-t / T2))) - y


def jac4(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    k, T1, T2 = x
    if np.isclose(T1, T2):
        T1 += 1e-5
    J = np.zeros((len(t), 3))
    J[:, 0] = 1 - 1 / (T1 - T2) * (T1 * np.exp(-t / T1) - T2 * np.exp(-t / T2))
    J[:, 1] = (k / (T2 - T1)) * (
        (t / T1) * np.exp(-t / T1)
        + (T2 / (T1 - T2)) * (np.exp(-t / T2) - np.exp(-t / T1))
    )
    J[:, 2] = (k / (T1 - T2)) * (
        (t / T2) * np.exp(-t / T2)
        + (T1 / (T2 - T1)) * (np.exp(-t / T1) - np.exp(-t / T2))
    )
    return J


x0_4 = [1.0, 1.0, 2.0]
res4 = least_squares(fun4, x0_4, jac=jac4, method="lm", args=(t4, y4))
print(f"Wynik Zadanie 4: k={res4.x[0]:.4f}, T1={res4.x[1]:.4f}, T2={res4.x[2]:.4f}")


print("\n--- ZADANIE 5 ---")
L, C = 197e-6, 100e-9
Vin, Vout = 14.0, 1.95
omega_s = 40 * 2 * np.pi * 1e3

omega_0_sq = 1 / (L * C)
M = (2 * np.sqrt(Vin**2 - Vout**2)) / (np.pi * np.abs(1 - omega_s**2 * L * C))
K = (2 * omega_0_sq * Vout) / (np.pi * M * omega_s)
Sigma = omega_s**2 + omega_0_sq
Delta = omega_s**2 - omega_0_sq

num = [-M * omega_s * K, -M * omega_s * 2 * Delta]
den = [1, K, 2 * Sigma, K * Sigma, Delta**2]
system_4th = signal.TransferFunction(num, den)

t5 = np.linspace(0, 0.0035, 1000)
_, y5 = signal.step(system_4th, T=t5)

# Przeskalowanie zgodnie ze wskazówkami w zadaniu 5
t5_scaled = t5 * 1000
y5_scaled = y5 * 1000


def fun5(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    k, gamma, beta = x
    return (
        k
        * (
            1
            - np.exp(-gamma * t)
            * (np.cos(beta * t) + (gamma / beta) * np.sin(beta * t))
        )
        - y
    )


def jac5(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    k, gamma, beta = x
    J = np.zeros((len(t), 3))
    exp_term = np.exp(-gamma * t)
    sin_term = np.sin(beta * t)
    cos_term = np.cos(beta * t)

    J[:, 0] = 1 - exp_term * (cos_term + (gamma / beta) * sin_term)
    J[:, 1] = k * exp_term * (t * cos_term - ((1 - t * gamma) / beta) * sin_term)
    J[:, 2] = (
        k
        * exp_term
        * ((t + gamma / (beta**2)) * sin_term - (gamma / beta) * t * cos_term)
    )
    return J


x0_5 = [1.0, 1.0, 1.0]
res5 = least_squares(fun5, x0_5, jac=jac5, method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12, args=(t5_scaled, y5_scaled))

k_opt = res5.x[0] * 1e-3
gamma_opt = res5.x[1] * 1e-3
beta_opt = res5.x[2] * 1e-3
T_opt = 1.0 / np.sqrt(beta_opt**2 + gamma_opt**2)
xi_opt = gamma_opt * T_opt

print("Wynik Zadanie 5:")
print(f"  k     = {k_opt:.6e}")
print(f"  gamma = {gamma_opt:.6e}")
print(f"  beta  = {beta_opt:.6e}")
print(f"  T     = {T_opt:.6e}")
print(f"  xi    = {xi_opt:.6e}")
