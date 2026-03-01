import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

data = np.loadtxt("data01.csv", delimiter=",")
x = data[:, 0]
y = data[:, 1]

N = len(x)

Phi = np.vstack((x, np.ones(N))).T

theta_LS = np.linalg.pinv(Phi) @ y
a_LS, b_LS = theta_LS[0], theta_LS[1]

c = np.concatenate(([0, 0], np.ones(N)))

I = np.eye(N)
A_ub = np.vstack((np.hstack((Phi, -I)), np.hstack((-Phi, -I))))
b_ub = np.concatenate((y, -y))

bounds = [(None, None), (None, None)] + [(0, None)] * N

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
a_LP, b_LP = res.x[0], res.x[1]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="red", marker=".", label="$(x_i, y_i)$")

x_line = np.array([np.min(x), np.max(x)])
plt.plot(
    x_line, a_LS * x_line + b_LS, color="black", linewidth=2, label="y = ax + b (LS)"
)
plt.plot(
    x_line, a_LP * x_line + b_LP, color="blue", linewidth=2, label="y = ax + b (LP)"
)

plt.ylim([np.min(y[(y > -20) & (y < 20)]) - 2, np.max(y[(y > -20) & (y < 20)]) + 2])

plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")
plt.grid(True)
plt.title("Porównanie dopasowania LP i LS w obecności punktów odstających")
plt.show()
