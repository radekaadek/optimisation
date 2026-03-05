import cvxpy as cp
from scipy.optimize import linprog

print("=== Rozwiązanie za pomocą CVXPY ===")
x1 = cp.Variable()
x2 = cp.Variable()
x3 = cp.Variable()

objective = cp.Minimize(0.15 * x1 + 0.25 * x2 + 0.05 * x3)

constraints = [
    70 * x1 + 121 * x2 + 65 * x3 >= 2000,
    70 * x1 + 121 * x2 + 65 * x3 <= 2250,
    107 * x1 + 500 * x2 >= 5000,
    107 * x1 + 500 * x2 <= 10000,
    45 * x1 + 40 * x2 + 60 * x3 <= 1000,
    x1 >= 0,
    x1 <= 10,
    x2 >= 0,
    x2 <= 10,
    x3 >= 0,
    x3 <= 10,
]

prob2 = cp.Problem(objective, constraints)
prob2.solve()

print(f"x1 (płatki): {x1.value:.4f}")
print(f"x2 (mleko):  {x2.value:.4f}")
print(f"x3 (chleb):  {x3.value:.4f}\n")

print("=== Rozwiązanie za pomocą scipy.optimize.linprog ===")
c = [0.15, 0.25, 0.05]

A_ub = [
    [-70, -121, -65],  # >= 2000  (mnożone przez -1)
    [70, 121, 65],  # <= 2250
    [-107, -500, 0],  # >= 5000  (mnożone przez -1)
    [107, 500, 0],  # <= 10000
    [45, 40, 60],  # <= 1000
]
b_ub = [-2000, 2250, -5000, 10000, 1000]

bounds = [(0, 10), (0, 10), (0, 10)]

for method in ["highs-ds", "highs-ipm"]:
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    print(f"Metoda: {method}")
    if res.success:
        print(f"x1: {res.x[0]:.4f}")
        print(f"x2: {res.x[1]:.4f}")
        print(f"x3: {res.x[2]:.4f}\n")
    else:
        print("Nie znaleziono rozwiązania.\n")
