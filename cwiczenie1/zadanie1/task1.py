import cvxpy as cp
from scipy.optimize import linprog

print("=== Rozwiązanie za pomocą CVXPY ===")
x1 = cp.Variable(nonneg=True)
x2 = cp.Variable(nonneg=True)
x3 = cp.Variable(nonneg=True)

objective = cp.Minimize(300 * x1 + 500 * x2 + 800 * x3)

constraints = [
    0.8 * x1 + 0.3 * x2 + 0.1 * x3 >= 0.3,
    0.01 * x1 + 0.4 * x2 + 0.7 * x3 >= 0.7,
    0.15 * x1 + 0.1 * x2 + 0.2 * x3 >= 0.1,
]

prob1 = cp.Problem(objective, constraints)
prob1.solve()

print(f"x1: {x1.value:.4f}")
print(f"x2: {x2.value:.4f}")
print(f"x3: {x3.value:.4f}\n")

print("=== Rozwiązanie za pomocą scipy.optimize.linprog ===")
# Funkcja celu (minimalizacja)
c = [300, 500, 800]

# Ograniczenia nierównościowe typu <= (dlatego mnożymy przez -1)
A_ub = [[-0.8, -0.3, -0.1], [-0.01, -0.4, -0.7], [-0.15, -0.1, -0.2]]
b_ub = [-0.3, -0.7, -0.1]

# Ograniczenia zmiennych (x >= 0)
bounds = [(0, None), (0, None), (0, None)]

for method in ["highs-ds", "highs-ipm"]:
    # highs-ds to odpowiednik dual-simplex, highs-ipm to interior-point
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    print(f"Metoda: {method}")
    if res.success:
        print(f"x1: {res.x[0]:.4f}")
        print(f"x2: {res.x[1]:.4f}")
        print(f"x3: {res.x[2]:.4f}\n")
    else:
        print("Nie znaleziono rozwiązania.\n")
