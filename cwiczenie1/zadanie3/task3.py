import cvxpy as cp
from scipy.optimize import linprog

print("=== Rozwiązanie za pomocą CVXPY ===")
x_SurI = cp.Variable(nonneg=True)
x_SurII = cp.Variable(nonneg=True)
x_LekI = cp.Variable(nonneg=True)
x_LekII = cp.Variable(nonneg=True)

f_costs = 100.00 * x_SurI + 199.90 * x_SurII + 700.00 * x_LekI + 800.00 * x_LekII
f_income = 6500.00 * x_LekI + 7100.00 * x_LekII

objective = cp.Minimize(f_costs - f_income)

constraints = [
    0.01 * x_SurI + 0.02 * x_SurII - 0.50 * x_LekI - 0.60 * x_LekII >= 0,
    x_SurI + x_SurII <= 1000,
    90.00 * x_LekI + 100.00 * x_LekII <= 2000,
    40.00 * x_LekI + 50.00 * x_LekII <= 800,
    f_costs <= 100000,
]

prob3 = cp.Problem(objective, constraints)
prob3.solve()

print(f"Lek I:       {x_LekI.value:.3f}")
print(f"Lek II:      {x_LekII.value:.3f}")
print(f"Surowiec I:  {x_SurI.value:.3f}")
print(f"Surowiec II: {x_SurII.value:.3f}\n")

print("=== Rozwiązanie za pomocą scipy.optimize.linprog ===")
# Kolejność zmiennych: x_SurI, x_SurII, x_LekI, x_LekII
# Funkcja celu to (100*x_SurI + 199.9*x_SurII + 700*x_LekI + 800*x_LekII) - (6500*x_LekI + 7100*x_LekII)
c = [100.00, 199.90, 700.00 - 6500.00, 800.00 - 7100.00]

A_ub = [
    [-0.01, -0.02, 0.50, 0.60],  # bilans czynnika (>= 0, zmieniony znak na <=)
    [1.00, 1.00, 0.0, 0.0],  # magazyn
    [0.0, 0.0, 90.00, 100.00],  # zasoby ludzkie
    [0.0, 0.0, 40.00, 50.00],  # zasoby sprzętowe
    [100.00, 199.90, 700.00, 800.00],  # budżet
]
b_ub = [0, 1000, 2000, 800, 100000]
bounds = [(0, None)] * 4

for method in ["highs-ds", "highs-ipm"]:
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    print(f"Metoda: {method}")
    if res.success:
        print(f"Lek I:       {res.x[2]:.3f}")
        print(f"Lek II:      {res.x[3]:.3f}")
        print(f"Surowiec I:  {res.x[0]:.3f}")
        print(f"Surowiec II: {res.x[1]:.3f}\n")
    else:
        print("Nie znaleziono rozwiązania.\n")
