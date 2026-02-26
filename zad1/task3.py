import cvxpy as cp

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
    f_costs <= 100000
]

prob3 = cp.Problem(objective, constraints)
prob3.solve()

print(f"{x_LekI.value:.3f}")
print(f"{x_LekII.value:.3f}")
print(f"{x_SurI.value:.3f}")
print(f"{x_SurII.value:.3f}")
