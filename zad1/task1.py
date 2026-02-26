import cvxpy as cp

x1 = cp.Variable(nonneg=True)
x2 = cp.Variable(nonneg=True)
x3 = cp.Variable(nonneg=True)

objective = cp.Minimize(300 * x1 + 500 * x2 + 800 * x3)

constraints = [
    0.8 * x1 + 0.3 * x2 + 0.1 * x3 >= 0.3,
    0.01 * x1 + 0.4 * x2 + 0.7 * x3 >= 0.7,
    0.15 * x1 + 0.1 * x2 + 0.2 * x3 >= 0.1
]

prob1 = cp.Problem(objective, constraints)
prob1.solve()

print(f"{x1.value:.4f}")
print(f"{x2.value:.4f}")
print(f"{x3.value:.4f}")
