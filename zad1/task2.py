import cvxpy as cp

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
    
    x1 >= 0, x1 <= 10,
    x2 >= 0, x2 <= 10,
    x3 >= 0, x3 <= 10
]

prob2 = cp.Problem(objective, constraints)
prob2.solve()

print(f"{x1.value:.4f}")
print(f"{x2.value:.4f}")
print(f"{x3.value:.4f}")
