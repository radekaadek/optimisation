import cvxpy as cp
import numpy as np

# ==============================================================================
# 1. Dane i inicjalizacja parametrów (Zgodnie z PDF zadania)
# ==============================================================================
# y - macierz współrzędnych 5 sensorów (każdy wiersz to jeden sensor)
y = np.array([[1.8, 2.5], [2.0, 1.7], [1.5, 1.5], [1.5, 2.0], [2.5, 1.5]])
# d - wektor zmierzonych odległości (zaszumionych) od źródła do sensorów
d = np.array([2.00, 1.24, 0.59, 1.31, 1.44])
m = len(d)

# Zgodnie ze wzorami z instrukcji tworzymy macierz A i wektor b.
# Wynikają one z rozwinięcia wyrażenia ||x - y_k||^2 i podstawienia zmiennej t = x^T*x
A = np.zeros((m, 3))
A[:, :2] = -2 * y  # Pierwsze dwie kolumny to -2 * y_k^T
A[:, 2] = 1  # Trzecia kolumna to 1 (odpowiada za dodane 't')

b = np.zeros(m)
for k in range(m):
    # Przekształcenie stałych elementów do wektora b
    b[k] = d[k] ** 2 - np.linalg.norm(y[k]) ** 2

# Macierz Q wyciągająca t z wektora z oraz wektor c z funkcji celu
Q = np.diag([1, 1, 0])
c = np.array([0, 0, -0.5])

# ==============================================================================
# 2. Rozwiązanie zadania dualnego SDP z użyciem pakietu CVXPY
# ==============================================================================
mu = cp.Variable()  # Zmienna dualna mu (skalar)
t = (
    cp.Variable()
)  # Pomocnicza zmienna t używana do zdefiniowania problemu w postaci epigrafowej

# Budowanie bloków macierzy, która musi być dodatnio półokreślona (LMI)
M_tl = A.T @ A + mu * Q  # Lewy górny blok: A^T A + mu*Q
v_tr = A.T @ b - mu * c  # Prawy górny / lewy dolny blok: A^T b - mu*c

# Konstrukcja macierzy blokowej LMI (Linear Matrix Inequality) zgodnie ze wzorem (6) z PDF
LMI = cp.bmat(
    [
        [M_tl, cp.reshape(v_tr, (3, 1), order="F")],
        [cp.reshape(v_tr, (1, 3), order="F"), cp.reshape(t, (1, 1), order="F")],
    ]
)

# Ograniczenia: LMI musi być dodatnio półokreślone (>> 0 to w CVXPY zapis semi-definite)
# Dodatkowo narzucamy warunek A^T A + mu*Q >= 0 (wynikający z ograniczoności Lagranżjanu z dołu)
constraints = [LMI >> 0, M_tl >> 0]

# Minimalizujemy t - ||b||^2. Prowadzi to do maksymalizacji funkcji dualnej
prob = cp.Problem(cp.Minimize(t - np.linalg.norm(b) ** 2), constraints)
prob.solve()

# Pobranie znalezionej optymalnej wartości zmiennej dualnej
mu_opt = mu.value

# ==============================================================================
# 3. Wyznaczenie z* i estymaty położenia (x*)
# ==============================================================================
# Skoro znamy już mu*, obliczamy z* rozwiązując układ równań (warunek KKT):
# (A^T A + mu*Q)z = A^T b - mu*c
z_opt = np.linalg.solve(A.T @ A + mu_opt * Q, A.T @ b - mu_opt * c)

# Wektor z_opt zawiera [x_1, x_2, t]. Nas interesują tylko współrzędne położenia źródła.
x_star = z_opt[:2]

print(f"Optymalne mu: {mu_opt:.4f}")
print(f"Rozwiązanie - położenie źródła (x*): [{x_star[0]:.2f}, {x_star[1]:.2f}]")

# ==============================================================================
# 4. Sprawdzenie warunku dla z* # ==============================================================================
# Weryfikujemy, czy znaleziona wartość spełnia warunek optymalności.
# Jeśli norma jest bliska zera (rzędu 1e-15), warunek z PDF jest spełniony.
warunek = np.linalg.norm((A.T @ A + mu_opt * Q) @ z_opt - (A.T @ b - mu_opt * c))
print(f"Wartość warunku (oczekiwane bliskie 0): {warunek:.2e}")

# ==============================================================================
# 5. Wykres poziomic funkcji f0 oraz lokalizacji
# ==============================================================================
# Tworzenie siatki punktów dla x1 i x2 do narysowania poziomic f0(x)
x1_vals = np.linspace(0, 3, 200)
x2_vals = np.linspace(0, 3, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x_vec = np.array([X1[i, j], X2[i, j]])
