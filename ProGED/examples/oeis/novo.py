import sympy as sp
from ProGED.diophantine_solver import diophantine_solve

X, Y = sp.Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), sp.Matrix([[1], [1]])
print(X, Y)



A, b = sp.Matrix([[0, 1], [1, 2]]), sp.Matrix([[1], [0]])

# x [Matrix([
# [-2],
# [ 1]])]

# A, b = model2diophant(model, X, Y)
x = diophantine_solve(A, b)
print(x)
