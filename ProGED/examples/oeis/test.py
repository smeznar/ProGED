moi, dsa = 1, True
print("--- --- >> inside check_it(), what is model.sym_vars and"
    + " model.expr", moi, dsa)


import sympy as sp

x, y, z = sp.symbols("x y z")

expr = x + y*z
print(expr)
a = expr.subs([(x, 1), (z, 435)])
print(a, expr)

print(f"double checking with {dsa} equations"
    f"really helped in finding False Positives!!!!!")

