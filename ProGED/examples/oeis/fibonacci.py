import numpy as np
import sympy as sp
# pg = ProGED
import ProGED as pg
# from pg.generate import generate_models
# gft = pg.generators.grammar_from_template
# from ProGED import generators.generate_models
from ProGED.equation_discoverer import EqDisco
# from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.generate import generate_models
# from ProGED.model import Model
# from ProGED.model_box import ModelBox
# from ProGED.parameter_estimation import fit_models

oeis = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
 1597,2584,4181,6765,10946,17711,28657,46368,75025,
 121393,196418,317811,514229,832040,1346269,
 2178309,3524578,5702887,9227465,14930352,24157817,
 39088169,63245986,102334155]
fibs = np.array(oeis).reshape(-1, 1)
ts = np.array([i for i in range(40+1)]).reshape(-1, 1)
# print(ts, type(fibs), fibs.shape, type(ts[0]), ts.shape)
data = np.hstack((ts, fibs))

# grammar:
# from pg.generators import grammar_constructionk
# from pg import model

# variables = ["'n'"]
# symbols = {"x":variables, "start":"S", "const":"C"}
# # p_vars = [1, 0.3, 0.4]
# p_T = [0.1, 0.9]
# p_R = [0.1, 0.9]
# grammar = grammar_from_template("polynomial", {"variables": variables, "p_R": p_R, "p_T": p_T})
# np.random.seed(0)
# # print(grammar.generate_one())
# # models = generate_models(grammar, symbols, strategy_settings = {"N":500})
# # print(models)

np.random.seed(0)
ED = EqDisco(data = data,
            task = None,
            target_variable_index = -1,
            variable_names=["n", "an"],
            sample_size = 10,
            verbosity = 0,
            generator = "grammar", 
            generator_template_name = "polynomial",
            # generator_settings={"variables":["'n'"]},
            estimation_settings={"verbosity": 0, "task_type": "algebraic", "lower_upper_bounds": (0,1)}# , "timeout": np.inf}
            )
# print(data, data.shape)
ED.generate_models()
ED.fit_models()
print(ED.models)
# print(ED.get_results())
# print(ED.get_stats())
print("\n", ED.models, "\n\nFinal score:")
for m in ED.models:
    print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    
phi = (1+5**(1/2))/2
psi = (1-5**(1/2))/2
c0 = 1/5**(1/2)
c1 = np.log(phi)
print(f"m  c0: {c0}", f"c1:{c1}")
# fib(n) = (phi**n - psi**n)/5**(1/2)
#         = round(phi**n/5**(1/2))
#         = floor(phi**n/5**(1/2) + 1/2)

# model = ED.models[5] 
model = ED.models[5] 
# print(model, model.params, model.sym_vars, model.full_expr(model.params))
# print(model, model.params, model.full_expr(model.params))
# print(model, model.params, model.full_expr(model.params))
# print(model, model.params, model.full_expr(model.params))
# print(model, model.params, model.full_expr(model.params))
# print(model, model.params, model.full_expr(model.params))
# print(model, model.params, model.full_expr(model.params))
# print(list(zip(model.sym_params, model.params)))
# print(model.expr)
# t = model.full_expr(*model.params)
# print(t)
# s = model.expr.subs(list(zip(model.sym_params, model.params)))
# print(s)
# # print(str(model.full_expr(model.params)))
res = model.evaluate(ts, *model.params)
res = [int(np.round(flo)) for flo in res]

print(res)
print(oeis)
error = 0
for i, j in zip(res, oeis):
    print(i,j, i-j, error)
    error += abs(i-j)

print(error)

# lamb_expr = sp.lambdify(model.sym_vars, model.full_expr(*model.params), "numpy")
# print(lamb_expr)
# res = lamb_expr(*ts.T)
# print(res)