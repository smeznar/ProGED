from ProGED.parameter_estimation import integer_brute_fit
import numpy as np
from ProGED.equation_discoverer import EqDisco
# from model.py import 
from scipy.optimize import brute

oeis = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
 1597,2584,4181,6765,10946,17711,28657,46368,75025,
 121393,196418,317811,514229,832040,1346269,
 2178309,3524578,5702887,9227465,14930352,24157817,
 39088169,63245986,102334155]
# fibs = np.array(oeis)
fibs = np.array(oeis)
# ts = np.array([i for i in range(40+1)]).reshape(-1, 1)
# ts = np.array([i for i in range(40+1)])
# print(ts, type(fibs), fibs.shape, type(ts[0]), ts.shape)
# data = np.hstack((ts, fibs))

# map( [0..m], fibs[i:-m+i])

# map([1,2,])

# we want:
# 0 1 2
# 1 2 3
# 2 3 4
# m = 2
# (n-m, m)
# def f(i,j):
#    return i+j 
# template = np.fromfunction((lambda i,j:i+j), (40+1-2,2+1), dtype=int)
# an = a{n-1} + a{n-2} is recurrece relation of order 2 (2 preceeding terms).
def mdata(order, fibs):
    """order -> (n-order) x (0/1+order) data matrix
    order ... number of previous terms in formula
    """
    n = fibs.shape[0] # (40+1)
    # if ts.shape != fibs.shape:
    #     print("ts and fibs differend dimensions !!!!!")
    #     return 1/0
    # np.hstack(ts[:-m].reshape(-1,1), fibs[:-m])
    indexes = np.fromfunction((lambda i,j:i+j), (n-order, order+1), dtype=int)
    # first_column = indexes[:, [0]]
    # return np.hstack((first_column, fibs[indexes]))
    return fibs[indexes]
# print(fibs)
# print(mdata(2, fibs))
order = 2
data = mdata(order, fibs)


# # variables = ["'n'"]
# # symbols = {"x":variables, "start":"S", "const":"C"}
# # # p_vars = [1, 0.3, 0.4]
p_T = [0.4, 0.6]
p_R = [0.9, 0.1]
# # grammar = grammar_from_template("polynomial", {"variables": variables, "p_R": p_R, "p_T": p_T})
# # np.random.seed(0)
# # # print(grammar.generate_one())
# # # models = generate_models(grammar, symbols, strategy_settings = {"N":500})
# # # print(models)

np.random.seed(0)
# seed 0 , size 20 (16)
# seed3 size 15 an-1 + an-2 + c3
ED = EqDisco(data = data,
            task = None,
            target_variable_index = -1,
            variable_names=["an_2", "an_1", "an"],
            sample_size = 16,
            # sample_size = 1,
            verbosity = 0,
            generator = "grammar", 
            generator_template_name = "polynomial",
            generator_settings={"variables": ["'an_2'", "'an_1'"], "p_T": p_T, "p_R": p_R},
            estimation_settings={"verbosity": 2,
            #  "task_type": "algebraic",
             "task_type": "oeis",
            # "task_type": "oeis_recursive_error",
            #  "lower_upper_bounds": (-1000, 1000), 
            # "lower_upper_bounds": (-2, 2), 
            "lower_upper_bounds": (-5, 5), 
            # "lower_upper_bounds": (-7, 7),  # long enough for brute
            # meja, ko se najde priblizno: (-10,8)}# 
            "optimizer": integer_brute_fit,
            }
            )
# # print(data, data.shape)
ED.generate_models()
print(ED.models)
ED.fit_models()
# # try:
# #     print(12/0)

# # print(ED.get_results())
# # print(ED.get_stats())
print("\nFinal score:")
for m in ED.models:
    print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    
# phi = (1+5**(1/2))/2
# psi = (1-5**(1/2))/2
# c0 = 1/5**(1/2)
# c1 = np.log(phi)
# print(f"m  c0: {c0}", f"c1:{c1}")
# # fib(n) = (phi**n - psi**n)/5**(1/2)
# #         = round(phi**n/5**(1/2))
# #         = floor(phi**n/5**(1/2) + 1/2)

# # model = ED.models[5] 
model = ED.models[-1] 
# print(type(model.params), "ispisi: type(model.params)")
# # model.params = np.round(model.params)
# an = model.lambdify()
# # an = model.lambdify(*np.round(model.params))
# # print(an, an(1, 2), "izpise: an(1,2)")
# cache = list(fibs[:order])
# for _ in range(order, len(fibs)):
#     cache += [an(*(cache[-order:]))]
# res = cache
# # print(len(oeis), len(res), "izpis len oeis in res")
# print(oeis)
# print(res)
# error = 0
# for i, j in zip(res, oeis):
#     print(i,j, i-j, error)
#     error += abs(i-j)
# print(error)

# brute
params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
def f1(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)

def f2(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))

def f3(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))

def f(z, *params):
    return f1(z, *params) + f2(z, *params) + f3(z, *params)

rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))

resbrute = brute(f, rranges, args=params, full_output=True,)
#                           finish=optimize.fmin)
# print(resbrute[0], resbrute[1])
# print(resbrute.x)
# print(resbrute["x"])
# print(resbrute)
# print({"x": resbrute[0], "fun": resbrute[1]})