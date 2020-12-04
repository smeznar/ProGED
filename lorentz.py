# Simulate Lorentz's system ODE and discover edes

import logging
logging.basicConfig(filename="myfile.log", level=logging.INFO)  # Overwrites
# my.log with program output.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# # 1.) Data construction (simulation of Lorentz):

T = np.linspace(0.48, 0.85, 1000)
# # Lorentz's sode:
# dx/dt = \sigma * (y-x)
# dy/dt = x*(\rho-z) - y
# dz/dt = x*y - \beta*z
sigma = 1.3  # 1 # 0 
rho = -15 # 1 # 0
beta = 3.4  # 1 # 0
y0 = [0.1, 0.4, 0.5]
def dy_dt(t, ys):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
    # \dot{y} = y^. = [y1^., y2^., y3^., ...]
    x, y, z = ys
    return [sigma * (y-x), x*(rho-z) - y, x*y - beta*z]
Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)
# plot simulated data:
plt.xlabel("T [time]")
plt.ylabel("solutions [ys(t)]")
plt.plot(T, Yode.y[0], label="solution x")
plt.plot(T, Yode.y[1], label="solution y")
plt.plot(T, Yode.y[2], label="solution z")
plt.legend()
plt.show()

data = np.concatenate((T[:, np.newaxis], Yode.y.T), axis=1)  # Embed Time column into dataset


# # # # 2.) Discover one ode at a time.

from generate import generate_models
# from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template  # Grammar
#nonterminals will depend on given dataset.
from parameter_estimation import fit_models
np.random.seed(0)

def eq_disco_demo (data, lhs_variables: list = ["stolpec 1"],
    rhs_variables: list = [2, 3],
    dimension = [0]):
    # header = ["column for x", "column for y", "column for z"]
    header = ["c_x", "c_y", "c_z"]
    T = data[:, 0]
    Y = data[:, features_on_left]
    X = data[:, features_on_right]
    variables = ["'"+header[i-1]+"'" for i in features_on_left] # [1,3] -> ["x1", "x3"]
    variables += ["'"+header[i-1]+"'" for i in features_on_right]
    print(variables)
    symbols = {"x": variables, "start":"S", "const":"C"}
    # start eq. disco.:
    grammar = grammar_from_template("polynomial", {
        "variables": variables,
        "p_S": [0.4, 0.6],
        "p_T": [0.4, 0.6],
        "p_vars": [0.33, 0.33, 0.34],
        "p_R": [1, 0],
        "p_F": [],
        "functions": []
    })
    print(grammar)
    models = generate_models(grammar, symbols, strategy_parameters = {"N":10})
    fit_models(models, X, Y, T)
    # print results:
    print(models)
    print("\nFinal score:")
    for m in models:
        print(f"model: {str(m.get_full_expr()):<70}; error: {m.get_error()}")
    return 1

# eq_disco_demo(data, features_on_left=[2], features_on_right=[1,3])
eq_disco_demo(data, features_on_left=[3], features_on_right=[1,2])
# eq_disco_demo(data, features_on_left=[1], features_on_right=[2,3])
