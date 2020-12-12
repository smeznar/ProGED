"""Simulate Lorentz's system ODE and discover edes.

    Script accepts also optional comand line arguments:
arg0 -- number of samples/models
arg1 -- custom nickname of log that is added to the log filename, which is of
    the form: log_lorenz_<custom nickname><random number>.log
"""

import sys  # To import from parent directory.

# from IPython.utils.io import Tee  # Log results using 3th package.
from tee_so import Tee  # Log using manually copied class from a forum.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# # 0.) Log output to lorenz_log_<random>.log file

# # # Input: # # # 
eqation = "123"  # Code for eq_disco([1], [2,3]).
samples_cardinality = 50 
log_nickname = ""
if len(sys.argv) >= 2:
    samples_cardinality = int(sys.argv[1])
if len(sys.argv) >= 3:
    log_nickname = sys.argv[2]
if len(sys.argv) >= 4:
    eqation = sys.argv[3]
aux = [int(i) for i in eqation]
aquation = (aux[:1], aux[1:])
random = str(np.random.random())
print(log_nickname + random)
try:
    log_object = Tee("examples/log_lorenz_" + log_nickname + random + ".log")
except FileNotFoundError:
    log_object = Tee("log_lorenz_" + log_nickname + random + ".log")


# # 1.) Data construction (simulation of Lorenz):

np.random.seed(0)
T = np.linspace(0.48, 0.85, 1000)
# # Lorenz's sode:
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
# plt.show()

data = np.concatenate((T[:, np.newaxis], Yode.y.T), axis=1)  # Embed Time column into dataset


# # # # 2.) Discover one ode at a time.

sys.path += ['.','..']
from generate import generate_models
# from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template  # Grammar's
#nonterminals will depend on given dataset.
from parameter_estimation import fit_models

def eq_disco_demo (data, lhs_variables: list = [1],
                  # ["stolpec 1"], # in case of header string reference
                    rhs_variables: list = [2, 3],
                    dimensions: list = [0]):
    # header = ["column for x", "column for y", "column for z"]
    header = ["x", "y", "z"]
    T = data[:, dimensions]
    # print(T.shape, "T")
    T = T.T[0]  # Temporary line since T is for still 1-D array.
    Y = data[:, lhs_variables]
    X = data[:, rhs_variables]
    variables = ["'"+header[i-1]+"'" for i in lhs_variables] # [1,3] -> ["x1", "x3"]
    variables += ["'"+header[i-1]+"'" for i in rhs_variables]
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
    print(samples_cardinality, "=samples cardinality")
    models = generate_models(grammar, symbols, 
                            strategy_parameters={"N":samples_cardinality})
    fit_models(models, X, Y, T)
    # print results:
    print(models)
    print("\nFinal score:")
    for m in models:
        if m.get_error() < 10**(-3):  # or True:
            print(f"model: {str(m.get_full_expr()):<70}; error: {m.get_error()}")
    return 1

# eq_disco_demo(data, lhs_variables=[2], rhs_variables=[1,3])
# eq_disco_demo(data, lhs_variables=[3], rhs_variables=[1,2])
# eq_disco_demo(data, lhs_variables=[1], rhs_variables=[2,3])
eq_disco_demo(data, lhs_variables=aquation[0], rhs_variables=aquation[1])
print(aquation[0], aquation[1])

