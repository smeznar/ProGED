"""Simulate Lorentz's system ODE and discover edes.

    Script accepts also optional comand line arguments:
arg0 -- number of samples/models
arg1 -- custom nickname of log that is added to the log filename, which is of
    the form: log_lorenz_<custom nickname><random number>.log
"""

import time
import os
import sys  # To import from parent directory.

import ProGED.examples.tee_so as te  # Log using manually copied class from a forum.

import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint


# # 0.) Log output to lorenz_log_<random>.log file

start = time.perf_counter()
# # # Input: # # # 
eqation = "123"  # Code for eq_disco([1], [2,3]).
sample_size = 5
# sample_size = 30  # It finds the equation at 30.
log_nickname = ""
isTee = False
# is_chaotic_wiki = "cw"
# is_chaotic_wiki = "c0"  # chaotic but not wiki
is_chaotic_wiki = "__"  # For hyperopt: not chaotic, not wiki.
if len(sys.argv) >= 2:
    sample_size = int(sys.argv[1])
if len(sys.argv) >= 3:
    isTee = True
    log_nickname = sys.argv[2]
if len(sys.argv) >= 4:
    eqation = sys.argv[3]
if len(sys.argv) >= 5:
    is_chaotic_wiki = sys.argv[4]
aux = [int(i) for i in eqation]
aquation = (aux[:1], aux[1:])
random = str(np.random.random())
print("Filename id: " + log_nickname + random)
if isTee:
    try:
        log_object = te.Tee("examples/log_lorenz_" + log_nickname + random + ".txt")
    except FileNotFoundError:
        log_object = te.Tee("log_lorenz_" + log_nickname + random + ".txt")
if len(is_chaotic_wiki) != 2:
    # print("Wrong cmd argument for chaotic/calm wiki/my diff. eq. configuration")
    print("Wrong 5th chaotic+wiki cmd argument: should be of length 2, e.g. cw or 0w.")
else:
    c, w = is_chaotic_wiki[0], is_chaotic_wiki[1]
    is_chaotic = True if c == "c" else False
    is_wiki = True if w == "w" else False

# # 1.) Data construction (simulation of Lorenz):

np.random.seed(0)
# is_chaotic = True
# is_wiki = False
T = np.linspace(0.48, 0.85, 1000)  # Times currently run at.
if is_wiki:
    T = np.linspace(0, 40, 4000)  # Chaotic Lorenz times noted on Wiki.
# # Lorenz's sode:
# dx/dt = \sigma * (y-x)
# dy/dt = x*(\rho-z) - y
# dz/dt = x*y - \beta*z
# non-chaotic configuration:
sigma = 1.3  # 1 # 0 
rho = -15  # 1 # 0
beta = 3.4  # 1 # 0
# Chaotic configuration:
if is_chaotic:
    sigma = 10  # 1 # 0 
    rho = 28  # 1 # 0
    beta = 8/3  # 1 # 0
y0 = [0.1, 0.4, 0.5]  # Lorenz initial values run at.
if is_wiki:
    y0 = [1, 1, 1]  # Chaotic Lorenz initial values noted on Wiki.
def dy_dt(t, ys):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
    # \dot{y} = y^. = [y1^., y2^., y3^., ...]
    x, y, z = ys
    return [sigma * (y-x), x*(rho-z) - y, x*y - beta*z]
# Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)
max_steps = 10**6
# Convert max_steps to min_step:
min_step_from_max_steps = abs(T[-1] - T[0])/max_steps
# The minimal min_step to avoid min step error in LSODA:
min_step_error = 10**(-15)
min_step = max(min_step_from_max_steps, min_step_error)  # Force them both.
rtol=10**(-6)
Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, method="LSODA", min_step=min_step, t_eval=T, rtol=rtol, atol=0).y
# Yode = odeint(dy_dt, y0, T, rtol=rtol, atol=0, tfirst=True, printmessg=0, hmin=min_step).T 

# # Plot simulated data:
# plt.xlabel("T [time]")
# plt.ylabel("solutions [ys(t)]")
# plt.plot(T, Yode[0], label="solution x")
# plt.plot(T, Yode[1], label="solution y")
# plt.plot(T, Yode[2], label="solution z")
# plt.legend()
# plt.show()
data = np.concatenate((T[:, np.newaxis], Yode.T), axis=1)  # Embed Time column into dataset.


# # # # 2.) Discover one ode at a time.

sys.path += ['.','..']
# from ProGED.generate import generate_models
from ProGED.equation_discoverer import EqDisco
# from ProGED.generators.grammar import GeneratorGrammar
# from ProGED.generators.grammar_construction import grammar_from_template  # Grammar's
#nonterminals will depend on given dataset.
# from ProGED.parameter_estimation import fit_models
from ProGED.parameter_estimation import DE_fit, DE_fit_metamodel



# Old function, when EqDisco() object was not yet implemented.
# Now probably useless. Try to use EqDisco object instead.
def eq_disco_demo (data, lhs_variables: list = [1],
                  # ["column 1"], # in case of header string reference
                    rhs_variables: list = [2, 3]):
    # header = ["column for x", "column for y", "column for z"]
    header = ["x", "y", "z"]
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
    print(sample_size, "= sample size")
    models = generate_models(grammar, symbols, 
                            strategy_settings={"N":sample_size})
    fit_models(
        models, 
        data, 
        target_variable_index=-1,  # or maybe = aquation[0][0]
        time_index=0,
        task_type="differential",
        estimation_settings={
            "timeout": 5, 
            "max_ode_steps": 10**6, 
            "lower_upper_bounds": (-30, 30),
            "verbosity": 1,
            }
        )
    print(models)
    print("\nFinal score:")
    for m in models:
        if m.get_error() < 10**(-3) or True:
            print(f"model: {str(m.get_full_expr()):<70}; "
                    + f"p: {m.p:<23}; "
                + f"error: {m.get_error()}")
    return 1

# eq_disco_demo(data, lhs_variables=[2], rhs_variables=[1,3])
# eq_disco_demo(data, lhs_variables=[3], rhs_variables=[1,2])
# eq_disco_demo(data, lhs_variables=[1], rhs_variables=[2,3])

# Run eqation discovery from command line:
np.random.seed(0)
# eq_disco_demo(data, lhs_variables=aquation[0], rhs_variables=aquation[1])
finnish = time.perf_counter()
print(f"Finnished in {round(finnish-start, 2)} seconds")

np.random.seed(0)
ED = EqDisco(data = data,
             task = None,
             task_type = "differential",
             time_index = 0,
             # target_variable_index = -1,
             target_variable_index = aquation[0][0],  # aquation = [123] -> target = 1 -> ([t,x,y,z]->x)
             # target_variable_index = 1,
             # variable_names=["t", "x", "y"],
             variable_names=["t", "x", "y", "z"],
             generator = "grammar",
             generator_template_name = "polynomial",
             # generator_settings={"variables": ["'an_2'", "'an_1'"], "p_T": p_T, "p_R": p_R},
             generator_settings={
                 # "variables": ["'x'", "'y'"],
                 "p_S": [0.4, 0.6],
                 "p_T": [0.4, 0.6],
                 "p_vars": [0.33, 0.33, 0.34],
                 "p_R": [1, 0],
                 "p_F": [],
                 "functions": [],
             },
             sample_size = sample_size,
             # verbosity = 1)
             verbosity = 4)
ED.generate_models()
ED.fit_models(
    estimation_settings={
        "timeout": 5, 
        "max_ode_steps": 10**6, 
        "lower_upper_bounds": (-30, 30),
        "optimizer": DE_fit,
        # "optimizer": DE_fit_metamodel,
        # "verbosity": 4,
        "verbosity": 1,
        })

print(ED.models)
print("\nFinal score:")
for m in ED.models:
    if m.get_error() < 10**(-3) or True:
        print(f"model: {str(m.get_full_expr()):<70}; "
                + f"p: {m.p:<23}; "
            + f"error: {m.get_error()}")
finnish = time.perf_counter()
print(f"Finnished in {round(finnish-start, 2)} seconds")
