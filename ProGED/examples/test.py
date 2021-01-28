"""Simulate Lorentz's system ODE and discover edes.

    Script accepts also optional comand line arguments:
arg0 -- number of samples/models
arg1 -- custom nickname of log that is added to the log filename, which is of
    the form: log_lorenz_<custom nickname><random number>.log
"""

import time
import os
import sys  # To import from parent directory.

import tee_so as te # Log using manually copied class from a forum.

import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint


# # 0.) Log output to lorenz_log_<random>.log file

# # 1.) Data construction (simulation of Lorenz):

np.random.seed(0)
# T = np.linspace(0.48, 0.85, 1000)  # Times currently run at.
T = np.linspace(0, 40, 1000)  # Chaotic Lorenz times noted on Wiki.
# # Lorenz's sode:
# dx/dt = \sigma * (y-x)
# dy/dt = x*(\rho-z) - y
# dz/dt = x*y - \beta*z
# non-chaotic configuration:
# sigma = 1.3  # 1 # 0 
# rho = -15  # 1 # 0
# beta = 3.4  # 1 # 0
# Chaotic configuration:
sigma = 10  # 1 # 0 
y0 = [1, 1]
# y0 = [1, 1, 1]  # Chaotic Lorenz initial values noted on Wiki.
X = 1 + T**2
def dy_dt(t, ys):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
    # \dot{y} = y^. = [y1^., y2^., y3^., ...]
    return [sigma * (ys[0]-)]
# Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)
# max_steps = 10**6
# Convert max_steps to min_step:
# min_step_from_max_steps = abs(T[-1] - T[0])/max_steps
# The minimal min_step to avoid min step error in LSODA:
min_step = 10**(-15)
# min_step = max(min_step_from_max_steps, min_step_error)  # Force them both.
rtol=10**(-6)
# Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, method="LSODA", min_step=min_step,
                #  t_eval=T, rtol=rtol, atol=0).y
Yode = odeint(dy_dt, y0, T, rtol=rtol, atol=0, tfirst=True, printmessg=0, hmin=min_step).T 
# Yode = odeint(dy_dt, y0, T, rtol=rtol, atol=0, tfirst=True, printmessg=0, hmin=min_step)
data = np.hstack((T[:, np.newaxis], Yode.T))  # Embed Time column into dataset.
# data = np.hstack((T[:, np.newaxis], Yode))  # Embed Time column into dataset.


# # # # 2.) Discover one ode at a time.

sys.path += ['.','..']
from generate import generate_models
# from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template  # Grammar's
#nonterminals will depend on given dataset.
from parameter_estimation import fit_models

def eq_disco_demo (data):
    # header = ["column for x", "column for y", "column for z"]
    variables = ["'x'", "'y'", "'z'"]
    symbols = {"x": variables, "start":"S", "const":"C"}
    grammar = grammar_from_template("polynomial", {
        "variables": variables,
        "p_S": [0.4, 0.6],
        "p_T": [0.4, 0.6],
        "p_vars": [0.33, 0.33, 0.34],
        "p_R": [1, 0],
        "p_F": [],
        "functions": []
    })
    samples_size = 50
    models = generate_models(grammar, symbols, 
                            strategy_settings={"N":samples_size})
    fit_models(models, data, target_variable_index=-1, time_index=0,
                timeout=5, max_ode_steps=10**6, task_type="differential",
                lower_upper_bounds=(-15, 15))
    # print(models)
    # print("\nFinal score:")
    # for m in models:
    #     if m.get_error() < 10**(-3) or True:
    #         print(f"model: {str(m.get_full_expr()):<70}; "
    #                 + f"p: {m.p:<23}; "
    #             + f"error: {m.get_error()}")
    return 1

eq_disco_demo(data)  #, lhs_variables=aquation[0], rhs_variables=aquation[1])
