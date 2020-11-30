import logging
logging.basicConfig(filename="my.log", level=logging.INFO)



# simulate Lorentz's system ODE

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #,  odeint
from scipy.interpolate import interp1d


# # 1.) data construction:

T = np.linspace(0.48, 0.85, 1000)
# # print(T.shape, T.ndim)#, T)

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
    # [ \sigma * (y[1]-y[0]), y[0]*(\rho-y[2]) - y, y[0]*y[1] - \beta*y[2] ]y[1]
    return [sigma * (y-x), x*(rho-z) - y, x*y - beta*z]

dummy = T*np.array([[1], [2], [3]])
Yode = np.array([T*2, T*1, T*3])
Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)
# # method='LSODA', t_eval=T, atol=0
# # print(f"Status: {Yode.status}, Success: {Yode.success}, message: {Yode.message}.")
# # print(Yode.y.shape, Yode.y.ndim)
xlabel = "T [time]"
ylabel = "solutions [ys(t)]"
plt.plot(T, Yode.y[0], label="solution x")
plt.plot(T, Yode.y[1], label="solution y")
plt.plot(T, Yode.y[2], label="solution z")
plt.legend()
# plt.show()
data = np.concatenate((T[:, np.newaxis], Yode.y.T), axis=1)
# data = np.concatenate((T[:, np.newaxis], dummy.T), axis=1)
# print(data.shape)
# print(T[:, np.newaxis].shape, dummy.T.shape)
# print(dummy.shape)


# # # # 2.) solve 1-dim ode.

from generate import generate_models
from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template
from parameter_estimation import fit_models
np.random.seed(0)

# grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
#                             T -> 'C' [0.6] | T "*" V [0.4]
#                             V -> 'x' [0.33] | 'x' [0.33] | 'y' [0.345]""")
# symbols = {"x":['x', 'y', 'z'], "start":"S", "const":"C"}


# # grammar = grammar_from_template("universal", {"variables":["'phi'", "'theta'", "'r'"], "p_vars":[0.2,0.4,0.4]})

def eq_disco (data, features_on_left: list = [1],
    features_on_right: list = [1, 2, 3]):
    # stolpci_desno = [i for i in range(0, data.shape[1])]
    # header = ["col1", "col2", "col3"]
    header = ["c_x", "c_y", "c_z"]
    T = data[:, 0]  # not [0], want 1D array
    Y = data[:, features_on_left]
    X = data[:, features_on_right]
    print(T.shape, Y.shape, X.shape)
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

    print(models)
    print("\nFinal score:")
    for m in models:
        print(f"model: {str(m.get_full_expr()):<60}; error: {m.get_error()}")
    return 1

# print(eq_disco(data, features_on_left=[2], features_on_right=[1,3]))
# print(eq_disco(data, features_on_left=[3], features_on_right=[1,2]))
print(eq_disco(data, features_on_left=[1], features_on_right=[2,3]))

# fit_models(models, )

# ts = np.arange(10)
# # x1x2 = np.concatenate((ts[np.newaxis, :]**2, ts[np.newaxis, :]**3), axis=0)
# x1x2 = np.concatenate((ts[:, np.newaxis]**2, ts[:, np.newaxis]**3), axis=1)
# print(x1x2)
# print(x1x2.shape, ts.shape)
# xf = interp1d(ts, x1x2, axis=0, kind="cubic")
# print(xf(ts).T,"\n", x1x2.T)

# print((xf(ts) == x1x2))
# print(np.all(xf(ts) == x1x2))
# print(xf())
# print(ts, ts[1], xf(0.5), xf(6.4), 6.4**2, 6.4**3)
# print(np.concatenate((np.arange(3), np.arange(4))))