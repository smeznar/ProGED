#%% dataset
import sympy as sp
import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #,  odeint
from scipy.interpolate import interp1d
from typing import List

# # go-away message: odeint works perfectly.
# %% 

# plt.plot(ts,X[:,1],"r-")
# plt.plot(ts, odey,'k--')

# Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T, atol=0) # spremeni v y0
# Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), method='LSODA', t_eval=T, atol=0) 
# Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), method='LSODA', t_eval=T) 




def simerr(y1,y2): return np.mean((y1-y2)**2)

# plt.plot(ts,X[:,1],"r-")
# # plt.plot(ts, odey,'g-')
# # plt.plot(ts, ode2,"k--")
# plt.plot(ts, ode4,"b*")
# print(np.mean((ode2-odey)**2))

# def ode_plot(ts, Xs, dx):
#     Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts)
#     plt.plot(ts,Xs,"r-")
#     plt.plot(ts, Xode.y[0],'k--')

np.random.seed(2)
from generate import generate_models    
from pyDOE import lhs
from generators.grammar import GeneratorGrammar


def testg (x):
    return 5*x[:,[0]]**2 - 2*x[:,[0]]**3 + 0.5
X2 = lhs(1, 10)*5
y2 = testg(X2)
print(X2, y2, X2.ndim, X2.shape, y2.ndim, y2.shape)


def testf (x):
    return 3*x[:,0]*x[:,1]**2 + 0.5

X = lhs(2, 10)*5
y = testf(X)
grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
models = generate_models(grammar, symbols, strategy_parameters = {"N":3})
T = np.linspace(1,50,X2.shape[0])
m = models[-1]
print([m], [m.params])
X1 = X[:,[0]]

from parameter_estimation import model_ode_error
# odey = ode_almost([m], [m.params], T, X1, np.array([y[0]]))

def ode_almost(models_list, params_matrix, T, X_data, y0, **kwargs):
    """Solves ode defined by model.
        Input specs:
        - models_list is list (not dictionary) of models that e.g.
        generate_models() generates.
        - params_matrix is list of ndarrays of parameters for
        corresponding models.
        - y0 is array (1-dim) of initial value of vector function y(t)
        i.e. y0 = y(T[0]) = [y1(T[0]), y2(T[0]), y3(T[0]),...].
        - X_data is 2-dim array (matrix) i.e. X = [X[0,:], X[1,:],...].
        - T is (1-dim) array, i.e. of shape (N,)
    """
    if not (isinstance(models_list, list)
            and (isinstance(params_matrix, list) 
                and len(params_matrix)>0 
                and isinstance(params_matrix[0], (list, np.ndarray)))
            and X_data.ndim == 2
            and y0.ndim == 1):
        print(type(params_matrix[0]))
        print(isinstance(models_list, list), 
            isinstance(params_matrix, list), 
            len(params_matrix)>0, 
            isinstance(params_matrix[0], (list, np.ndarray)),
            X_data.ndim == 2,
            y0.ndim == 1 )
        print("Programmer's defined error: Input arguments are not"
                        +" in required form! Bugs can happen.")
        raise TypeError("Programmer's defined error: Input arguments are not"
                        +" in required form! Bugs can happen.")
    elif not T.shape[0] == X_data.shape[0]: 
        print("Number of samples in T and X does not match.")
        raise IndexError("Number of samples in T and X does not match.")
    elif not (y0.shape[0] == len(models_list)  #len(equations)=len(models used)
            and len(models_list[0].sym_vars) == X_data.shape[1] + y0.shape[0]): 
        print("Number of symbols in models and combination of "
                        + "number of equations and dimensions of input data"
                        + " does not match.")
        raise IndexError("Number of symbols in models and combination of "
                        + "number of equations and dimensions of input data"
                        + " does not match.")
    ### 1-dim version of X_data currently: ###
    X = interp1d(T, X_data.T[0], kind='cubic')  # 1 -dim
    lamb_exprs = [
        sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
        for model, params in zip(models_list, params_matrix)
    ]
    def dy_dt(t, y):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
        ### 1-dim interpol causes ###
        b = np.concatenate((y,np.array([X(t)]))) # =[y,X(t)] =[y,X1(t),X2(t),...] 
        return np.array([lamb_expr(*b) for lamb_expr in lamb_exprs])  # older version with *b.T
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, **kwargs)
    # method='LSODA', t_eval=T, atol=0
    # print(f"Status: {Yode.status}, Success: {Yode.success}, message: {Yode.message}.")
    return Yode.y

odey = ode_almost([m], [m.params], T, X2, y2[0])
odey_atol = ode_almost([m], [m.params], T, X2, y2[0], atol=0)
odey_lsoda = ode_almost([m], [m.params], T, X2, y2[0], method='LSODA')
# print(odey)
# for n, i in zip(['k.', 'gx', 'b--'],[odey, odey_atol, odey_lsoda]):
#     print(simerr(i.T, y2), (i.T-y2)**2)
#     print(i)
#     # plt.figure(0)
#     plt.plot(T, y2,"r-")
#     plt.plot(T, odey[0], n)
#     # plt.pause(5)
# # print([m], [m.params], T, X2, y2[0])
# print(y2)

from parameter_estimation import fit_models

models = generate_models(grammar, symbols, strategy_parameters = {"N":10})
models2 = fit_models(models, X2, y2, np.linspace(1,50,X.shape[0]))    
print(models2)
