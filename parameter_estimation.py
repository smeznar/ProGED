# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct 22 09:12:29 2020

# @author: Jure
# """


import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sympy as sp
# import sympy.core as sp
# from nltk import PCFG

# from model import Model
from model_box import ModelBox
# from generate import generate_models
# from generators.grammar import GeneratorGrammar



# """Methods for estimating model parameters. Currently implemented: differential evolution.

# Methods:
#     fit_models: Performs parameter estimation on given models. Main interface to the module.
# """

def model_error (model, params, X, Y):
    """Defines mean squared error as the error metric."""
    testY = model.evaluate(X, *params)
    res = np.mean((Y-testY)**2)
    if np.isnan(res) or np.isinf(res) or not np.isreal(res):
#        print(model.expr, model.params, model.sym_params, model.sym_vars)
        return 10**9
    return res

def model_constant_error (model, params, X, Y):
    """Alternative to model_error, intended to allow the discovery of physical constants.
    Work in progress."""
    
    testY = model.evaluate(X, *params)
    return np.std(testY)#/np.linalg.norm(params)
    

    # uporabi lamdify!!!!!
def ode1d(model, params, T, X_data, y0):
    if T.shape[0] != X_data.shape[0]: 
        raise IndexError("Number of samples in T and X does not match.")
    X = interp1d(T, X_data, kind='cubic')  # testiral, zgleda da dela.
    lamb_expr = sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
    def dy_dt(t, y):  # \frac{dy}{dt}
    # uporabi lamdify!!!!!
    # uporabi lamdify!!!!!
    # uporabi lamdify!!!!!
    # uporabi lamdify!!!!!
    # uporabi lamdify!!!!!
        # return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
        return lamb_expr(*np.array([[y[0], X(t)]]).T)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T) # spremeni v y0
    return Yode.y[0]

def model_ode_error (model, params, T, X, Y):
    """Defines mean squared error of solution to differential equation
    as the error metric.
        Input:
        - T is column of times at which samples in X and Y happen.
        - X are columns that do not contain variables that are derived.
        - Y is column containing variable that is derived.
    """
    
    # print("inside ode error. Model:", model)
    odeY = ode1d(model, params, T, X, y0=Y[0]) # spremeni v Y[:1]
    res = np.mean((Y-odeY)**2)
    # print("Before isnan. Model:", model, "res: ", res)
    if np.isnan(res) or np.isinf(res) or not np.isreal(res):
#        print(model.expr, model.params, model.sym_params, model.sym_vars)
        return 10**9
    return res

def optimization_wrapper (x, *args):
    """Calls the appropriate error function. The choice of error function is made here.
    
    TODO:
        We need to pass information on the choice of error function from fit_models all the way to here,
            and implement a library framework, similarly to grammars and generation strategies."""
    
    return model_error (args[0], x, args[1], args[2])

def optimization_wrapper_ODE (x, *args):
    """Calls the appropriate error function. The choice of error function is made here.
    
    TODO:
        We need to pass information on the choice of error function from fit_models all the way to here,
            and implement a library framework, similarly to grammars and generation strategies."""
    
    return model_ode_error(args[0], x, args[3], args[1], args[2])
    
def DE_fit (model, X, Y, p0, T="algebraic", **kwargs):
    """Calls scipy.optimize.differential_evolution. 
    Exists to make passing arguments to the objective function easier."""
    
    bounds = [[-10**1, 10**1] for i in range(len(p0))]
    # print("in DEfit. before if", bo)
    if isinstance(T, str):
        print("in DEfit. if is in algebraic", model)
        return differential_evolution(optimization_wrapper, bounds, args = [model, X, Y],
                                    maxiter=10**2, popsize=10)
    else:
        print("If in ode DEfit. Model:", model)
        return differential_evolution(optimization_wrapper_ODE, bounds, args = [model, X, Y, T],
                                    maxiter=10**2, popsize=10)
    
def min_fit (model, X, Y):
    """Calls scipy.optimize.minimize. Exists to make passing arguments to the objective function easier."""
    
    return minimize(optimization_wrapper, model.params, args = (model, X, Y))

def find_parameters (model, X, Y, T="algebraic"):
    """Calls the appropriate fitting function. 
    
    TODO: 
        add method name input, matching to a dictionary of fitting methods.
    """
#    try:
#        popt, pcov = curve_fit(model.evaluate, X, Y, p0=model.params, check_finite=True)
#    except RuntimeError:
#        popt, pcov = model.params, 0
#    opt_params = popt; othr = pcov
    print("in find_parametres")
    res = DE_fit(model, X, Y, p0=model.params, T=T)
    
#    res = min_fit (model, X, Y)
#    opt_params = res.x; othr = res
    
    return res

class ParameterEstimator:
    """Wraps the entire parameter estimation, so that we can pass the map function in fit_models
        a callable with only a single argument.
        Also checks some basic requirements, suich as minimum and maximum number of parameters.
        
        TODO:
            add inputs to make requirements flexible
            add verbosity input
    """
    def __init__(self, X, Y, T="algebraic"):
        self.X = X
        self.Y = Y
        self.T = T
        
    def fit_one (self, model):
        print("Estimating model " + str(model.expr))
        try:
            if len(model.params) > 5:
                pass
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun":model_error(model, [], self.X, self.Y)})
            else:
                print("Obicno, find parameters! Model:", model)
                res = find_parameters(model, self.X, self.Y, self.T)
                model.set_estimated(res)
        except:
            print("Excepted an error!! Model:", model)
            model.set_estimated({}, valid=False)
        return model
    
def fit_models (models, X, Y, T="algebraic", pool_map = map, verbosity=0):
    """Performs parameter estimation on given models. Main interface to the module.
    
    Supports parallelization by passing it a pooled map callable.
    
    Arguments:
        models (ModelBox): Instance of ModelBox, containing the models to be fitted. 
        X (numpy.array): Input data of shape N x M, where N is the number of samples 
            and M is the number of variables.
        Y (numpy.array): Output data of shape N x D, where N is the number of samples
            and D is the number of output variables.
        pool_map (function): Map function for parallelization. Example use with 8 workers:
                from multiprocessing import Pool
                pool = Pool(8)
                fit_models (models, X, Y, pool_map = pool.map)
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
    """
    estimator = ParameterEstimator(X, Y, T)
    return ModelBox(dict(zip(models.keys(), list(pool_map(estimator.fit_one, models.values())))))



if __name__ == "__main__":
    
    print("--- parameter_estimation.py test --- ")
    np.random.seed(2)
    from generate import generate_models    
    from pyDOE import lhs
    from generators.grammar import GeneratorGrammar
    
    
    def testf (x):
        return 3*x[:,0]*x[:,1]**2 + 0.5
    
    
    X = lhs(2, 10)*5
    y = testf(X)
   
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10

    models = generate_models(grammar, symbols, strategy_parameters = {"N":10})
    
    print(models, models[-1].params)
    # models1 = fit_models(models, X, y)    
    # print(models1, models1[-1].params, [model.params for model in models1])
    models2 = fit_models(models, X[:,0], y, np.linspace(1,50,X.shape[0]))    
    print(models2, models2[-1].params, [model.params for model in models2])

    # m = models[-1]
    # print(model_error(m,m.params, X, y), m.params)
    # m.params = []
    # print(m.params)
    # ParameterEstimator(X, y).fit_one(m)
    # print(model_error(m, [], X, y))
    # # models = fit_models(models, [], X, y)    

#########################
# ode radosti
    # print("Oda radosti.")
    
    # def ode1d(model, params, T, X_data, y0):
    #     if T.shape[0] != X_data.shape[0]: 
    #         raise IndexError("Number of samples in T and X does not match.")
    #     X = interp1d(T, X_data, kind='cubic')  # testiral, zgleda da dela.
    #     def dy_dt(t, y):  # \frac{dy}{dt}
    #         # model.evaluate(np.array([[y]+X(t)]), *params)  # if X is vector(array)
    #         return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    #     Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T)
    #     # plt.plot(T, Y, "r-")
    #     # plt.plot(T, Yode.y[0],'k--')
    #     return Yode.y[0]
    #     # return dy_dt(11,3)

    # def model_ode_error (model, params, T, X, Y):
    #     """Defines mean squared error of solution to differential equation
    #     as the error metric.
    #         Input:
    #         - T is column of times at which samples in X and Y happen.
    #         - X are columns that do not contain variables that are derived.
    #         - Y is column containing variable that is derived.
    #     """
    #     odeY = ode1d(model, params, T, X, y0=Y[0])
    #     res = np.mean((Y-odeY)**2)
    #     if np.isnan(res) or np.isinf(res) or not np.isreal(res):
    # #        print(model.expr, model.params, model.sym_params, model.sym_vars)
    #         return 10**9
    #     return res

    # print(X,y)
    # x1 = X[:,0]

    # def ode1d(model, params, T, X_data, y0):
    #     if T.shape[0] != X_data.shape[0]: 
    #         raise IndexError("Number of samples in T and X does not match.")
    #     X = interp1d(T, X_data, kind='cubic')  # testiral, zgleda da dela.
    #     lamb_expr = sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
    #     def dy_dt(t, y):  # \frac{dy}{dt}
    #     # uporabi lamdify!!!!!
    #     # uporabi lamdify!!!!!
    #     # uporabi lamdify!!!!!
    #     # uporabi lamdify!!!!!
    #     # uporabi lamdify!!!!!
    #         # return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    #         return lamb_expr(*np.array([[y[0], X(t)]]).T)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    #     Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T) # spremeni v y0
    #     return Yode.y[0]  
    
    # def model_ode_error (model, params, T, X, Y):
    #     """Defines mean squared error of solution to differential equation
    #     as the error metric.
    #         Input:
    #         - T is column of times at which samples in X and Y happen.
    #         - X are columns that do not contain variables that are derived.
    #         - Y is column containing variable that is derived.
    #     """
    #     odeY = ode1d(model, params, T, X, y0=Y[0]) # spremeni v Y[:1]
    #     res = np.mean((Y-odeY)**2)
    #     if np.isnan(res) or np.isinf(res) or not np.isreal(res):
    # #        print(model.expr, model.params, model.sym_params, model.sym_vars)
    #         return 10**9
    #     return res

    # T = np.linspace(5, 10, X.shape[0])
    # X = interp1d(T, x1, kind='cubic')
    # print("x1,y,T, X(11))", x1,y,T, X(11))
    # print("x1,y,T, X(11))", x1.shape,y.shape,T.shape, X(11).shape)
    # print("X(t)", X(11), type(X(11)), X(11).shape)
    # models = generate_models(grammar, symbols, strategy_parameters = {"N":1})
    # model = models[-1]
    # print(model.sym_vars,model.expr, (model.params), model.full_expr(*model.params))
    # labd = sp.lambdify(model.sym_vars, model.full_expr(*model.params), "numpy")
    # print(labd.__doc__)
    # print(labd(100, 1))
    # # print(labd([100, 1]))
    # print( labd( *np.array([[100, 1]]).T ))
    # ode = ode1d(model, model.params, T, X[:,0], y[0])
    # print(ode)
    
    # import matplotlib.pyplot as plt
    # plt.plot(T, y,"r-")
    # plt.plot(T, ode,'k--')
    
    # err = model_ode_error(model, model.params, T, X[:,0], y)
    # er = model_error(model, model.params, X, y)
    # # print(models)
    # print(err)
    # print(er)
    # model.evaluate(np.array([y[0]]+[X(11)]), *model.params) 
    # print(model.evaluate(np.array([[y[0]]+[X(11)]]), *model.params) )


