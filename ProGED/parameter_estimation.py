# -*- coding: utf-8 -*-

import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize, brute, shgo, dual_annealing
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint
import sympy as sp
from sklearn import ensemble #, tree

import ProGED.mute_so as mt
from _io import TextIOWrapper as stdout_type

from ProGED.model_box import ModelBox
from ProGED.task import TASK_TYPES

# glitch-doctor downloaded from github:
# from ProGED.glitch_doctor.metamodel import Metamodel
# import ProGED.glitch_doctor.metamodel.Metamodel
# import ProGED.glitch_doctor.model.Model

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="overflow encountered in square")
warnings.filterwarnings("ignore", message="overflow encountered in double_scalars")


"""Methods for estimating model parameters. Currently implemented: differential evolution.

Methods:
    fit_models: Performs parameter estimation on given models. Main interface to the module.
"""

DUMMY = 10**30

def model_error (params, model, X, Y, *residue):
    """Defines mean squared error as the error metric."""
    try:
        testY = model.evaluate(X, *params)
        res = np.mean((Y-testY)**2)
        if np.isnan(res) or np.isinf(res) or not np.isreal(res):
            # print("isnan(res), ... ")
            # print(model.expr, model.params, model.sym_params, model.sym_vars)
            return DUMMY
        return res
    except Exception as error:
        print("Programmer1 model_error: Params at error:", params, f"and {type(error)} with message:", error)
        return DUMMY

# def model_constant_error (model, params, X, Y):
#     """Alternative to model_error, intended to allow the discovery of physical constants.
#     Work in progress."""
    
#     testY = model.evaluate(X, *params)
#     return np.std(testY)#/np.linalg.norm(params)


def model_error_general (params, model, X, Y, T, **estimation_settings):
    """Calculate error of model with given parameters in general with
    type of error given.

        Input = TODO:
    - X are columns without features that are derived.
    - Y are columns of features that are derived via ode fitting.
    - T is column of times at which samples in X and Y happen.
    - estimation_settings: look description of fit_models()
    """
    task_type = estimation_settings["task_type"]
    if task_type == "algebraic":
        return model_error(params, model, X, Y)
    elif task_type == "oeis":
        return model_error(params, model, X, Y)
    elif task_type == "differential":
        # Model_ode_error might use estimation[verbosity] agrument for
        # ode solver's settings and suppresing its warnnings:
        return model_ode_error(params, model, X, Y, T, estimation_settings)
    else:
        types_string = "\", \"".join(TASK_TYPES)
        raise ValueError("Variable task_type has unsupported value "
                f"\"{task_type}\", while list of possible values: "
                f"\"{types_string}\".")

def ode (models_list, params_matrix, T, X_data, y0, **estimation_settings):
    """Solve system of ODEs defined by equations in models_list.

    Raise error if input is incompatible.
        Input:
    models_list -- list (not dictionary) of models that e.g.
        generate_models() generates.
    params_matrix -- list of lists or ndarrays of parameters for
        corresponding models.
    y0 -- array (1-dim) of initial value of vector function y(t)
        i.e. y0 = y(T[0]) = [y1(T[0]), y2(T[0]), y3(T[0]),...].
    X_data -- 2-dim array (matrix) i.e. X = [X[0,:], X[1,:],...].
    T -- (1-dim) array, i.e. of shape (N,)
    max_ode_steps -- maximal number of steps inside ODE solver to
        determine the minimal step size inside ODE solver.
        Output:
    Solution of ODE evaluated at times T.
    """
    if not (isinstance(models_list, list)
            and (isinstance(params_matrix, list)
                and len(params_matrix)>0
                and isinstance(params_matrix[0], (list, np.ndarray)))
            and X_data.ndim == 2
            and y0.ndim == 1):
        message = str(type(params_matrix[0])) + "\n"
        info = (isinstance(models_list, list),
            isinstance(params_matrix, list),
            len(params_matrix)>0,
            isinstance(params_matrix[0], (list, np.ndarray)),
            X_data.ndim == 2,
            y0.ndim == 1 )
        print(message, info)
        print("Programmer's defined error: Input arguments are not"
                        +" in required form!")
        raise TypeError(f"Programmer's defined error: Input arguments are not"
                        +f" in required form!"
                        +f"\n{message, info}")
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
    X = interp1d(T, X_data, axis=0, kind='cubic', fill_value="extrapolate")  # N-D
    lamb_exprs = [
        # sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
        model.lambdify(*params)
        for model, params in zip(models_list, params_matrix)
    ]
    def dy_dt(t, y):
        """Represents  \frac{dy}{dt}.

        y -- [y1,y2,y3,...] i.e. ( shape= (n,) ) """

        # N-D:
        b = np.concatenate((y, X(t))) # =[y,X(t)] =[y,X1(t),X2(t),...]
        # Older version with *b.T:
        return np.array([lamb_expr(*b) for lamb_expr in lamb_exprs])
    # Older (default RK45) method:
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)  
    # Set min_step via prescribing maximum number of steps:
    if "max_ode_steps" in estimation_settings:
        max_steps = estimation_settings["max_ode_steps"]
    else:
        # max_steps = 10**6  # On laptop, this would need less than 3 seconds.
        max_steps = T.shape[0]*10**3  # Set to |timepoints|*1000.
    # Convert max_steps to min_step:
    min_step_from_max_steps = abs(T[-1] - T[0])/max_steps
    # The minimal min_step to avoid min step error in LSODA:
    min_step_error = 10**(-15)
    min_step = max(min_step_from_max_steps, min_step_error)  # Force them both.
    rtol = 10**(-4)
    atol = 10**(-6)
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, method="LSODA", rtol=rtol, atol=atol, min_step=min_step).y
    # Alternative LSODA using odeint (may be faster?):
    Yode = odeint(dy_dt, y0, T, rtol=rtol, atol=atol, tfirst=True, hmin=min_step).T 
    return Yode

def model_ode_error (params, model, X, Y, T, estimation_settings):
    """Defines mean squared error of solution to differential equation
    as the error metric.

        Input:
        - T is column of times at which samples in X and Y happen.
        - X are columns without features that are derived.
        - Y are columns of features that are derived via ode fitting.
    """
    model_list = [model]; params_matrix = [params] # 12multi conversion (temporary)
    DUMMY = 10**9
    try:
        # Next few lines strongly suppress any warnning messages 
        # produced by LSODA solver, called by ode() function.
        # Suppression further complicates if making log files (Tee):
        change_std2tee = False  # Normaly no need for this mess.
        if not isinstance(sys.stdout, stdout_type):
            # In this case the real standard output (sys.stdout) is not
            # saved in original location sys.stdout. We have to obtain
            # it inside of Tee object (look module tee_so).
            tee_object = sys.stdout  # obtain Tee object that has sys.stdout
            std_output = tee_object.stdout  # Obtain sys.stdout.
            sys.stdout = std_output  # Change fake stdout to real stdout.
            change_std2tee = True  # Remember to change it back.
        # Next line works only when sys.stdout is real. Thats why above.
        with open(os.devnull, 'w') as f, mt.stdout_redirected(f):
            odeY = ode(model_list, params_matrix, T, X, y0=Y[:1],
                        **estimation_settings)  # Y[:1] if _ or Y[0] if |
        if change_std2tee: 
            sys.stdout = tee_object  # Change it back to fake stdout (tee).

        # odeY = odeY.T  # solve_ivp() returns in _ oposite (DxN) shape.
        odeY = odeY[0]  # If Y is landscape, i.e. _.
        if not odeY.shape == Y.shape:
            if estimation_settings["verbosity"] >= 3:
                print("The ODE solver did not found ys at all times -> returning dummy error.")
            if estimation_settings["verbosity"] >= 4:
                print(odeY.shape, Y.shape)
            return DUMMY
        try:
            res = np.mean((Y-odeY)**2)
            if estimation_settings["verbosity"] >= 4:
                print("succesfully returning now inside model_ode_error")
            if np.isnan(res) or np.isinf(res) or not np.isreal(res):
# #                print(model.expr, model.params, model.sym_params, model.sym_vars)
                return DUMMY
            return res
        except Exception as error:
            print("Programmer1 ode() mean(Y-odeY): Params at error:", params, f"and {type(error)} with message:", error)
            return DUMMY

    except Exception as error:
        print("Programmer: Excerpted an error inside ode() of model_ode_error.")
        print("Programmer: Params at error:", params, f"and {type(error)} with message:", error)
        print("Returning dummy error. All is well.")
        return DUMMY

def model_oeis_error (params, model, X, Y, _T, estimation_settings):
    """Defines mean squared error as the error metric."""
    if estimation_settings["verbosity"] >= 5:
        print(params, "print: params before rounding")
    try:
        params = np.round(params)
        if estimation_settings["verbosity"] >= 4:
            print(params, "print: params after round")
        testY = model.evaluate(X, *params)
        res = np.mean((Y-testY)**2)
        if np.isnan(res) or np.isinf(res) or not np.isreal(res):
            if estimation_settings["verbosity"] >= 2:
                print("isnan(res), ... ")
                print(model.expr, model.params, model.sym_params, model.sym_vars)
            return DUMMY
        return res
    except Exception as error:
        print("Programmer1 model_oeis_error: Params at error:", params, f"and {type(error)} with message:", error)
        return DUMMY

def hyperopt_fit (model, X, Y, T, p0, **estimation_settings):
    """Calls Hyperopt.
    Exists to make passing arguments to the objective function easier."""

    from hyperopt import hp, fmin, rand
    lu_bounds = estimation_settings["lower_upper_bounds"]
    lower_bound, upper_bound = lu_bounds[0]+1e-30, lu_bounds[1]+1e-30
    # lower_bound, upper_bound = lu_bounds[0], lu_bounds[1]
    print("This is *hp.randint* version of Hyperopt running.")
    # Currently implemented Hyperopt optimisation via hp.randint only.
    space = [hp.randint('C'+str(i), lower_bound, upper_bound)
                 for i in range(len(p0))]
    def objective(params):
        return estimation_settings["objective_function"](
            params, model, X, Y, T, estimation_settings)
    # Use user's hyperopt specifications or use the default ones:
    algo = estimation_settings.get("hyperopt_algo", rand.suggest)
    max_evals = estimation_settings.get("hyperopt_max_evals", 500)

    if str(model.expr) == "C0*exp(C1*n)":
        estimation_settings["timeout"] = estimation_settings["timeout_privilege"]
        max_evals = max_evals*10
        print("This model is privileged.")

    best = fmin(
        fn=objective, 
        space=space, 
        algo=algo,
        timeout=estimation_settings["timeout"], 
        max_evals=max_evals,
        )
    params = list(best.values())
    ###################################################
    #### TODO: REPAIR LINE BELOW!!! DO NOT EVALUATE OBJ. FUNCTION AGAIN!!!
    result = {"x":params, "fun":objective(params)}
    #### TODO: REPAIR LINE above!!! DO NOT EVALUATE OBJ. FUNCTION AGAIN!!!
    ###################################################
    return result

def DE_fit (model, X, Y, T, p0, **estimation_settings):
    """Calls scipy.optimize.differential_evolution. 
    Exists to make passing arguments to the objective function easier."""
    
    lu_bounds = estimation_settings["lower_upper_bounds"]
    lower_bound, upper_bound = lu_bounds[0]+1e-30, lu_bounds[1]+1e-30
    # lower_bound, upper_bound = (estimation_settings["lower_upper_bounds"][i]+1e-30 for i in (0, 1))
    bounds = [[lower_bound, upper_bound] for i in range(len(p0))]

    start = time.perf_counter()
    def diff_evol_timeout(x=0, convergence=0):
        now = time.perf_counter()
        if (now-start) > estimation_settings["timeout"]:
            print("Time out!!!")
            return True
        else:
            return False
    
    if estimation_settings["verbosity"] >= 4:
        print("inside DE_fit")
    return differential_evolution(
        estimation_settings["objective_function"],
        bounds,
        args=[model, X, Y, T, estimation_settings],
        callback=diff_evol_timeout, 
        maxiter=10**2,
        popsize=10,  # orig
        # tol=10
        )

def DE_fit_metamodel (model, X, Y, T, p0, **estimation_settings):
    """DE with additional metamodel embedded."""
    
    lower_bound, upper_bound = (estimation_settings["lower_upper_bounds"][i]+1e-30 for i in (0, 1))
    bounds = [[lower_bound, upper_bound] for i in range(len(p0))]

    metamodel_kwargs = {"seed": 0}
    model_kwargs = {"dimension": len(p0),
                    "function": estimation_settings["objective_function"]}
    surrogate_kwargs = {"rebuild_interval": 100,
    # surrogate_kwargs = {"rebuild_interval": 10,
                        "predictor": ensemble.RandomForestRegressor()}
    threshold_kwargs = {"type": "alpha-beta",
                        "desired_surr_rate": 0.7,
                        "acceptable_offset": 0.05,
                        "step": 0.0001,
                        "alpha": 42,
                        "beta": 10}
    relevator_kwargs = {"rebuild_interval": 100,
    # relevator_kwargs = {"rebuild_interval": 10,
                        "threshold_kwargs": threshold_kwargs,
                        "fresh_info": None,
                        "predictor": ensemble.RandomForestRegressor()}
    history_kwargs = {"size": 500,
    # history_kwargs = {"size": 50,
                    "use_size": 200}
                    # "use_size": 20}
    metamodel = Metamodel(metamodel_kwargs, model_kwargs, surrogate_kwargs,
                          relevator_kwargs, history_kwargs)
    start = time.perf_counter()
    def diff_evol_timeout(x=0, convergence=0):
        now = time.perf_counter()
        if (now-start) > estimation_settings["timeout"]:
            print("Time out!!!")
            return True
        else:
            return False
    
    if estimation_settings["verbosity"] >= 4:
        print("inside DE_fit_metamodel")
    return differential_evolution(
        metamodel.evaluate,
        bounds,
        args=[model, X, Y, T, estimation_settings],
        callback=diff_evol_timeout, 
        maxiter=10**2,
        popsize=10,
        )

def min_fit (model, X, Y):
    """Calls scipy.optimize.minimize. Exists to make passing arguments to the objective function easier."""
    
    return minimize(optimization_wrapper, model.params, args = (model, X, Y))

def integer_brute_fit (model, X, Y, _T, p0, **estimation_settings):
    """Find minimum via brute force."""
    lower_bound, upper_bound = (estimation_settings["lower_upper_bounds"][i]+1e-30 for i in (0, 1))

    ranges = tuple(slice(lower_bound, upper_bound, 1) for i in range(len(p0)))
    res =  brute(
        func=estimation_settings["objective_function"],  # = model_oeis_error
        ranges=ranges,
        args=(model, X, Y, _T, estimation_settings),
        full_output=True,
        finish=None,
        )
    return {"x": res[0], "fun": res[1]} if len(p0) >= 2 else {
        "x": np.array([res[0]]), "fun": res[1]}

def DAnnealing_fit (model, X, Y, _T, p0, **estimation_settings):
    """Find minimum via Dual Annealing algorithm."""
    lower_bound, upper_bound = (estimation_settings["lower_upper_bounds"][i]+1e-30 for i in (0, 1))

    bounds = [(lower_bound, upper_bound)]*len(p0)
    res = dual_annealing (
        estimation_settings["objective_function"],  # = model_oeis_error
        bounds=bounds,
        args=(model, X, Y, _T, estimation_settings),
        no_local_search=True,
        # maxiter=2*10**3,
        # maxiter=4*10**3,
        )
    return res

def shgo_fit (model, X, Y, _T, p0, **estimation_settings):
    """Find minimum via shgo algorithm."""
    lower_bound, upper_bound = (estimation_settings["lower_upper_bounds"][i]+1e-30 for i in (0, 1))

    bounds = [(lower_bound, upper_bound)]*len(p0)
    res =  shgo(
        estimation_settings["objective_function"],  # = model_oeis_error
        bounds=bounds,
        args=(model, X, Y, _T, estimation_settings),
        iters=5,
        )
    return res

def find_parameters (model, X, Y, T, **estimation_settings):
    """Calls the appropriate fitting function. 
    
    TODO: 
        add method name input, matching to a dictionary of fitting methods.
    """
#    try:
#        popt, pcov = curve_fit(model.evaluate, X, Y, p0=model.params, check_finite=True)
#    except RuntimeError:
#        popt, pcov = model.params, 0
#    opt_params = popt; othr = pcov

    task_type = estimation_settings["task_type"]
    if task_type == "algebraic":
        estimation_settings["objective_function"] = model_error
    elif task_type == "differential":
        estimation_settings["objective_function"] = model_ode_error
#     elif task_type == "differential_surrogate":
#         estimation_settings["objective_function"] = meta_model_ode_error
    elif task_type == "oeis":
        # model.params = np.round(model.params)
        estimation_settings["objective_function"] = model_oeis_error
    elif task_type == "oeis_recursive_error":
        estimation_settings["objective_function"] = model_oeis_recursive_error
    else:
        types_string = "\", \"".join(TASK_TYPES)
        raise ValueError("Variable task_type has unsupported value "
                f"\"{task_type}\", while list of possible values: "
                f"\"{types_string}\".")

    # metamodel_kwargs = {"random_seed": 28537}
    # model_kwargs = {"dimension": 1,
    #                 "function": lambda x: 0}
    # surrogate_kwargs = {"rebuild_interval": 100,
    #                     "predictor": ensemble.RandomForestRegressor()}
    # threshold_kwargs = {"type": "alpha-beta",
    #                     "desired_surr_rate": 0.5,
    #                     "acceptable_offset": 0.05,
    #                     "step": 0.0001,
    #                     "alpha": 42,
    #                     "beta": 10}
    # relevator_kwargs = {"rebuild_interval": 100,
    #                     "threshold_kwargs": threshold_kwargs,
    #                     "fresh_info": None,
    #                     "predictor": ensemble.RandomForestRegressor()}
    # history_kwargs = {"size": 500,
    #                 "use_size": 200}
    # if surrogately:
    #     res = estimation_settings["optimizer"](
    #         metamodel.evaluate, X, Y, T, p0=model.params, **estimation_settings)
    # else:

    res = estimation_settings["optimizer"](
        model, X, Y, T, p0=model.params, **estimation_settings)
    # res = DE_fit(model, X, Y, T, p0=model.params, **estimation_settings)


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
        Input:
            estimation_settings: Dictionary with multiple parameters
                that determine estimation process more specifically.
    """
    def __init__(self, data, target_variable_index, time_index, estimation_settings):
        #data = np.atleast_2d(data)
        var_mask = np.ones(data.shape[-1], bool)
        var_mask[target_variable_index] = False
        if estimation_settings["task_type"] == "differential":
            var_mask[time_index] = False
            self.T = data[:, time_index]
        else:
            self.T = None
            
        self.X = data[:, var_mask]
        self.Y = data[:, target_variable_index]
        self.estimation_settings = estimation_settings
        
    def fit_one (self, model):
        if self.estimation_settings["verbosity"] > 0:
            print("Estimating model " + str(model.expr))
        try:
            if len(model.params) > 5:
                pass
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun":model_error_general(
                    [], model, self.X, self.Y, self.T,
                    **self.estimation_settings)})
            else:
                res = find_parameters(model, self.X, self.Y, self.T,
                                     **self.estimation_settings)
                model.set_estimated(res)
                # if self.estimation_settings["verbosity"] >= 2:
                #     print(res, type(res["x"]), type(res["x"][0]))
        except Exception as error:
            print((f"Excepted an error: Of type {type(error)} and message:"
                    f"{error}!! \nModel:"), model)
            model.set_estimated({}, valid=False)

        if self.estimation_settings["verbosity"] > 0:
            print(f"model: {str(model.get_full_expr()):<70}; "
                    + f"p: {model.p:<23}; "
                    + f"error: {model.get_error()}")

        return model
    
def fit_models (
    models, 
    data, 
    target_variable_index, 
    time_index=None, 
    pool_map=map, 
    verbosity=0,
    task_type="algebraic",
    estimation_settings={}):
    """Performs parameter estimation on given models. Main interface to the module.
    
    Supports parallelization by passing it a pooled map callable.
    
    Arguments:
        models (ModelBox): Instance of ModelBox, containing the models to be fitted. 
        data (numpy.array): Input data of shape N x M, where N is the number of samples 
            and M is the number of variables.
        target_variable_index (int): Index of column in data that belongs to the target variable.
        time_index (int): Index of column in data that belongs to measurement of time. 
                Required for differential equations, None otherwise.
        pool_map (function): Map function for parallelization. Example use with 8 workers:
                from multiprocessing import Pool
                pool = Pool(8)
                fit_models (models, data, -1, pool_map = pool.map)
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
        task_type (str): Type of equations, e.g. "algebraic" or "differential", that
            equation discovery algorithm tries to discover.
        estimation_settings (dict): Dictionary where majority of optional arguments is stored
                and where additional optional arguments can be passed to lower level parts of 
                equation discovery.
            arguments to pass via estimation_settings dictionary:
                timeout (float): Maximum time in seconds consumed for whole 
                    minimization optimization process, e.g. for differential evolution, that 
                    is performed for each model.
                lower_upper_bounds (tuple[float]): Pair, i.e. tuple of lower and upper
                    bound used to specify the boundaries of optimization, e.g. of 
                    differential evolution.
                max_ode_steps (int): Maximum number of steps used in one run of LSODA solver.
    """
    estimation_settings_preset = {
        "task_type": task_type,
        "verbosity": verbosity,
        "timeout": np.inf,
        "lower_upper_bounds": (-30,30),
        "optimizer": DE_fit,}
    estimation_settings_preset.update(estimation_settings)
    estimation_settings = estimation_settings_preset
    estimator = ParameterEstimator(data, target_variable_index, time_index, estimation_settings)
    
    return ModelBox(dict(zip(models.keys(), list(pool_map(estimator.fit_one, models.values())))))



if __name__ == "__main__":
    print("--- parameter_estimation.py test --- ")
    np.random.seed(2)
    
    from pyDOE import lhs
    from generators.grammar import GeneratorGrammar
    from generate import generate_models

    def testf (x):
        return 3*x[:,0]*x[:,1]**2 + 0.5
    
    X = lhs(2, 10)*5
    X = X.reshape(-1, 2)
    y = testf(X).reshape(-1,1)
    data = np.hstack((X,y))
    
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10
    
    models = generate_models(grammar, symbols, strategy_settings = {"N":10})
    
    models = fit_models(models, data, target_variable_index=-1, task_type="algebraic")
    print(models)

