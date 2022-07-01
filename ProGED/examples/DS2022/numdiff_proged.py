import numpy as np
import ProGED as pg
import time

from utils.generate_data_ODE_systems import generate_ODE_data, lorenz
from ProGED.generators.grammar_construction import construct_production

if __name__ == "__main__":
    np.random.seed(0)

    """  SETTINGS """
    inits = [1,1,1]
    rho = 16
    gram = "polylim"
    N = 500
    save_models = True
    name = "numdiff_lorenz_polylim"

    """ DATA """
    lor = lambda t, x: lorenz(t, x, rho=rho)
    data = generate_ODE_data(lor, inits)
    T = data[:,0]
    deltaT = T[1]-T[0]
    X = data[:, 1:]
    """ NUMERIC DIFFERENTIATION """
    dX = np.array([np.gradient(Xi, deltaT) for Xi in X.T])

    """ TRUE MODEL """
    symbols = {"x": ["x","y","z"], "const":"C"}
    optimal_model = ["C*(y-x)", "x*(C-z)-y", "x*y-C*z"]
    optimal_model = ["C*x + C*y", "C*x + C*y + C*x*z", "C*x*y + C*z"]

    """ GRAMMAR """
    
    if gram == "poly":
        grammar_settings = {"variables": ["'x'", "'y'", "'z'"], 
                            "p_vars": [1/3, 1/3, 1/3], 
                            "functions": [], "p_F": []}
        grammar = pg.grammar_from_template("polynomial", grammar_settings)
    elif gram == "polylim":
        grammarstr = construct_production("P", ["P '+' M", "M"], [0.3, 0.7])
        grammarstr += construct_production("M", ["M '*' V", "'C' '*' V", "V"], [0.3, 0.3, 0.4])
        grammarstr += construct_production("V", ["'x'", "'y'", "'z'"], [1/3, 1/3, 1/3])
        grammar = pg.GeneratorGrammar(grammarstr)
    elif gram == "custom":
        grammarstr = construct_production("E", ["E '+' F", "E '-' F", "F"], [0.15, 0.15, 0.7])
        grammarstr += construct_production("F", ["F '*' T", "T"], [0.2, 0.8])
        grammarstr += construct_production("T", ["'(' E ')'", "V", "'C'"], [0.2, 0.5, 0.3])
        grammarstr += construct_production("V", ["'x'", "'y'", "'z'"], [1/3, 1/3, 1/3])

        grammar = pg.GeneratorGrammar(grammarstr)

    
    optimizer_settings = {"lower_upper_bounds": (-30,30),
                            "atol": 0.01, "tol": 0.01,
                            "max_iter": 500,
                            "pop_size": 20}
    estimation_settings = {"target_variable_index": 0,
                            "verbosity": 1,
                            "optimizer_settings": optimizer_settings,
                            "max_constants": 3}

    """ ITERATE THROUGH THE THREE DIMENSIONS """
    t1 = time.time()
    for i in range(3):
        """ SETUP THE MODELS """
        models = pg.generate.generate_models(grammar, symbols, strategy_settings={"N":N, "max_repeat":100})
        #models = pg.ModelBox()
        #models.add_model(optimal_model[i], symbols)
        print(models)

        """ FIT PARAMETERS """
        dat = np.hstack((dX[i].reshape((-1,1)), X))
        models_fit = pg.fit_models(models, dat, task_type="algebraic", 
                                    estimation_settings=estimation_settings)
        
        if save_models:
            models_fit.dump(name + "_eq" + str(i) + "_models_fit.pg")
        print("------- best models -------")
        print(models_fit.retrieve_best_models(3))
    t2 = time.time()
    print("Time: ", t2-t1)
    print(grammar)