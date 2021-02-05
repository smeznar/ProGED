import numpy as np
# pg = ProGED
import ProGED as pg
# from pg.generate import generate_models
# gft = pg.generators.grammar_from_template
# from ProGED import generators.generate_models
from ProGED.equation_discoverer import EqDisco
# from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.generate import generate_models
# from ProGED.model import Model
# from ProGED.model_box import ModelBox
# from ProGED.parameter_estimation import fit_models

oeis = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
 1597,2584,4181,6765,10946,17711,28657,46368,75025,
 121393,196418,317811,514229,832040,1346269,
 2178309,3524578,5702887,9227465,14930352,24157817,
 39088169,63245986,102334155]
fibs = np.array(oeis).reshape(-1, 1)
ts = np.array([i for i in range(40+1)]).reshape(-1, 1)
# print(ts, type(fibs), fibs.shape, type(ts[0]), ts.shape)
data = np.hstack((ts, fibs))

# grammar:
# from pg.generators import grammar_constructionk
# from pg import model

# variables = ["'n'"]
# symbols = {"x":variables, "start":"S", "const":"C"}
# # p_vars = [1, 0.3, 0.4]
# p_T = [0.1, 0.9]
# p_R = [0.1, 0.9]
# grammar = grammar_from_template("polynomial", {"variables": variables, "p_R": p_R, "p_T": p_T})
# np.random.seed(0)
# # print(grammar.generate_one())
# # models = generate_models(grammar, symbols, strategy_settings = {"N":500})
# # print(models)

np.random.seed(0)
ED = EqDisco(data = data,
            task = None,
            target_variable_index = -1,
            sample_size = 20,
            verbosity = 0,
            generator = "grammar", 
            generator_template_name = "polynomial", 
            )
# print(data, data.shape)
ED.generate_models()
ED.fit_models()
print(ED.models)
# print(ED.get_results())
# print(ED.get_stats())
    



