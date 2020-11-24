# %%
# import matplotlib.pyplot as plt
# %%

import numpy as np
from odes import example_tB_data # import datasets T, X and Y
from parameter_estimation import (
    ode, 
    ParameterEstimator,
    fit_models,
    model_ode_error)
from generate import generate_models    
from generators.grammar import GeneratorGrammar
from model_box import ModelBox

T, Xs, Ys, _, a = example_tB_data() # n,t1,t2,B,a)
X = np.array([Ys]).T; Y = np.array([Xs]).T
print(T.shape, X.shape, Y.shape)

grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                            T -> 'C' [0.6] | T "*" V [0.4]
                            V -> 'x' [0.5] | 'y' [0.5]""")
grammar = GeneratorGrammar("""S -> 'C' '*' 'y' '+' 'x' [1]""")
grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                            T -> V [0.6] | 'C' "*" V [0.4]
                            V -> 'x' [0.5] | 'y' [0.5]""")
symbols = {"x":['y', 'x'], "start":"S", "const":"C"}
# resitev = dot{y} = a*y + x, tj. C0*y+x
np.random.seed(2)
models = generate_models(grammar, symbols, strategy_parameters = {"N":10})
# m = models[-1]
# print(type(m))
# print(m, m.params, 
# m.grammar)
# print(models)
model = [m for m in models][2]
######################## pravilno resi.      ###################
# print(a)
# odeY = ode([model], [[a]], T, X, Y[0])
# b = 0.46070049
# try:
#     odeY = ode([model], [[b]], T, X, Y[0])
# except Exception as err:
#     print(err.args)

# plt.plot(T, Y,"r-")
# plt.plot(T, odeY[0], "k--")
# print(model_ode_error(model, [a], T, X, Y))
######################## ################## ###################
print(model.params)
model.params = [a]
print(model.params, model)
# new_mb = ModelBox()
# new_mb.
estimator = ParameterEstimator(X, Y, T)
fit_model = estimator.fit_one(model)

# fitted_model = fit_models(models, X, Y, T)    
# # print([(m, m.params) for m in models])
# print(models)

