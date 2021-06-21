

import os
import pandas as pd
import numpy as np
from ProGED.parameter_estimation import fit_models
from ProGED.parameter_estimation import model_error
from ProGED.generate import generate_models
from ProGED.generators.grammar import GeneratorGrammar
from sklearn.metrics import mean_squared_error

np.random.seed(0)

data_file = "data.csv"
df = pd.read_csv(data_file)
print(df)
print('this was df')

def generate_data(examples):
    """
    Generated s = 1/2 a t**2 + v0 t
    :param examples:
    :return:
    """
    np.random.seed(1234)
    d = {"s": [], "a": [], "t": [], "v0": []}
    for i in range(examples):
        t, a, v0 = 2 * np.random.rand(3)
        s = 0.5 * a * t ** 2 + v0 * t
        d["s"].append(s)
        d["a"].append(a)
        d["v0"].append(v0)
        d["t"].append(t)
    df = pd.DataFrame(d)
    df.to_csv(data_file, sep=",", index=None)


if not os.path.exists(data_file):
    # generate_data(1000)
    generate_data(100)


def load_data(path, target_var):
    df = pd.read_csv(path)
    columns = list(df.columns)
    print(columns)
    i_target = columns.index(target_var)
    # print(df, 'df load_data')
    # print(np.array(df), i_target)
    return np.array(df), i_target
load_data(data_file, 's')
# print(load_data(data_file, 's'))
print('load_data')


# data = np.eye(4)
def solve_with_grammar(data_path, target_var, grammar_str, symbols, n_models):
    data, i_target = load_data(data_path, target_var)
    print(data.shape, 'data.shape')
    # data = data[:3,]
    # data = data[:1,]
    # data = np.eye(4)
    # data = np.diag((1,2,3,4))
    # data = np.arange(16).reshape(-1,4)
    print(data.shape, 'data.shape')

    X = data[:, 1:]
    Y = data[:, 0]
    # print(X, Y, 'X', 'Y')

    grammar = GeneratorGrammar(grammar_str)
    models = generate_models(grammar, symbols, strategy_settings={"N": n_models}, verbosity=1)
    models = fit_models(models, data, i_target)
    print("##################### Results:")
    # print(data, 'data fit')
    print(data[:6, :], 'data fit')
    # triplets = [(m.get_error(), m.p, m.params, str(m.get_full_expr()), mod) for m in models]
    triplets = [(m.get_error(), m.p, m.params, str(m.get_full_expr())) for m in models]
    triplets.sort()
    # for e, p, mparams, m, mod in triplets:
    for e, p, mparams, m in triplets:
        print(f"error(model): {e:.4e}, m.params: {mparams}, p(model): {p:.4e}, model: {m}")
        # print(f"error(model): {e:.4e}, m.params: {mparams}, p(model): {p:.4e}, model: {m}, model_error: {model_error(mparams, mod, data, X, Y, estimation_settings={ 'task_type': 'algebraic', 'verbosity': 2, 'timeout': np.inf, 'lower_upper_bounds': (-30,30), 'optimizer': 'differential_evolution', 'default_error': 10**9, } )}")


g2 = """S -> Em [1]
       Em -> Em '+' Fm [0.6] | Fm [0.4]
       Fm -> Tm [0.5] | 'C' '*' Tm [0.5]
       Tm -> 'v0' '*' 't' [0.5] | 'a' '*' 't' '*' 't' [0.5]"""


# s = {"x": ['v0', 'a', 't'], "start": "S", "const": "C"}
s = {"x": ['a', 't', 'v0'], "start": "S", "const": "C"}


# solve_with_grammar(data_file, "s", g2, s, 20)
# solve_with_grammar(data_file, "s", g2, s, 5)
solve_with_grammar(data_file, "s", g2, s, 2)

# error(model): 8.5642e-01, p(model): 3.3750e-04, model: 0.791088537337644*a*t**2 + 0.331703218947017*t*v0
# error(model): 1.2748e+00, p(model): 1.5000e-02, model: 0.472149023309932*a*t**2 + t*v0

data = np.loadtxt(data_file, delimiter=',', skiprows=1)
data = np.loadtxt(data_file, delimiter=',', skiprows=1)[:3,]
data = np.loadtxt(data_file, delimiter=',', skiprows=1)[:1,]
# data = np.eye(4)
# data = np.diag((1,2,3,4))
# data = np.arange(16).reshape(-1,4)
s = data[:, 0]
a = data[:, 1]
t = data[:, 2]
v0 = data[:, 3]
# print('data', data.shape)
print(data, 'data test')
# 1/0
# 0.791088542282186*a*t**2 + 0.331703201877482*t*v0

# model 1
c1_1 = 0.791088537337644
c1_2 = 0.331703218947017

# # model 2
c2_1 = 0.472149023309932

# # model 3 
c3_1 = 3.06203065913309
c3_2 = -2.19294135693102

# # # model 4 
# c3_1 = -0.716273364866414
# c3_2 = 1.04119139269982

# # # model 5 
# c3_1 = -10.9193097754316
# c3_2 = 3.33455303535458

# 3.06203065913309*a*t**2 - 2.19294135693102*t*v0
# -0.716273364866414*a*t**2 + 1.04119139269982*t*v0
# -10.9193097754316*a*t**2 + 3.33455303535458*t*v0
c3_1 = 0.00424077540807095
s_hat_3 = 3_1*a*t**2

# # model 1
# c1_1 = 0.791088537337644
# c1_2 = 0.331703218947017

# model 1
c1_1 = 0.5
c1_2 = 1.0

s_hat_1 = c1_1 * a * t * t + c1_2 * t * v0
s_hat_2 = c2_1 * a * t * t + t * v0
# s_hat_3 = c3_1 * a * t * t + c3_2 * t * v0

# print('s', s)
# print('s_hat_1', s_hat_1)
# print('diff', s-s_hat_1)
# print('quad', (s-s_hat_1)**2)
# print('mean', np.mean((s-s_hat_1)**2))
# print('model_error', model_error())

print("MSE1", mean_squared_error(s, s_hat_1))
print("MSE2", mean_squared_error(s, s_hat_2))
print("MSE3", mean_squared_error(s, s_hat_3))

# MSE1 0.3561383095100407
# MSE2 0.0032075002430028625
