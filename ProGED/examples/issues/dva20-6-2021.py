import os
import pandas as pd
import numpy as np
from ProGED.parameter_estimation import fit_models
from ProGED.generate import generate_models
from ProGED.generators.grammar import GeneratorGrammar
from sklearn.metrics import mean_squared_error


data_file = "data.csv"


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
    generate_data(1000)


def load_data(path, target_var):
    df = pd.read_csv(path)
    columns = list(df.columns)
    i_target = columns.index(target_var)
    return np.array(df), i_target


def solve_with_grammar(data_path, target_var, grammar_str, symbols, n_models):
    data, i_target = load_data(data_path, target_var)
    grammar = GeneratorGrammar(grammar_str)
    models = generate_models(grammar, symbols, strategy_settings={"N": n_models}, verbosity=1)
    models = fit_models(models, data, i_target)
    print("##################### Results:")
    triplets = [(m.get_error(), m.p, str(m.get_full_expr())) for m in models]
    triplets.sort()
    for e, p, m in triplets:
        print(f"error(model): {e:.4e}, p(model): {p:.4e}, model: {m}")


g2 = """S -> Em [1]
       Em -> Em '+' Fm [0.6] | Fm [0.4]
       Fm -> Tm [0.5] | 'C' '*' Tm [0.5]
       Tm -> 'v0' '*' 't' [0.5] | 'a' '*' 't' '*' 't' [0.5]"""


s = {"x": ['v0', 'a', 't'], "start": "S", "const": "C"}


solve_with_grammar(data_file, "s", g2, s, 20)

# error(model): 8.5642e-01, p(model): 3.3750e-04, model: 0.791088537337644*a*t**2 + 0.331703218947017*t*v0
# error(model): 1.2748e+00, p(model): 1.5000e-02, model: 0.472149023309932*a*t**2 + t*v0

data = np.loadtxt(data_file, delimiter=',', skiprows=1)
s = data[:, 0]
a = data[:, 1]
t = data[:, 2]
v0 = data[:, 3]

# model 1
c1_1 = 0.791088537337644
c1_2 = 0.331703218947017

# model 2
c2_1 = 0.472149023309932

s_hat_1 = c1_1 * a * t * t + c1_2 * t * v0
s_hat_2 = c2_1 * a * t * t + t * v0

print("MSE1", mean_squared_error(s, s_hat_1))
print("MSE2", mean_squared_error(s, s_hat_2))

# MSE1 0.3561383095100407
# MSE2 0.0032075002430028625
