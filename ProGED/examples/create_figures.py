import os, sys

my_lib_path = os.path.abspath('../../')
sys.path.append(my_lib_path)

from itertools import chain
import json
from glob import glob
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import pandas as pd
from ProGED import Model
from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221, tokens_to_tree

import signal

def read_test_data(eq_number):
    test = []
    with open(f"data/nguyen/nguyen{eq_number}_corrected_test.csv", "r") as file:
        file.readline()
        for row in file:
            line = [float(t) for t in row.strip().split(",")]
            test.append(line)
    return np.array(test)

def eval_tree(tree, data, symbols):
    l = None
    r = None
    if tree.left is not None:
        l = eval_tree(tree.left, data, symbols)
    if tree.right is not None:
        r = eval_tree(tree.right, data, symbols)
    return symbols[tree.symbol]["fun"](l, r, data)

universal_symbols = [{"symbol": 'X', "type": SymType.Var, "precedence": 5, "fun": lambda l, r, d: d[:, 0]},
                     # {"symbol": 'Y', "type": SymType.Var, "precedence": 5, "fun": lambda l, r, d: d[:, 1]},
                     {"symbol": '^2', "type": SymType.Fun, "precedence": -1, "fun": lambda l, r, d: np.power(l, 2)},
                     {"symbol": '^3', "type": SymType.Fun, "precedence": -1, "fun": lambda l, r, d: np.power(l, 3)},
                     {"symbol": '^4', "type": SymType.Fun, "precedence": -1, "fun": lambda l, r, d: np.power(l, 4)},
                     {"symbol": '^5', "type": SymType.Fun, "precedence": -1, "fun": lambda l, r, d: np.power(l, 5)},
                     # {"symbol": '^6', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^7', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^8', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '+', "type": SymType.Operator, "precedence": 0, "fun": lambda l, r, d: l + r},
                     {"symbol": '-', "type": SymType.Operator, "precedence": 0, "fun": lambda l, r, d: l - r},
                     {"symbol": '*', "type": SymType.Operator, "precedence": 1, "fun": lambda l, r, d: l * r},
                     {"symbol": '/', "type": SymType.Operator, "precedence": 1, "fun": lambda l, r, d: l / r},
                     {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5, "fun": lambda l, r, d: np.sqrt(l)},
                     {"symbol": 'sin', "type": SymType.Fun, "precedence": 5, "fun": lambda l, r, d: np.sin(l)},
                     {"symbol": 'cos', "type": SymType.Fun, "precedence": 5, "fun": lambda l, r, d: np.cos(l)},
                     {"symbol": 'exp', "type": SymType.Fun, "precedence": 5, "fun": lambda l, r, d: np.exp(l)},
                     {"symbol": 'log', "type": SymType.Fun, "precedence": 5, "fun": lambda l, r, d: np.log(r)}]
s_for_tokenization = {t["symbol"]: t for i, t in enumerate(universal_symbols)}


def tokenize_expr(expr):
    common_functions = ["cos", "sin", "exp", "sqrt", "log", "^2", "^3", "^4", "^5"]
    for fn in common_functions:
        expr = expr.replace(fn, f" {fn} ")
    first_tokens = expr.split(" ")
    tokens = []
    for t in first_tokens:
        if t in common_functions:
            tokens += [t]
        else:
            tokens += [c for c in t]
    return tokens


def read_json_data(jo, test, file, baseline):
    changed = False
    exprs = []
    real_values = test[:, -1]
    bottom = np.sum((real_values - np.mean(real_values)) ** 2)
    name = baseline + "_" + file.split("/")[-1]

    index_invalid = None
    for e in jo:
        e["valid"] = True
        if "test" not in e or "r2" not in e:
            changed = True
            try:
                # e["eq"] = e["eq"].replace("**", "^")
                tokens = tokenize_expr(e["eq"])
                tree = tokens_to_tree(tokens, s_for_tokenization)
                predicted_values = eval_tree(tree, test, s_for_tokenization)
                # expr = Model(e["eq"], sym_vars=(["X"] if test.shape[1] == 2 else ["X", "Y"]))
                # predicted_values = expr.evaluate(test[:, :-1])
                if any(np.iscomplex(predicted_values)):
                    predicted_values = predicted_values.real()
                real_values = test[:, -1]
                se = np.sum((predicted_values - real_values) ** 2)
                if not np.isfinite(se):
                    se = (10e10*real_values.shape[0])**2
                rmse = np.sqrt(se/real_values.shape[0])
                r2 = max(1 - (se / bottom), 0)
            except:
                rmse = 1e10
                e["valid"] = False
                r2 = 0
            finally:
                if "valid" in e and not e["valid"]:
                    if index_invalid is None:
                        index_invalid = len(exprs)
                        exprs.append(["", float(1e10), float(1e10), 0.0, int(e["trees"])])
                    else:
                        exprs[index_invalid][-1] += e["trees"]
                else:
                    exprs.append([e['eq'], float(e['error']), rmse, r2, int(e["trees"])])
        else:
            exprs.append([e['eq'], float(e['error']), float(e["test"]), float(e["r2"]), int(e["trees"])])


    if changed:
        with open(f"results/others/{name}", "w") as f:
            json.dump(exprs, f)
    return exprs


# new.append(mutation_scale * X[i] + np.random.random(X.shape[1])*std)
def create_sequential_results(exprs, step=50):
    repeated_steps = []
    success = False
    invalid = 0
    best_ind, best_val = 0, exprs[0][1]
    for k in range(1, len(exprs)):
        if exprs[k][0] == '':
            invalid += 1
        if exprs[k][2] < best_val:
            best_ind, best_val = k, exprs[k][2]

            if best_val < 5e-7:
                success = True
                break
        else:
            exprs[k] = exprs[best_ind]

    steps = []
    i = 0
    while i < k:
        steps.append(exprs[i])
        i += step
    if success:
        steps.append(exprs[k])
    repeated_steps.append(steps)
    return repeated_steps, [(exprs[k][0], k, success, exprs[k][3], invalid)]


def create_random_results(exprs_start, step=50, repeats=1):
    expressions = []
    invalid = 0
    for e in exprs_start:
        for i in range(e[4]):
            expressions.append(e)
    expressions += [("inf", float(1000000000.0), float(1000000000.0), float(0), int(1)) for i in range(100000-len(expressions))]

    repeated_steps, other = [], []
    for j in range(repeats):
        success = False
        exprs = np.random.permutation(expressions)
        seen_exprs = set()
        new_exprs = []
        for e in exprs:
            if e[0] == '':
                invalid += 1
            if e[0] not in seen_exprs:
                new_exprs.append(e)
                seen_exprs.add(e[0])
        exprs = new_exprs
        best_ind, best_val = 0, float(exprs[0][2])
        for k in range(1, len(exprs)):
            if float(exprs[k][2]) < best_val:
                best_ind, best_val = k, float(exprs[k][2])

                if best_val < 1e-8:
                    success = True
                    break
            else:
                exprs[k] = exprs[best_ind]

        steps = []
        i = 0
        while i < k:
            steps.append(exprs[i])
            i += step
        if success:
            steps.append(exprs[k])
        repeated_steps.append(steps)
        other.append((exprs[k][0], k, success, float(exprs[k][3]), invalid))
    return repeated_steps, other


def create_steps_figure(eq_num, results, name, step=50):
    x = []
    y = []
    color = []
    for col, res in results:
        for arr in res:
            for j, e in enumerate(arr):
                x.append(int(j)*step)
                y.append(float(e[3]))
                color.append(col)


    plot = sns.lineplot(data=pd.DataFrame(data={"Epoch": x, "r^2": y, "Baseline": color}), x="Epoch", y="r^2", hue="Baseline")
    plot.set_ylim(-0.1, 1.1)
    # plot.set(yscale="log")
    plt.title(f'Nguyen {eq_num}', fontsize=20)
    plt.legend(title='', fontsize=18)
    plt.xlabel("Unique expressions tested", fontsize=18)
    plt.ylabel("R^2", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()


if __name__ == '__main__':
    eq_num = 10
    # baselines = ["HVAE Evo", "HVAE Random"]
    # baselines = ["HVAE", "HVAE Evo", "ProGED"]
    # baselines = ["HVAE16", "HVAE32", "HVAE64", "HVAE128"]
    baselines = ["HVAE", "ProGED", "HVAE,Evo"]
    name = {"HVAE": "HVAR", "ProGED": "ProGED", "HVAE,Evo": "EDHiE"}
    # baselines = ["CVAE32", "CVAE64", "CVAE128", "GVAE32", "GVAE64", "GVAE128", "GVAE,Evo"]

    paths = ["/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/cvae_32",
             "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/cvae_64",
             "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/cvae_128",
             "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/gvae_32",
             "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/gvae_64",
             "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/gvae_128",
             "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/gvae_evo"]
    # paths = ["/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/hvae_random_32",
    #          "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/ProGED_corrected",
    #          "/home/sebastianmeznar/Projects/ProGED/ProGED/examples/results/hvae_evo"]
    for eq_num in [eq_num]:
        test = read_test_data(eq_num)
        steps = []
        other = []
        # for baseline, path in zip(baselines, paths):
        for baseline in baselines:
            files = glob(f"results/ready/{baseline}_nguyen_{eq_num}_*")
            # files = glob(path+f"/nguyen_{eq_num}_*")
            for file in files:
                exists = True
                # name = f"{baseline}_{file.split('/')[-1]}"
                # if os.path.exists(f"results/dims/{name}"):
                # if os.path.exists(f"results/others/{name}"):
                # file = f"results/dims/{name}"
                # file = f"results/others/{name}"
                # exists = True
                with open(file, 'r') as f:
                    if exists:
                        exprs = json.load(f)
                    else:
                        print(file)
                        jo = json.load(f)
                        exprs = read_json_data(jo, test, file, baseline)
                    if baseline in ["HVAE,Evo", "GVAE,Evo"]:
                        a, b = create_sequential_results(exprs)
                    else:
                        a, b = create_random_results(exprs)
                    steps.append((name[baseline], a))
                    other.append((name[baseline], b))
                    # successes.append((baseline, c))

        for baseline in baselines:
            max_len = max([len(l) for l in list(chain(*[arr[1] for arr in steps if arr[0] == name[baseline]]))])
            for i in range(len(steps)):
                if steps[i][0] == name[baseline]:
                    for j in range(len(steps[i][1])):
                        steps[i][1][j] += [steps[i][1][j][-1] for _ in range(max_len-len(steps[i][1][j]))]

        print(f"Nguyen {eq_num}")
        for baseline in baselines:
            best_exprs = []
            tested = []
            successful = 0
            r2 = []
            invalid = []
            total = 0
            for bl in other:
                if bl[0] == name[baseline]:
                    for (e, k, s, r, i) in bl[1]:
                        best_exprs.append(e)
                        tested.append((k, s))
                        invalid.append(i)
                        r2.append(r)
                        if s:
                            successful += 1
                        total += 1
            print(f"{name[baseline]}: {successful}/{total}={(successful*100)/total}%")
            print("Recreated expressions: ", end="")
            print(best_exprs)
            print(f"Tested expressions on success: {np.mean([t for (t, s) in tested if s])} (\pm {np.std([t for (t, s) in tested if s])})")
            print(f"Tested expressions: {np.mean([t for (t, s) in tested])} (\pm {np.std([t for (t, s) in tested])})")
            print(f"R^2: {np.mean(r2)} (\pm {np.std(r2)})")
            print(f"Invalid: {np.mean(invalid)} (\pm {np.std(invalid)})")
            print("")
            print("-----------------------------------------------")
            print("")

        create_steps_figure(eq_num, steps, name)
        # print("")
        # print("Average num of tested expressions:")
        # for baseline in baselines:
        #     k = []
        #     for r in ks:
        #         if r[0] == baseline:
        #             for val in r[1]:
        #                 k.append(val)
        #     print(f"{baseline.replace(',', ' ')}:")
        #     print(f"\tAverage: {np.mean(k)}")
        #     print(f"\tAll: {','.join([str(v) for v in k])}")

        # name = baseline + "_" + file.split("/")[-1]
        # print(f"{name}\t{exists(f'results/ready/{name}')}")
        # if not exists(f"results/ready/{name}"):
        #     jo = None
        #     with open(file, 'r') as f:
        #         jo = json.load(f)
        #
        #     exprs = read_json_data(jo, test, file, baseline)

# c32 : 1 , 0 , 0 , 0 , 0 , 0 , 0 , 7
# c64 : 2 , 0 , 0 , 0 , 0 , 0 , 0 , 8
# c128: 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0
# g32 : 10, 1 , 0 , 0 , 0 , 1 , 0 , 10
# g64 : 10, 4 , 0 , 0 , 0 , 0 , 0 , 10
# g128: 10, 2 , 0 , 0 , 0 , 0 , 0 , 10
# gevo: 10, 3 , 1 , 0 , 0 , 0 , 0 , 10
# DSO : 10, 10, 10, 8 , 0 , 1 , 10, 10, 2 , 0
#NG-eq: 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10

#gvae64 2,3