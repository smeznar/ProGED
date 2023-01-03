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


def read_test_data(eq_number):
    test = []
    with open(f"data/nguyen/nguyen{eq_number}_corrected_test.csv", "r") as file:
        file.readline()
        for row in file:
            line = [float(t) for t in row.strip().split(",")]
            test.append(line)
    return np.array(test)


def read_json_data(jo, test, file, baseline):
    changed = False
    exprs = []
    real_values = test[:, -1]
    bottom = np.sum((real_values - np.mean(real_values)) ** 2)
    name = baseline + "_" + file.split("/")[-1]

    for e in jo:
        if "test" not in e or "r2" not in e:
            changed = True
            expr = Model(e["eq"], sym_vars=(["X"] if test.shape[1] == 2 else ["X", "Y"]))
            try:
                predicted_values = expr.evaluate(test[:, :-1])
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
                r2 = 0
            exprs.append([e['eq'], float(e['error']), rmse, r2, int(e["trees"])])
        else:
            exprs.append([e['eq'], float(e['error']), float(e["test"]), float(e["r2"]), int(e["trees"])])

    if changed:
        with open(f"results/ready/{name}", "w") as f:
            json.dump(exprs, f)
    return exprs


# new.append(mutation_scale * X[i] + np.random.random(X.shape[1])*std)
def create_sequential_results(exprs, step=50):
    repeated_steps = []
    success = False

    best_ind, best_val = 0, exprs[0][1]
    for k in range(1, len(exprs)):
        if exprs[k][1] < best_val:
            best_ind, best_val = k, exprs[k][1]

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
    return repeated_steps, [k], [exprs[-1][2] < 1e-10]


def create_random_results(exprs_start, step=50, repeats=1):
    expressions = []
    for e in exprs_start:
        for i in range(e[4]):
            expressions.append(e)
    expressions += [("inf", float(1000000000.0), float(1000000000.0), float(0), int(1)) for i in range(100000-len(expressions))]

    repeated_steps, ks, successes = [], [], []
    for j in range(repeats):
        success = False
        exprs = np.random.permutation(expressions, )
        seen_exprs = set()
        new_exprs = []
        for e in exprs:
            if e[0] not in seen_exprs:
                new_exprs.append(e)
                seen_exprs.add(e[0])
        exprs = new_exprs
        best_ind, best_val = 0, float(exprs[0][1])
        for k in range(1, len(exprs)):
            if float(exprs[k][1]) < best_val:
                best_ind, best_val = k, float(exprs[k][1])

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
        ks.append(k)
        successes.append(exprs[k][2] < 1e-10)
    return repeated_steps, ks, successes


def create_steps_figure(eq_num, results, step=50):
    x = []
    y = []
    color = []
    for col, res in results:
        for arr in res:
            for j, e in enumerate(arr):
                x.append(int(j)*step)
                y.append(float(e[3]))
                color.append(col.replace(",", " "))


    plot = sns.lineplot(data=pd.DataFrame(data={"Epoch": x, "r^2": y, "Baseline": color}), x="Epoch", y="r^2", hue="Baseline")
    plot.set_ylim(-0.1, 1.1)
    # plot.set(yscale="log")
    plot.set(title=f'Nguyen {eq_num}')
    plt.show()


if __name__ == '__main__':
    eq_num = 9
    # baselines = ["HVAE Evo", "HVAE Random"]
    # baselines = ["HVAE", "HVAE Evo", "ProGED"]
    # # baselines = ["HVAE 16", "HVAE 32", "HVAE 64", "HVAE 128", "HVAE Evo", "ProGED"]
    baselines = ["HVAE", "ProGED", "HVAE,Evo"]
    # baselines = ["HVAE 16", "HVAE 32", "HVAE 64", "HVAE 128", "HVAE Evo", "ProGED"]
    test = read_test_data(eq_num)

    steps = []
    ks = []
    successes = []
    for baseline in baselines:
        files = glob(f"results/ready/{baseline}_nguyen_{eq_num}_*")
        for file in files:
            with open(file, 'r') as f:
                exprs = json.load(f)
                # exprs = read_json_data(jo, test, file, baseline)
                if baseline == "HVAE,Evo":
                    a, b, c = create_sequential_results(exprs)
                else:
                    a, b, c = create_random_results(exprs)

                steps.append((baseline, a))
                ks.append((baseline, b))
                successes.append((baseline, c))

    for baseline in baselines:
        max_len = max([len(l) for l in list(chain(*[arr[1] for arr in steps if arr[0] == baseline]))])
        for i in range(len(steps)):
            if steps[i][0] == baseline:
                for j in range(len(steps[i][1])):
                    steps[i][1][j] += [steps[i][1][j][-1] for _ in range(max_len-len(steps[i][1][j]))]

    print(f"Nguyen {eq_num}")
    print("Evaluation was succesfull:")
    for baseline in baselines:
        total = 0
        successful = 0
        for s in successes:
            if s[0] == baseline:
                for val in s[1]:
                    if val:
                        successful += 1
                    total += 1
        print(f"{baseline.replace(',', ' ')}: {successful}/{total}={(successful*100)/total}%")

    print("")
    print("Average num of tested expressions:")
    for baseline in baselines:
        k = []
        for r in ks:
            if r[0] == baseline:
                for val in r[1]:
                    k.append(val)
        print(f"{baseline.replace(',', ' ')}:")
        print(f"\tAverage: {np.mean(k)}")
        print(f"\tAll: {','.join([str(v) for v in k])}")

    create_steps_figure(eq_num, steps)
        # name = baseline + "_" + file.split("/")[-1]
        # print(f"{name}\t{exists(f'results/ready/{name}')}")
        # if not exists(f"results/ready/{name}"):
        #     jo = None
        #     with open(file, 'r') as f:
        #         jo = json.load(f)
        #
        #     exprs = read_json_data(jo, test, file, baseline)
