import os, sys
my_lib_path = os.path.abspath('../../')
sys.path.append(my_lib_path)

import numpy as np
import argparse

from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221, tokens_to_tree
from ProGED.generators.grammar import GeneratorGrammar

# equations_path = "nguyen_expressions.txt"
equations_path = None
save_expressions_path = "nguyen_expressions2.txt"
param_path = "./parameters/nguyen2_32.pt"
# param_path = None

universal_symbols = [{"symbol": 'X', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'Y', "type": SymType.Var, "precedence": 5},
                     {"symbol": '^2', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^3', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^4', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^5', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^6', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^7', "type": SymType.Fun, "precedence": -1},
                     # {"symbol": '^8', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '+', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '-', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '*', "type": SymType.Operator, "precedence": 1},
                     {"symbol": '/', "type": SymType.Operator, "precedence": 1},
                     {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'sin', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'cos', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'exp', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'log', "type": SymType.Fun, "precedence": 5}]

grammar = """E -> E '+' F [0.2]
E -> E '-' F [0.2]
E -> F [0.6]
F -> F '*' T [0.2]
F -> F '/' T [0.2]
F -> T [0.6]
T -> V [0.4]
T -> '(' E ')' P [0.2]
T -> '(' E ')' [0.2]
T -> R '(' E ')' [0.2]
V -> 'X' [0.5]
V -> 'Y' [0.5]
P -> '^2' [0.38961039]
P -> '^3' [0.25974026]
P -> '^4' [0.19480519]
P -> '^5' [0.15584416]
R -> 'sin' [0.2]
R -> 'cos' [0.2]
R -> 'exp' [0.2]
R -> 'log' [0.2]
R -> 'sqrt' [0.2]"""


def generate_train_expressions(cfg, num_expressions=100, max_generated=None, symbols=universal_symbols):
    if max_generated is None:
        max_generated = num_expressions*1000
    generator = GeneratorGrammar(cfg)
    tokenzation_s = {t["symbol"]: t for i, t in enumerate(symbols)}
    precedence = {t["symbol"]: t["precedence"] for t in symbols}

    generated = 0
    expressions = set()
    while len(expressions) < num_expressions and generated < max_generated:
        if len(expressions) % 1000 == 0:
            print(len(expressions))
        generated += 1
        expression = generator.generate_one()[0]
        tree = tokens_to_tree(expression, tokenzation_s)
        if tree.height() > 7:
            continue
        expressions.add(' '.join(tree.to_list(with_precedence=True, precedence=precedence)))
    return list(expressions)


def read_eq_data(eq_number):
    train = []
    with open(f"data/nguyen/nguyen{eq_number}_train.csv", "r") as file:
        file.readline()
        for row in file:
            line = [float(t) for t in row.strip().split(",")]
            train.append(line)

    test = []
    with open(f"data/nguyen/nguyen{eq_number}_test.csv", "r") as file:
        file.readline()
        for row in file:
            line = [float(t) for t in row.strip().split(",")]
            test.append(line)
    return np.array(train), np.array(test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Nguyen benchmark', description='Run a ED benchmark')
    parser.add_argument("-eq_num", choices=range(1, 10), required=True, action="store", type=int)
    args = parser.parse_args()

    # Read/Generate training expressions
    if equations_path is None:
        train_expressions = generate_train_expressions(grammar, num_expressions=35000)
        if save_expressions_path is not None:
            with open(save_expressions_path, "w") as file:
                for e in train_expressions:
                    file.write(f"{e}\n")
        train_expressions = [e.split(" ") for e in train_expressions]
    else:
        train_expressions = []
        with open(equations_path, "r") as file:
            for l in file:
                train_expressions.append(l.strip().split(" "))

    # Read data
    train, test = read_eq_data(args.eq_num)

    # Train/Load the model
    # if param_path is None:
    generator = GeneratorHVAE.train_and_init(train_expressions, ["X", "Y"], universal_symbols, epochs=20,
                                             hidden_size=32, representation_size=32,
                                             model_path="./parameters/nguyen2_32.pt")
    # else:
    #     generator = GeneratorHVAE(param_path, ["X"], universal_symbols)

    # Run Bayesian search and print out the results
    # bs = BayesianSearch(generator=generator, initial_samples=1000, default_error=1000000)
    # x, y, best_x, best_y = bs.search(train, test, iterations=99, eqs_per_iter=1000)
    # x, y, best_x, best_y = bs.random(train, test, iterations=100, eqs_per_iter=1000)

    # grammar = GeneratorGrammar(grammar)
    # symbols = {"x": ['X'], "start": "E", "const": "C"}
    # ed = EqDisco(data=train, variable_names=["X", 'Y'], generator=grammar, sample_size=100000, verbosity=0)
    # ed.generate_models()
    # ed.fit_models()
    # print(len(ed.models))
    # print(ed.get_results())
    # ed.write_results(f"results/proged/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
    # plt.plot(best_y)
    # plt.show()
    # print(best_y)
    # best_equation = sp.parse_expr("".join(generator.decode_latent(x)).replace("^", "**"))
    # predicted_values = []
    # for i in range(test.shape[0]):
    #     if test.shape[1] == 2:
    #         predicted_values.append(best_equation.subs("X", test[i, 0]))
    #     else:
    #         predicted_values.append(best_equation.subs([("X", test[i, 0]), ("Y", test[i, 1])]))
    # predicted_values = np.array(predicted_values)
    # real_values = test[:, -1]
    # average = np.mean(real_values)
    # r2 = 1 - (np.sum((predicted_values-real_values)**2)/np.sum((real_values-average)**2))
    # print(best_equation)
    # print(f"R^2: {r2}")
    # Save results
    # with open(f"results/nguyen_hvae_random_32_{args.eq_num}.txt", "a") as file:
    #     file.write(f"{r2}\t{best_equation}\t{';'.join([str(z) for z in best_y])}\n")
