import numpy as np
# import matplotlib.pyplot as plt

from ProGED.bayesian_search import BayesianSearch
from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221, tokens_to_tree
from ProGED.generators.grammar import GeneratorGrammar

eq_number = 8
# equations_path = "nguyen_expressions.txt"
equations_path = None
save_expressions_path = "nguyen_expressions.txt"
# param_path = "./parameters/nguyen1.pt"
param_path = None

universal_symbols = [{"symbol": 'X', "type": SymType.Var, "precedence": 5},
                     {"symbol": '^2', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^3', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^4', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^5', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^6', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^7', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^8', "type": SymType.Fun, "precedence": -1},
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
V -> 'X' [1.0]
P -> '^2' [0.29106029]
P -> '^3' [0.19404019]
P -> '^4' [0.14553015]
P -> '^5' [0.11642412]
P -> '^6' [0.0970201]
P -> '^7' [0.08316008]
P -> '^8' [0.07276507]
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
    # Read/Generate training expressions
    if equations_path is None:
        train_expressions = generate_train_expressions(grammar, num_expressions=30000)
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
    train, test = read_eq_data(eq_number)

    # Train/Load the model
    if param_path is None:
        generator = GeneratorHVAE.train_and_init(train_expressions, ["X"], universal_symbols, epochs=20,
                                                 hidden_size=64, representation_size=64,
                                                 model_path="./parameters/nguyen1.pt")
    else:
        generator = GeneratorHVAE(param_path, ["X"], universal_symbols)

    # Run Bayesian search and print out the results
    bs = BayesianSearch(generator=generator, initial_samples=1000)
    x, y, best_x, best_y = bs.search(train, iterations=99, eqs_per_iter=1000)
    plt.plot(best_y)
    plt.show()
    print(best_y)
    # Save results
