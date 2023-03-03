import os, sys

my_lib_path = os.path.abspath('../../')
other_baselines = "/home/sebastianmeznar/Downloads/other_baselines_2/other_baselines"
sys.path.append(my_lib_path)
sys.path.append(other_baselines)

import argparse
import json
import time
import random

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.termination.max_gen import MaximumGenerationTermination
from sympy.testing.pytest import ignore_warnings
from tqdm import tqdm

from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221, tokens_to_tree
from ProGED.generators.grammar import GeneratorGrammar
from ProGED import EqDisco
from ProGED import Model
from ProGED.rust_evaluator import RustEval

import equation_vae

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
V -> 'X' [0.5]
P -> '^2' [0.38961039]
P -> '^3' [0.25974026]
P -> '^4' [0.19480519]
P -> '^5' [0.15584416]
R -> 'sin' [0.2]
R -> 'cos' [0.2]
R -> 'exp' [0.2]
R -> 'log' [0.2]
R -> 'sqrt' [0.2]"""


def read_eq_data(eq_number):
    train = []
    with open(f"data/nguyen/nguyen{eq_number}_corrected_train.csv", "r") as file:
        file.readline()
        for row in file:
            line = [float(t) for t in row.strip().split(",")]
            train.append(line)

    test = []
    with open(f"data/nguyen/nguyen{eq_number}_corrected_test.csv", "r") as file:
        file.readline()
        for row in file:
            line = [float(t) for t in row.strip().split(",")]
            test.append(line)
    return np.array(train), np.array(test)


class SRProblem(ElementwiseProblem):
    def __init__(self, generator, tdata, dim, default_value=1e10):
        self.generator = generator
        self.tdata = tdata
        self.default_value = default_value
        self.input_mean = torch.zeros(next(generator.model.decoder.parameters()).size(0))
        self.models = []
        self.evaluated_models = dict()
        self.best_f = 9e+50
        self.best_expression = None
        self.evaluator = RustEval(tdata)
        super().__init__(n_var=dim, n_obj=1)

    def add_model(self, model, score):
        model_str = str(model)
        if model_str in self.evaluated_models:
            self.evaluated_models[model_str] += 1
        else:
            if isinstance(score, complex):
                score = score.real
            self.evaluated_models[model_str] = 1
            self.models.append({"eq": model_str, "error": score})

    def _evaluate(self, x, out, *args, **kwargs):
        eq, eq_pof = self.generator.decode_latent(torch.tensor(x)[None, None, :])
        s = ["X"] if self.tdata.shape[1] == 2 else ["X", "Y"]
        # model = Model("".join(eq), sym_vars=s)
        rmse_re = self.evaluator.get_error(eq_pof)
        out["F"] = rmse_re if rmse_re is not None else self.default_value
        # try:
        #     with ignore_warnings(RuntimeWarning):
        #         yp = model.evaluate(self.tdata[:, :-1])
        #         if any(np.iscomplex(yp)):
        #             yp = yp.real()
        #     rmse = np.sqrt(np.square(np.subtract(self.tdata[:, -1], yp)).mean())
        #     if np.isfinite(rmse):
        #         out["F"] = rmse
        #     else:
        #         out["F"] = self.default_value
        # except Exception:
        #     out["F"] = self.default_value
        # finally:
        #     print(f"Original: {''.join(eq)}, simplified: {str(model)}")
        #     print(f'With simplify: {out["F"]}, without simplify {rmse_re}')
        #     print("--------------------------------------------------------")
        # print(f"Expr: {''.join(eq)}: {out['F']}")
        if out['F'] < self.best_f:
            self.best_f = out['F']
            self.best_expression = ["".join(eq), out['F']]
            print(f"{self.best_expression[0]}: {self.best_expression[1]}")
        self.add_model("".join(eq), out['F'])


class SRProblemOther(Problem):
    def __init__(self, generator, tdata, dim, default_value=1e10):
        self.generator = generator
        self.tdata = tdata
        self.default_value = default_value
        self.input_dim = dim
        self.models = dict()
        self.best_f = 9e+50
        super().__init__(n_var=dim, n_obj=1)

    def check_model(self, model):
        if isinstance(model, str):
            tokens = tokenize_expr(model)
        else:
            tokens = model
            model = "".join(tokens)

        if model in self.models:
            self.models[model]["trees"] += 1
            return self.models[model]["error"]
        else:
            if model == "":
                self.models[""] = {"eq": "", "error": self.default_value, "trees": 1, "valid": False}
                return self.default_value
            else:
                rmse, good = eval_eq(tokens, self.tdata, self.default_value)
                self.models[model] = {"eq": model, "error": rmse, "trees": 1, "valid": good}
                return rmse

    def _evaluate(self, x, out, *args, **kwargs):
        eqs = self.generator.decode(x)
        out_rmse = np.zeros(x.shape[0])
        for i, eq in enumerate(eqs):
            try:
                rmse = self.check_model(eq)
                out_rmse[i] = rmse
            except:
                rmse = self.check_model("")
                out_rmse[i] = rmse
            finally:
                if rmse < self.best_f:
                    self.best_f = rmse
        out["F"] = out_rmse


class TorchNormalSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [torch.normal(problem.input_mean).numpy() for i in range(n_samples)]


class TensorflowNormalSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.random.normal(size=(n_samples, problem.input_dim))


class BestTermination(Termination):
    def __init__(self, min_f=1e-10, n_max_gen=500) -> None:
        super().__init__()
        self.min_f = min_f
        self.max_gen = MaximumGenerationTermination(n_max_gen)

    def _update(self, algorithm):
        if algorithm.problem.best_f < self.min_f:
            self.terminate()
        return self.max_gen.update(algorithm)


class LICrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        weights = np.random.random(X.shape[1])
        # The input of has the following shape (n_parents, n_matings, n_var)
        return (X[0, :]*weights[:, None] + X[1, :]*(1-weights[:, None]))[None, :, :]


class RandomMutationOther(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        mean, var = problem.generator.encode_s([eq for eq in problem.generator.decode_latent(X) if eq != ""])
        mutation_scale = np.random.random(mean.shape[0])
        std = mutation_scale[:, None]*(np.exp(var / 2.0) - 1) + 1
        new = [np.random.normal(mutation_scale[i]*mean[i], std[i]) for i in range(mean.shape[0])]
        for i in range(X.shape[0]-mean.shape[0]):
            new.append(np.random.normal(size=problem.input_dim))
        return np.array(new)


class RandomMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        new = []
        for i in range(X.shape[0]):
            eq = problem.generator.decode_latent(torch.tensor(X[i, :])[None, None, :])[0]
            var = problem.generator.encode_list(eq)[1][0, 0].detach().numpy()
            mutation_scale = np.random.random()
            std = mutation_scale * (np.exp(var / 2.0) - 1) + 1
            new.append(torch.normal(torch.tensor(mutation_scale*X[i]), std=torch.tensor(std)).numpy())
        return np.array(new, dtype=np.float32)


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):
    print("Exception")# Custom signal handler
    raise TimeoutException


def eval_tree(tree, data, symbols):
    l = None
    r = None
    if tree.left is not None:
        l = eval_tree(tree.left, data, symbols)
    if tree.right is not None:
        r = eval_tree(tree.right, data, symbols)
    return symbols[tree.symbol]["fun"](l, r, data)


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

def eval_eq(eq, data, default_value=1e10):
    good = True
    try:
        tree = tokens_to_tree(eq, symbols=s_for_tokenization)
        with ignore_warnings(RuntimeWarning):
            yp = eval_tree(tree, data, s_for_tokenization)
            if any(np.iscomplex(yp)):
                yp = yp.real()
        rmse = np.sqrt(np.square(np.subtract(data[:, -1], yp)).mean())
        if np.isfinite(rmse):
            value = rmse
        else:
            value = default_value
    except:
        good = False
        value = default_value
    return value, good


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Nguyen benchmark', description='Run a ED benchmark')
    parser.add_argument("-eq_num", choices=range(1, 11), required=True, action="store", type=int)
    parser.add_argument("-baseline", choices=['ProGED', 'HVAE_random', 'HVAE_evo', 'CVAE_random', 'CVAE_evo', 'GVAE_random', 'GVAE_evo'], action='store', required=True)
    parser.add_argument("-params", action='store', default=None)
    parser.add_argument("-dimension", action="store", type=int)
    parser.add_argument("-seed", type=int)
    args = parser.parse_args()

    # Read data

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    train, test = read_eq_data(args.eq_num)

    if args.baseline == "ProGED":
        grammar = GeneratorGrammar(grammar)
        ed = EqDisco(data=train, variable_names=["X", 'Y'], generator=grammar, sample_size=100000, verbosity=0)
        ed.generate_models()
        ed.fit_models()
        print(len(ed.models))
        print(ed.get_results())
        ed.write_results(f"results/ProGED_corrected/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
    elif args.baseline == "HVAE_random":
        generator = GeneratorHVAE(args.params, ["X"], universal_symbols)
        ed = EqDisco(data=train, variable_names=["X", 'Y'], generator=generator, sample_size=100000, verbosity=0)
        ed.generate_models()
        ed.fit_models()
        print(len(ed.models))
        print(ed.get_results())
        ed.write_results(f"results/hvae_random_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
    elif args.baseline == "HVAE_evo":
        if train.shape[1] == 3:
            universal_symbols.insert(1, {"symbol": 'Y', "type": SymType.Var, "precedence": 5, "fun": lambda l, r, d: d[:, 1]})
        s = ["X"] if train.shape[1] == 2 else ["X", "Y"]
        generator = GeneratorHVAE(args.params, s, universal_symbols)
        ga = GA(pop_size=200, sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(generator, train, args.dimension)
        res = minimize(problem, ga, BestTermination(), verbose=True)
        with open(f"results/reproducable2rust/nguyen_{args.eq_num}_{time.time()}.json", "w") as file:
            # json.dump({"best": problem.best_expr, "all": list(problem.models.values())}, file)
            for i in range(len(problem.models)):
                problem.models[i]["trees"] = problem.evaluated_models[problem.models[i]["eq"]]
            json.dump({"best": problem.best_expression, "all": problem.models}, file)

    # -----------------------------------------------------------------------------------------------------------------

    elif args.baseline == "CVAE_random":
        charlist = ['X', '+', '-', '*', '/', '^2', '^3', '^4', '^5', 'sin', 'cos', 'exp', 'log', 'sqrt', '(', ')', '']
        model = equation_vae.EquationCharacterModel(args.params, latent_rep_size=args.dimension, charlist=charlist, ml=56)
        seen_eqs = dict()
        for _ in tqdm(range(2000)):
            eqs = model.decode(np.random.normal(size=(50, args.dimension)))
            for eq in eqs:
                eq_str = "".join(eq)
                if eq_str in seen_eqs:
                    seen_eqs[eq_str]["trees"] += 1
                else:
                    rmse, good = eval_eq(eq, train)
                    seen_eqs[eq_str] = {"eq": eq_str, "error": rmse, "trees": 1, "valid": good}
        with open(f"results/cvae_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(seen_eqs.values()), file)
    elif args.baseline == "CVAE_evo":
        charlist = ['X', '+', '-', '*', '/', '^2', '^3', '^4', '^5', 'sin', 'cos', 'exp', 'log', 'sqrt', '(', ')', '']
        model = equation_vae.EquationCharacterModel(args.params, latent_rep_size=args.dimension, charlist=charlist, ml=56)
        ga = GA(pop_size=200, sampling=TensorflowNormalSampling(), crossover=LICrossover(), mutation=RandomMutationOther(),
                eliminate_duplicates=False)
        problem = SRProblemOther(model, train, args.dimension)
        res = minimize(problem, ga, BestTermination(), verbose=True)
        with open(f"results/cvae_evo/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(problem.models.values()), file)
    elif args.baseline == "GVAE_random":
        model = equation_vae.EquationGrammarModel(args.params, latent_rep_size=args.dimension, mlength=84)
        seen_eqs = dict()
        for _ in tqdm(range(2000)):
            eqs = model.decode(np.random.normal(size=(50, args.dimension)))
            for eq in eqs:
                if eq in seen_eqs:
                    seen_eqs[eq]["trees"] += 1
                else:
                    rmse, good = eval_eq(eq, train)
                    seen_eqs[eq] = {"eq": eq, "error": rmse, "trees": 1, "valid": good}
        with open(f"results/gvae_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(seen_eqs.values()), file)
    elif args.baseline == "GVAE_evo":
        model = equation_vae.EquationGrammarModel(args.params, latent_rep_size=args.dimension, mlength=84)
        ga = GA(pop_size=200, sampling=TensorflowNormalSampling(), crossover=LICrossover(), mutation=RandomMutationOther(),
                eliminate_duplicates=False)
        problem = SRProblemOther(model, train, args.dimension)
        res = minimize(problem, ga, BestTermination(), verbose=True)
        with open(f"results/gvae_evo/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(problem.models.values()), file)



