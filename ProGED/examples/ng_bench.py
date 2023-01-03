import os, sys

my_lib_path = os.path.abspath('../../')
other_baselines = "/home/sebastian/Downloads/10174_efficient_generator_of_algebra-Supplementary Material/hvae_code/other_baselines"
sys.path.append(my_lib_path)
sys.path.append(other_baselines)

import argparse
import json

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.termination.max_gen import MaximumGenerationTermination
from sympy.testing.pytest import ignore_warnings
from sympy import parse_expr
from tqdm import tqdm

from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221, tokens_to_tree
from ProGED.generators.grammar import GeneratorGrammar
from ProGED import EqDisco
from ProGED import Model

import equation_vae

universal_symbols = [{"symbol": 'X', "type": SymType.Var, "precedence": 5},
                     # {"symbol": 'Y', "type": SymType.Var, "precedence": 5},
                     {"symbol": '^2', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^3', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^4', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^5', "type": SymType.Fun, "precedence": -1},
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


class SRProblem(ElementwiseProblem):
    def __init__(self, generator, tdata, dim, default_value=1e10):
        self.generator = generator
        self.tdata = tdata
        self.default_value = default_value
        self.input_mean = torch.zeros(next(generator.model.decoder.parameters()).size(0))
        self.models = []
        self.evaluated_models = dict()
        self.best_f = 9e+50
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
        eq = self.generator.decode_latent(torch.tensor(x)[None, None, :])
        model = Model("".join(eq), sym_vars=["X"])
        try:
            with ignore_warnings(RuntimeWarning):
                yp = model.evaluate(self.tdata[:, :-1])
                if any(np.iscomplex(yp)):
                    yp = yp.real()
            rmse = np.sqrt(np.square(np.subtract(self.tdata[:, -1], yp)).mean())
            if np.isfinite(rmse):
                out["F"] = rmse
            else:
                out["F"] = self.default_value
        except Exception:
            out["F"] = self.default_value
        finally:
            if out['F'] < self.best_f:
                self.best_f = out['F']
            self.add_model(model, out['F'])


class TorchNormalSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [torch.normal(problem.input_mean).numpy() for i in range(n_samples)]


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


class RandomMutation(Mutation):
    def __init__(self, random_prob=0.75):
        super().__init__()
        self.random_prob = random_prob

    # def _do(self, problem, X, **kwargs):
    #     new = []
    #     for i in range(X.shape[0]):
    #         new.append(torch.normal(problem.input_mean).numpy())
    #         if np.random.random() < self.random_prob:
    #             new.append(torch.normal(problem.input_mean).numpy())
    #         else:
    #             eq = problem.generator.decode_latent(torch.tensor(X[i, :])[None, None, :])
    #             var = problem.generator.encode_list(eq)[1][0, 0].detach().numpy()
    #             std = np.exp(var / 2.0)
    #             new.append(X[i] + np.random.random(X.shape[1])*std)
    #     return np.array([torch.normal(problem.input_mean).numpy() for i in range(X.shape[0])])

    def _do(self, problem, X, **kwargs):
        new = []
        for i in range(X.shape[0]):
            eq = problem.generator.decode_latent(torch.tensor(X[i, :])[None, None, :])
            var = problem.generator.encode_list(eq)[1][0, 0].detach().numpy()
            mutation_scale = np.random.random()
            std = mutation_scale * (np.exp(var / 2.0) - 1) + 1
            new.append(torch.normal(torch.tensor(mutation_scale*X[i]), std=torch.tensor(std)).numpy())
            # new.append(mutation_scale * X[i] + np.random.random(X.shape[1])*std)
        return np.array(new, dtype=np.float32)


def eval_eq(eq, data, default_value=1e10):
    try:
        eq_model = Model(eq, sym_vars=["X"])
        with ignore_warnings(RuntimeWarning):
            yp = eq_model.evaluate(data[:, :-1])
            if any(np.iscomplex(yp)):
                yp = yp.real()
        rmse = np.sqrt(np.square(np.subtract(data[:, -1], yp)).mean())
        if np.isfinite(rmse):
            return rmse
        else:
            return default_value
    except:
        return default_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Nguyen benchmark', description='Run a ED benchmark')
    parser.add_argument("-eq_num", choices=range(1, 11), required=True, action="store", type=int)
    parser.add_argument("-baseline", choices=['ProGED', 'HVAE_random', 'HVAE_evo', 'CVAE_random', 'CVAE_evo', 'GVAE_random', 'GVAE_evo'], action='store', required=True)
    parser.add_argument("-params", action='store', default=None)
    parser.add_argument("-dimension", action="store", type=int)
    args = parser.parse_args()

    # Read data
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
        generator = GeneratorHVAE(args.params, ["X"], universal_symbols)
        ga = GA(pop_size=200, sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(generator, train, args.dimension)
        res = minimize(problem, ga, BestTermination(), verbose=True)
        with open(f"results/hvae_evo/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            for i in range(len(problem.models)):
                problem.models[i]["trees"] = problem.evaluated_models[problem.models[i]["eq"]]
            json.dump(problem.models, file)
    elif args.baseline == "CVAE_random":
        charlist = ['X', 'c', '+', '-', '*', '/', '^', 'sin', 'cos', '(', ')', '']
        char_weights = "/home/sebastian/Downloads/10174_efficient_generator_of_algebra-Supplementary Material/hvae_code/other_baselines/eq_cvae_h32_c234_L32_E5_batchB.hdf5"
        model = equation_vae.EquationCharacterModel(char_weights, latent_rep_size=args.dimension, charlist=charlist)
        seen_eqs = dict()
        for _ in tqdm(range(2000)):
            eqs = model.decode(np.random.normal(size=(50, args.dimension)))
            for eq in eqs:
                if eq in seen_eqs:
                    seen_eqs[eq]["trees"] += 1
                else:
                    rmse = eval_eq(eq, train)
                    seen_eqs[eq] = {"eq": eq, "error": rmse, "trees": 1}
        with open(f"results/cvae_random/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(seen_eqs.values()), file)
    elif args.baseline == "CVAE_evo":
        pass
    elif args.baseline == "GVAE_random":
        grammar_weights = "/home/sebastian/Downloads/10174_efficient_generator_of_algebra-Supplementary Material/hvae_code/other_baselines/eq_gvae_h128_c234_L128_E150_batchB.hdf5"
        model = equation_vae.EquationGrammarModel(grammar_weights, latent_rep_size=args.dimension)
        seen_eqs = dict()
        for _ in tqdm(range(2000)):
            eqs = model.decode(np.random.normal(size=(50, args.dimension)))
            for eq in eqs:
                if eq in seen_eqs:
                    seen_eqs[eq]["trees"] += 1
                else:
                    rmse = eval_eq(eq, train)
                    seen_eqs[eq] = {"eq": eq, "error": rmse, "trees": 1}
        with open(f"results/gvae_random/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(seen_eqs.values()), file)
    elif args.baseline == "GVAE_evo":
        pass



