import os, sys

my_lib_path = os.path.abspath('../../')
# other_baselines = "/home/sebastianmeznar/Downloads/other_baselines_2/other_baselines"
sys.path.append(my_lib_path)
# sys.path.append(other_baselines)

import argparse
import json
import signal

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
from sympy import parse_expr
from tqdm import tqdm

from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221, tokens_to_tree
from ProGED.generators.grammar import GeneratorGrammar
from ProGED import EqDisco
from ProGED import Model
from ProGED import ModelBox
from ProGED import EDTask
from ProGED.parameter_estimation import fit_models

# import equation_vae

universal_symbols = [{"symbol": 'X', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'Y', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'C', "type": SymType.Const, "precedence": 5},
                     {"symbol": '^2', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '^3', "type": SymType.Fun, "precedence": -1},
                     {"symbol": '+', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '-', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '*', "type": SymType.Operator, "precedence": 1},
                     {"symbol": '/', "type": SymType.Operator, "precedence": 1},
                     {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'sin', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'cos', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'exp', "type": SymType.Fun, "precedence": 5}]

grammar = """E -> E '+' F [0.2]
E -> E '-' F [0.2]
E -> F [0.6]
F -> F '*' T [0.2]
F -> F '/' T [0.2]
F -> T [0.6]
T -> V [0.4]
T -> 'C' [0.3]
T -> A [0.3]
A -> '(' E ')' P [0.1]
A -> '(' E ')' [0.55]
A -> R '(' E ')' [0.35]
V -> 'X' [0.5]
V -> 'Y' [0.5]
P -> '^2' [0.8]
P -> '^3' [0.2]
R -> 'sin' [0.25]
R -> 'cos' [0.25]
R -> 'exp' [0.25]
R -> 'sqrt' [0.25]"""


def read_eq_data(eq_number):
    train = []
    test = []
    with open(f"/home/sebastian/Downloads/Feynman_with_units/{eq_number}", "r") as file:
        for i, row in enumerate(file):
            line = [float(t) for t in row.strip().split(" ")]
            if i < 10000:
                train.append(line)
            elif i < 20000:
                test.append(line)
            else:
                break
    return np.array(train), np.array(test)


class SRProblem(ElementwiseProblem):
    def __init__(self, generator, tdata, dim, default_value=1e10):
        self.generator = generator
        self.tdata = tdata
        self.default_value = default_value
        self.input_mean = torch.zeros(next(generator.model.decoder.parameters()).size(0))
        self.models = dict()
        self.best_f = 9e+50
        super().__init__(n_var=dim, n_obj=1)

    def check_model(self, model):
        if isinstance(model, str):
            model_s = model
        else:
            model_s = str(model)
        if model_s in self.models:
            self.models[model_s]["trees"] += 1
            return self.models[model_s]["error"]
        else:
            if model_s == "":
                self.models[""] = {"eq": "", "error": self.default_value, "trees": 1}
                return self.default_value
            else:
                try:
                    with ignore_warnings(RuntimeWarning):
                        mb = ModelBox()
                        mb.add_model(model_s, {'const': 'C', 'x': ["X", "Y"]})
                        fit_models(mb, self.tdata)
                        model = list(mb.values())[0]
                        rmse = model.get_error(dummy=self.default_value)
                except:
                    rmse = self.default_value
                finally:
                    self.models[model_s] = {"eq": str(model), "error": rmse, "trees": 1}
                    return rmse

    def _evaluate(self, x, out, *args, **kwargs):
        eq = self.generator.decode_latent(torch.tensor(x)[None, None, :])
        try:
            model = Model(''.join(eq), sym_vars=["X", "Y"], sym_params=["C"])
            rmse = self.check_model(model)
            out["F"] = rmse
        except:
            rmse = self.check_model("")
            out["F"] = rmse
        finally:
            if rmse < self.best_f:
                self.best_f = rmse


# class SRProblemOther(Problem):
#     def __init__(self, generator, tdata, dim, default_value=1e10):
#         self.generator = generator
#         self.tdata = tdata
#         self.default_value = default_value
#         self.input_dim = dim
#         self.models = dict()
#         self.best_f = 9e+50
#         super().__init__(n_var=dim, n_obj=1)
#
#     def check_model(self, model):
#         if isinstance(model, str):
#             model_s = model
#         else:
#             model_s = str(model)
#         if model_s in self.models:
#             self.models[model_s]["trees"] += 1
#             return self.models[model_s]["error"]
#         else:
#             if model_s == "":
#                 self.models[""] = {"eq": "", "error": self.default_value, "trees": 1}
#                 return self.default_value
#             else:
#                 try:
#                     with ignore_warnings(RuntimeWarning):
#                         yp = model.evaluate(self.tdata[:, :-1])
#                         if any(np.iscomplex(yp)):
#                             yp = yp.real()
#                     rmse = np.sqrt(np.square(np.subtract(self.tdata[:, -1], yp)).mean())
#                     if not np.isfinite(rmse):
#                         rmse = self.default_value
#                 except:
#                     rmse = self.default_value
#                 finally:
#                     self.models[model_s] = {"eq": model_s, "error": rmse, "trees": 1}
#                     return rmse
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         eqs = self.generator.decode(x)
#         out_rmse = np.zeros(x.shape[0])
#         for i, eq in enumerate(eqs):
#             try:
#                 model = Model(eq, sym_vars=["X"])
#                 rmse = self.check_model(model)
#                 out_rmse[i] = rmse
#             except:
#                 rmse = self.check_model("")
#                 out_rmse[i] = rmse
#             finally:
#                 if rmse < self.best_f:
#                     self.best_f = rmse
#         out["F"] = out_rmse


class TorchNormalSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [torch.normal(problem.input_mean).numpy() for i in range(n_samples)]


# class TensorflowNormalSampling(Sampling):
#     def _do(self, problem, n_samples, **kwargs):
#         return np.random.normal(size=(n_samples, problem.input_dim))


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


# class RandomMutationOther(Mutation):
#     def __init__(self):
#         super().__init__()
#
#     def _do(self, problem, X, **kwargs):
#         mean, var = problem.generator.encode_s([eq for eq in problem.generator.decode(X) if eq != ""])
#         mutation_scale = np.random.random(mean.shape[0])
#         std = mutation_scale[:, None]*(np.exp(var / 2.0) - 1) + 1
#         new = [np.random.normal(mutation_scale[i]*mean[i], std[i]) for i in range(mean.shape[0])]
#         for i in range(X.shape[0]-mean.shape[0]):
#             new.append(np.random.normal(size=problem.input_dim))
#         return np.array(new)


class RandomMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        new = []
        for i in range(X.shape[0]):
            eq = problem.generator.decode_latent(torch.tensor(X[i, :])[None, None, :])
            var = problem.generator.encode_list(eq)[1][0, 0].detach().numpy()
            mutation_scale = np.random.random()
            std = mutation_scale * (np.exp(var / 2.0) - 1) + 1
            new.append(torch.normal(torch.tensor(mutation_scale*X[i]), std=torch.tensor(std)).numpy())
        return np.array(new, dtype=np.float32)


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

#
# def eval_eq(eq, data, default_value=1e10):
#     value = 0
#     default_handler = signal.getsignal(signal.SIGALRM)
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(1)
#     good = True
#     try:
#         eq_model = Model(eq, sym_vars=["X"])
#         with ignore_warnings(RuntimeWarning):
#             yp = eq_model.evaluate(data[:, :-1])
#             if any(np.iscomplex(yp)):
#                 yp = yp.real()
#         rmse = np.sqrt(np.square(np.subtract(data[:, -1], yp)).mean())
#         if np.isfinite(rmse):
#             value = rmse
#         else:
#             value = default_value
#     except:
#         good = False
#         value = default_value
#     finally:
#         signal.alarm(0)
#         signal.signal(signal.SIGALRM, default_handler)
#         return value, good


if __name__ == '__main__':
    eq_2var = ['I.6.2', 'I.12.1', 'I.14.4', 'I.25.13', 'I.26.2', 'I.29.4', 'I.34.27', 'I.39.1', 'II.3.24',
               'II.8.31', 'II.11.28', 'II.27.18', 'II.38.14', 'III.12.43']

    parser = argparse.ArgumentParser(prog='Nguyen benchmark', description='Run a ED benchmark')
    parser.add_argument("-eq_num", choices=eq_2var, required=True, action="store")
    parser.add_argument("-baseline", choices=['ProGED', 'HVAE_random', 'HVAE_evo', 'CVAE_random', 'CVAE_evo', 'GVAE_random', 'GVAE_evo'], action='store', required=True)
    parser.add_argument("-params", action='store', default=None)
    parser.add_argument("-dimension", action="store", type=int)
    args = parser.parse_args()

    # Read data
    train, test = read_eq_data(args.eq_num)

    if args.baseline == "ProGED":
        grammar = GeneratorGrammar(grammar)
        ed = EqDisco(data=train, variable_names=["X", 'Y', 'Z'], generator=grammar, sample_size=100000, verbosity=0)
        ed.generate_models()
        ed.fit_models()
        print(len(ed.models))
        print(ed.get_results())
        ed.write_results(f"results/ProGED_corrected/feynman_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
    elif args.baseline == "HVAE_random":
        generator = GeneratorHVAE(args.params, ["X", "Y"], universal_symbols)
        ed = EqDisco(data=train, variable_names=["X", 'Y', 'Z'], generator=generator, sample_size=100000, verbosity=0)
        ed.generate_models()
        ed.fit_models()
        print(len(ed.models))
        print(ed.get_results())
        ed.write_results(f"results/hvae_random_{args.dimension}/feynman_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
    elif args.baseline == "HVAE_evo":
        generator = GeneratorHVAE(args.params, ["X", 'Y'], universal_symbols)
        ga = GA(pop_size=200, sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(generator, train, args.dimension)
        res = minimize(problem, ga, BestTermination(), verbose=True)
        with open(f"results/hvae_evo/feynman_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
            json.dump(list(problem.models.values()), file)

    # -----------------------------------------------------------------------------------------------------------------

    # elif args.baseline == "CVAE_random":
    #     charlist = ['X', '+', '-', '*', '/', '^2', '^3', '^4', '^5', 'sin', 'cos', 'exp', 'log', 'sqrt', '(', ')', '']
    #     model = equation_vae.EquationCharacterModel(args.params, latent_rep_size=args.dimension, charlist=charlist, ml=56)
    #     seen_eqs = dict()
    #     for _ in tqdm(range(2000)):
    #         eqs = model.decode(np.random.normal(size=(50, args.dimension)))
    #         for eq in eqs:
    #             if eq in seen_eqs:
    #                 seen_eqs[eq]["trees"] += 1
    #             else:
    #                 rmse, good = eval_eq(eq, train)
    #                 seen_eqs[eq] = {"eq": eq, "error": rmse, "trees": 1, "valid": good}
    #     with open(f"results/cvae_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
    #         json.dump(list(seen_eqs.values()), file)
    # elif args.baseline == "CVAE_evo":
    #     charlist = ['X', '+', '-', '*', '/', '^2', '^3', '^4', '^5', 'sin', 'cos', 'exp', 'log', 'sqrt', '(', ')', '']
    #     model = equation_vae.EquationCharacterModel(args.params, latent_rep_size=args.dimension, charlist=charlist, ml=56)
    #     ga = GA(pop_size=200, sampling=TensorflowNormalSampling(), crossover=LICrossover(), mutation=RandomMutationOther(),
    #             eliminate_duplicates=False)
    #     problem = SRProblemOther(model, train, args.dimension)
    #     res = minimize(problem, ga, BestTermination(), verbose=True)
    #     with open(f"results/cvae_evo/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
    #         json.dump(list(problem.models.values()), file)
    # elif args.baseline == "GVAE_random":
    #     model = equation_vae.EquationGrammarModel(args.params, latent_rep_size=args.dimension, mlength=84)
    #     seen_eqs = dict()
    #     for _ in tqdm(range(2000)):
    #         eqs = model.decode(np.random.normal(size=(50, args.dimension)))
    #         for eq in eqs:
    #             if eq in seen_eqs:
    #                 seen_eqs[eq]["trees"] += 1
    #             else:
    #                 rmse, good = eval_eq(eq, train)
    #                 seen_eqs[eq] = {"eq": eq, "error": rmse, "trees": 1, "valid": good}
    #     with open(f"results/gvae_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
    #         json.dump(list(seen_eqs.values()), file)
    # elif args.baseline == "GVAE_evo":
    #     model = equation_vae.EquationGrammarModel(args.params, latent_rep_size=args.dimension, mlength=84)
    #     ga = GA(pop_size=200, sampling=TensorflowNormalSampling(), crossover=LICrossover(), mutation=RandomMutationOther(),
    #             eliminate_duplicates=False)
    #     problem = SRProblemOther(model, train, args.dimension)
    #     res = minimize(problem, ga, BestTermination(), verbose=True)
    #     with open(f"results/gvae_evo/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
    #         json.dump(list(problem.models.values()), file)



