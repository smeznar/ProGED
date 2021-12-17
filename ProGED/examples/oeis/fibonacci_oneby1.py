"""Run equation discovery on OEIS sequences to discover 
direct, recursive or even direct-recursive equations.
"""

import numpy as np
import sympy as sp
import pandas as pd
import time
import sys
import re 
# from scipy.optimize import brute, shgo, rosen, dual_annealing

from ProGED.equation_discoverer import EqDisco
from ProGED.parameter_estimation import DE_fit, hyperopt_fit  # integer_brute_fit, shgo_fit, DAnnealing_fit
import ProGED.examples.tee_so as te  # Log using manually copied class from a forum.
from hyperopt import hp, tpe, rand

print("\n"*5, "TRY GRAMMAR WITH / DIVIDING NONCONSTANT VARIABLES (an_1, n, ...)")
print("TRY GRAMMAR WITH / DIVIDING NONCONSTANT VARIABLES (an_1, n, ...)")
print("TRY GRAMMAR WITH / DIVIDING NONCONSTANT VARIABLES (an_1, n, ...)")
print("parameter_estimation avoids more than 5 parameters to estimate. In oeis exact may use more than 5 constants? ")
print("TRY GRAMMAR WITH / DIVIDING NONCONSTANT VARIABLES (an_1, n, ...)", "\n"*5, )

##############################
# Quick usage is with flags:
#  --seq_only=A000045 --sample_size=3 # (Fibonacci with 3 models fited)  
# search for flags with: flags_dict
###############

#####  To log output to file, for later inspection.  ########
# Command line arguments (to avoid logging):
is_tee_flag = True  # Do not change manually!! Change is_tee.
message = ""
double_flags = set(sys.argv[1:])
flags_dict = { i.split("=")[0]:  i.split("=")[1] for i in sys.argv[1:] if len(i.split("="))>1}
# Usage of flags_dict: $ fibonacci.py --order=3 --is_direct=True.
# List of all flags:
#   -n or --no-log .. do not log  # --do-log .. do log  # --msg .. log name  
# --is_direct=<bool>  # --order=<num>  # --sample_size=<num>
if len(sys.argv) >= 2:
    if sys.argv[1][0] == "-" and not sys.argv[1][1] == "-":
        single_flags = set(sys.argv[1][1:])
        # print(single_flags)
        if "n" in single_flags:
            is_tee_flag = False
    if "--no-log" in double_flags:
        is_tee_flag = False
    if "--msg" in double_flags:
        message = sys.argv[2] + "_"
    if "--is_direct" in flags_dict:
        if flags_dict["--is_direct"] == "True":
            flags_dict["--is_direct"] = True
        elif flags_dict["--is_direct"] == "False":
            flags_dict["--is_direct"] = False
        else:
            flags_dict["--is_direct"] = int(flags_dict["--is_direct"])
if not is_tee_flag:
    print("\nNo-log flag detected!\n")

is_tee, log_name, log_directory = False, "log_oeis_", "outputs/"
# is_tee, log_name, log_directory = True, "log_oeis_", "outputs/"
# random = str(np.random.random())[2:]
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

log_filename = log_name + message + timestamp + ".txt"
if (is_tee and is_tee_flag) or ("--do-log" in double_flags) or ("--is_tee" in double_flags):
    try:
        log_object = te.Tee(f"{log_directory}{log_filename}")
        print(f"Output will be logged in: {log_directory}{log_filename}")
    except FileNotFoundError:
        log_object = te.Tee(log_filename)
        print(f"Output will be logged in: {log_filename}")
######## End Of Logging ##  ##  ##                  ########

np.random.seed(0)
has_titles = 1
csv = pd.read_csv('oeis_selection.csv')[has_titles:]
# csv = csv.astype('int64')
# print("csv", csv)
# csv = csv.astype('float')
print("csv", csv)
# 1/0
terms_count, seqs_count = csv.shape
# Old for fibonacci only:
seq_id = "A000045"
prt_id = "A000041"
fibs = list(csv[seq_id])  # fibonacci = A000045
prts = list(csv[prt_id])  # fibonacci = A000045
print("fibs", fibs)
fibs = np.array(fibs)
prts = np.array(prts)
# oeis = fibs

# oeis = fibs[2:]
# fibs_old = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
#  1597,2584,4181,6765,10946,17711,28657,46368,75025,
#  121393,196418,317811,514229,832040,1346269,
#  2178309,3524578,5702887,9227465,14930352,24157817,
#  39088169,63245986,102334155]
# print(type(fibs[0]))
# assert fibs_old[len(fibs_old)-1] == fibs[len(fibs_old)-1]
# primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
#  61,67,71,73,79,83,89,97,101,103,107,109,113,127,
#  131,137,139,149,151,157,163,167,173,179,181,191,
#  193,197,199,211,223,227,229,233,239,241,251,257,
#  263,269,271][:40+1]
# oeis = primes
# seq = numpy_seq

# seq = np.array(oeis)

# we want:
# 0 1 2
# 1 2 3
# 2 3 4
# an = a{n-1} + a{n-2} is recurrece relation of order 2 (2 preceeding terms).
def grid (order, seq, direct=False):
    """order -> (n-order) x (0/1+order+1) data matrix
    order ... number of previous terms in formula
    0/1+ ... for direct formula also (a_n = a_{n-1} + n).
    +1 ... a_n column.
    """
    n = seq.shape[0] # (40+1)
    # print(n)
    indexes = np.fromfunction((lambda i,j:i+j), (n-order, order+1), dtype=int)
    # print(indexes)
    # print(seq[indexes])
    first_column = indexes[:, [0]]
    # print(first_column)
    if direct:
        return np.hstack((first_column, seq[indexes]))
    # print('indexes')
    # print(indexes)
    return seq[indexes]
# def fij(i, j):
#     # return 0 if (j<=i) else 2
#     return max(i-j, 0)
# def aij(i, j):
#     # return fibs[i-j] if (j<=i) else 0
#     # return fibs[i-j] if True else 0
#     a = i
#     b = j
#     return 1 if a>2 else 0
# def cutgrid(seq):
#     n = len(seq)
#     return np.fromfunction((lambda i,j: np.maximum(i-j,0) , (n-1, n-1))


def grid2(seq: np.ndarray):  # seq.shape=(1, N)
    n = len(seq)
    # n = seq.shape[0]
    indexes = np.fromfunction((lambda i,j: np.maximum(i-j,0)) , (n-1, n-1)).astype(int)
    cut_zero = seq[indexes] * np.tri(n-1).astype(int)
    # data = np.hstack((np.array(seq)[:(n-1)].reshape(-1,1), np.arange(n-1).reshape(-1,1), seq[indexes]))
    data = np.hstack((np.array(seq)[1:].reshape(-1, 1), np.arange(1, n).reshape(-1, 1), cut_zero))
    return data

def grid_numpy(seq_id: str, number_of_terms: int):
    seq = np.array(sp.Matrix(list(csv[seq_id])[:number_of_terms]).T)[0]
    n = len(seq)
    # n = seq.shape[0]
    indexes = np.fromfunction((lambda i,j: np.maximum(i-j,0)) , (n-1, n-1)).astype(int)
    cut_zero = seq[indexes] * np.tri(n-1).astype(int)
    data = np.hstack((np.array(seq)[1:].reshape(-1, 1), np.arange(1, n).reshape(-1, 1), cut_zero))
    return data

def grid_sympy(seq_id: str, number_of_terms: int):  # seq.shape=(N, 1)
    seq = sp.Matrix(csv[seq_id][:number_of_terms])
    n = len(seq)
    indexes_sympy_uncut = sp.Matrix(n-1, n-1, (lambda i,j: (seq[max(i-j,0)])*(1 if i>=j else 0)))
    data = sp.Matrix.hstack(
                seq[1:,:], 
                sp.Matrix([i for i in range(1, n)]),
                indexes_sympy_uncut)
    return data

# seq_id='A000108'
# # seq_id='A000045'
# grids = grid_sympy(seq_id, 50)
# datal = grids
# print('bool', grid_sympy(seq_id, 50) == sp.Matrix(grid2(np.array(list(csv[seq_id])[:50]).astype('float')).astype(int)))
# print('bool', grids == sp.Matrix(grid2(np.array(list(csv[seq_id])[:50]).astype('float')).astype(int)))
# # 1/0
# datan =  np.array(sp.Matrix(list(csv[seq_id])[:50]).T)[0]
# grid_nu = grid_numpy(seq_id, 50)
# print('datan', datan)
# gridy2 =  grid2(datan)
# print('bool lte', grids == sp.Matrix(gridy2))
# # print('bool lte', sum(not (gridy2 == grid_nu)))
# print('bool lte', sum(sum(gridy2 != grid_nu)))
# 1/0
# # gridy2 =  grid2(np.array(list(csv[seq_id])[:50]).astype('float')).astype(int)
# print('grid_sympy', grids.shape, gridy2.shape)
# print('type', type(grids), type(gridy2))
# # 1/0
# print('grid_sympy', grids)
# print('grid2', gridy2)
# print('grid_sympy', grids[:10, :10])
# print('grid2', gridy2[:10, :10])
# print('grid_sympy',  grids[-4:, :4])
# print('grid2',      gridy2[-4:, :4])
# print('eq', grids[-4:, :4] == gridy2[-4:, :4])

# 1/0
# # datal = grid2(np.array(list(csv[seq_id])[:50]).astype('float'))
# # print('input grid2', np.array(list(csv[seq_id])[:50]))
# # datal = np.array(sp.Matrix(csv[seq_id][:50]).T)[0]
# # datal = grid2(np.array(sp.Matrix(csv[seq_id][:50]).T)[0])
# print("datal", datal)
# print("datal[10]", datal[:10, :10])
# print("datal[-10:]", datal[-4:, :4])
# 1/0

# # print("csv[seq_id]", type(list(csv[seq_id])[0]))
# print("csv[seq_id]", (list(csv[seq_id])))
# print("sympy", sp.Matrix(list(csv[seq_id])))
# print("sympy", np.array(sp.Matrix(csv[seq_id])))
# print("csv[seq_id]", type(np.array(list(csv[seq_id]))))
# print("csv[seq_id]", [type(i) for i in csv[seq_id]])
# 1/0
# # print(grid2(['a_n', 'a_{n-1}']))
# # print([f"a_{{n-{i}}} & 1 & " for i in range(1, 50)])
# # 1/0

# print(prts)
# print(fibs)
# # # grid(3, fibs, True)
# print(grid2(fibs))
# print(grid2(prts))
# print(grid2(fibs)[:17,:48].astype(int))
# print('cats:')
# print(grid2(prts)[:7,:6])
# # row = grid[3]
# # print(aij(1,1))
# # for i in range(10):
# #     for j in range(10):
# #         print(aij(i,j))

# # print(fibs)
# # print(fibs[row])
# print(grid2(fibs))
# # print(fibs[grid2(fibs)])
# # print(fij(1, 45))
# print('end of prog')
# 1/0


########main#settings#####################################
## Note: order and is_direct is overwritten by commandline arguments.
## order, is_direct = 2, False  # recursive
## order, is_direct = 4, False  # recursive
#order, is_direct = 0, True  # direct
## order, is_direct = 2, True  # direct
## order, is_direct = 4, True  # direct
## Override manual settings with input cmd line flags:
#order = int(flags_dict.get("--order", order))
#is_direct = flags_dict.get("--is_direct", is_direct)

# seq_name = "fibonacci"
seq_name = "general_wnb"
# grammar_template_name = "polynomial"
grammar_template_name = "polynomial2"
# grammar_template_name = "rational"
# grammar_template_name = "rational"
# grammar_template_name = "simplerational"
# grammar_template_name = "simplerational2"
# grammar_template_name = "universal"
# grammar_template_name = "polytrig"
# grammar_template_name = "trigonometric"

# sample_size = 1
sample_size = 4
# sample_size = 2
# sample_size = 3
# sample_size = 6
# sample_size = 10
# sample_size = 16
# sample_size = 15
# sample_size = 20
# sample_size = 47
# sample_size = 30
# sample_size = 50
# sample_size = 100
# sample_size = 1000
sample_size = int(flags_dict.get("--sample_size", sample_size))
### lower_upper_bounds = (-5, 5) if is_direct else (-10, 10)
# lower_upper_bounds = (-10, 10)  # recursive
lower_upper_bounds = (-5, 5)  # direct
lower_upper_bounds = (-4, 4)  # new grammar
# lower_upper_bounds = (-2, 2)  # direct
#########################################################

p_T = [0.4, 0.6]  # default settings, does nothing
p_R = [0.6, 0.4]
# p_F = [0.1, 0.8, 0.1]
p_F = [0.333, 0.333, 0.333]  # before exact
p_F = []  # for exact
functions = ["'sqrt'", "'exp'", "'log'",]  # before exact
functions = []
generator_settings = {
    # "variables": variables,
    # "functions": ["'exp'"],
    "functions": functions,
     # "p_T": p_T, "p_R": p_R, 
     # "p_R": p_R, 
     "p_F": p_F,
     }

# if (seq_name, order, is_direct) == ("fibonacci", 2, False):
#     p_T = [0.4, 0.6]  # for rec fib
#     p_R = [0.9, 0.1]
q = 1/2  # For direct Fibonacci.
# q = 0.01/10  # For recursive Fibonacci.
# p = 8/10
p = 3/10

random_seed = 0  # polynomial2 grammar (rec is 41. model)
# random_seed = 1  # rec
# random_seed = 5  # simplerational2 grammar (65. model is rec, 57. model direct (exp) )   
# random_seed = 10  # simplerational2 grammar for rec (sec or third model)
# random_seed = 10  # simplerational2 grammar for rec (17. model)
# random_seed = 33  # simplerational2 grammar for rec (first 5 models)
# random_seed = 97  # simplerational2 grammar for rec (first 5 model)
# random_seed = 70 # has additional terms, try it out # simplerational2 grammar for rec 
random_seed = 86  # simplerational2 grammar for rec (second model) 
random_seed = int(flags_dict.get("--seed", random_seed))
# seed 0 , size 20 (16)
# ??? seed3 size 15 an-1 + an-2 + c3 rec  ???
# seed 1 size 20 ali 4 an-1 + an-2 rec 

task_type = "algebraic"  # Originalno blo nastimano do 5.5.2021.
# task_type = "oeis_exact"  # For oeis exact equation discovery.
# task_type = "integer_algebraic"  # Originalno blo nastimano do 5.5.2021.
# If recursive, allways integer algebraic:
# if order > 0:
#     task_type = 'integer_algebraic'
optimizer = 'differential_evolution'
optimizer = 'oeis_exact'
timeout = np.inf

NUMBER_OF_TERMS_PRESET = 20
number_of_terms = int(flags_dict.get("--number_of_terms", NUMBER_OF_TERMS_PRESET))
oeis_number_of_terms = number_of_terms
MAX_ORDER_PRESET = 20
max_order = int(flags_dict.get("--number_of_terms", MAX_ORDER_PRESET))

# def oeis_eq_disco(seq_id: str, is_direct: bool, order: int): 
# def oeis_eq_disco(seq_id: str, number_of_terms=50, max_order=20): 
def oeis_eq_disco(seq_id: str, number_of_terms: int, max_order: int): 
    """Run eq. discovery of given OEIS sequence."""

    doc2 = """
    data = grid ... 20+20 rows
    ED.disco(estimation_settings['number_of_terms'] = 20)
    """
    # data = grid(order, np.array(list(csv[seq_id])), is_direct)
    # data = grid2(np.array(list(csv[seq_id])))
    # First 30 instead 50 terms in sequence (for catalan):
    # data = grid2(np.array(list(csv[seq_id])[:number_of_terms]))
    data = grid_sympy(seq_id, number_of_terms=(number_of_terms + max_order))
    print('data shape', data.shape)
    print('data:', data)
    print('data[:4][:4] :', data[:6, :6], data[:, -2])
    n = data.shape[0] + 1  # = 50
    # variable_names_ = [f"an_{i}" for i in range(order, 0, -1)] + ["an"]
    variable_names = ["an", "n"] + [f"an_{i}" for i in range(1, (n-1)+1)]
    #%# print('len variable_names', len(variable_names))
    # variable_names = ["n"]+variable_names_ if is_direct else variable_names_
    # variables = [f"'{an_i}'" for an_i in variable_names[1:]]
    # print('len variables', len(variables))
    # print(variable_names)
    # print(variables)
    # print(data.shape, type(data), data)
    # q = q
    # p = p
    pis = [p**i for i in range(1, (n-1)+1)]
    # pis = [max(p**i, 0) for i in range(1, (n-1)+1)]
    # pis = [p**i+1e-04 for i in range(1, (n-1)+1)]
    # print(pis)
    # print(len(pis), 'len pis')
    coef = (1-q)/sum(pis)
    # pis = coef * np.array(pis) + 1e-04
    pis = coef * np.array(pis) + 1e-03
    coef = (1-q)/sum(pis)
    variable_probabilities = np.hstack((np.array([q]), coef*np.array(pis)))
    # variable_probabilities = [0.00001, 0.99999]
    # variable_probabilities = [1, 0]
    # 1/0

    # check probs
    #%# print('variable_probabilities', variable_probabilities, sum(variable_probabilities), len(variable_probabilities))
    # print('variable_probabilities', variable_probabilities)
    # print('min(variable_probabilities)', min(variable_probabilities))
    # print('sum(variable_probabilities)', sum(variable_probabilities))
    # for i in range(len(variable_probabilities)-1):
    #     print(variable_probabilities[i]/variable_probabilities[i+1], pis[i]/pis[i+1])
    # 1/0

    np.random.seed(random_seed)
    ED = EqDisco(
        data=data,
        task=None,
        # target_variable_index=-1,
        target_variable_index=0,
        # variable_names=["an_2", "an_1", "an"],
        variable_names=variable_names,
        variable_probabilities=variable_probabilities,
        # sample_size = 16,  # for recursive
        # sample_size = 10,  # for direct fib
        sample_size=sample_size,
        # sample_size = 50,  # for recursive
        # sample_size = 38,  # for recursive
        # sample_size = 100,  # for recursive
        verbosity=0,
        # verbosity = 3,
        generator="grammar", 
        generator_template_name = grammar_template_name,
        # generator_settings={"variables": ["'an_2'", "'an_1'"],
        generator_settings=generator_settings,

        estimation_settings={
            'oeis_number_of_terms': number_of_terms,
            "verbosity": 3,
            # "verbosity": 1,
            # "verbosity": 0,
             "task_type": task_type,
             # "task_type": "algebraic",
             # "task_type": "integer algebraic",
            # "task_type": "oeis_recursive_error",  # bad idea
             "lower_upper_bounds": lower_upper_bounds,
             # "lower_upper_bounds": (-1000, 1000),  # najde (1001 pa ne) rec fib
            #  "lower_upper_bounds": (-100, 100),  # ne najde DE
            #  "lower_upper_bounds": (-25, 25),  # DA ne najde
            #  "lower_upper_bounds": (-11, 11),  # shgo limit
            #  "lower_upper_bounds": (-14, 14),  # DA dela
            # "lower_upper_bounds": (-2, 2),  # for direct
            # "lower_upper_bounds": (-4, 4),  # for direct fib
            # "lower_upper_bounds": (-5, 5), 
            # "lower_upper_bounds": (-8, 8),  # long enough for brute
            # "optimizer": 'differential_evolution',
            "optimizer": optimizer,
            # "optimizer": 'hyperopt',
            "timeout": timeout,
            # "timeout": 1,
            # "timeout": 13,
            # "timeout_privilege": 30,

                ## hyperopt:
            # "hyperopt_max_evals": 3250,
            # "hyperopt_max_evals": 550,  # finds if result=min(10**6, hyperopt...)
            # "hyperopt_max_evals": 50,
            # "hyperopt_max_evals": 750,
            # "hyperopt_max_evals": 2000,
            # "hyperopt_max_evals": 700,
            # "hyperopt_max_evals": 500,
            # "hyperopt_max_evals": 300,
            # "hyperopt_space_fn": hp.randint,
            # "hyperopt_space_fn": hp.choice,
            # "hyperopt_space_fn": hp.loguniform,  # Seems working, but loss 785963174750.8921 in 2000 evals.
            # "hyperopt_space_fn": hp.qnormal,
            # "hyperopt_space_fn": hp.normal,
            # "hyperopt_space_args": (lower_upper_bounds[0], lower_upper_bounds[1]),
            # "hyperopt_space_args": ([-2, -1, 0, 1],),
            # "hyperopt_space_args": (0, 5, 1),
            # "hyperopt_space_args": (0, 2),  # Seems working for hp.normal, but loss 1492702199922.7131 in 2000 evals.
            # "hyperopt_space_args": [lower_upper_bounds[0]],
            # "hyperopt_space_kwargs": {"high": lower_upper_bounds[1]},
            # "hyperopt_algo": tpe.suggest,  # Works, with 1000 evals and (-2, 2) finds recursive. 
            # "hyperopt_algo": rand.suggest,
        }
    )

    print(f"=>> Grammar used: \n{ED.generator}\n")

    # 1/0
    # for i in range(0, 10):
    #     np.random.seed(i)
    #     print("seed", i)
    #     print(ED)
    #     ED.generate_models()
    #     print(ED.models)
    #     ED.models = None
    # 1/0

    ED.generate_models()
    
    print(ED.models)
    print("task data", type(ED.task.data))
    # exact ed:
    # return 0

    X = ED.task.data[:, 1:]  # dangerous if evaling big integers
    print('X origin', X)
    # X = np.array(ED.task.data[:, 1:], dtype='int')  # solution 1
    # print('X numpy-int', X)
    # X = sp.Matrix(ED.task.data[:, 1:]).applyfunc(sp.Integer)
    # print('X sympy-int', X)
    # s = np.array([[13.], [655594.], [509552245179617111378493440.000]], dtype='int')
    # print(s)

    Y = ED.task.data[:, [0]]  # dangerous if evaling big integers, e.g. lambdify
    print('Y origin', Y)
    # Y = np.array(ED.task.data[:, [0]], dtype='int')  # solution 1
    # print('Y numpy-int', Y)
    # Y = sp.Matrix(ED.task.data[:, [0]]).applyfunc(sp.Integer)
    # print('Y sympy-int', Y)
    # Y = sp.Matrix(np.array(ED.task.data[:, [0]], dtype='int'))  # solution 1
    # Xp = ED.task.data[4:8, 1:]
    # Y = Yp
    # print('Xp and Yp', Xp, '\n', Yp)
    print(f"shapes: task.data {ED.task.data.shape}, X {X.shape}, Y {Y.shape}, ")
    # return 0

    # return 0

#    def model2data (model, X, Y, number_of_terms: int):
#        print('-->> inside model2data begin')
#        print(model.sym_vars, model.expr, type(model.sym_vars))
#        # indie = [(index, model.expr.has(var)) for index, var in enumerate(model.sym_vars)]
#        has_vars = [None for var in model.sym_vars if model.expr.has(var)]
#        nonrecursive_variables = 1  # vars= [n, an_1, ... an_49] \mapsto 1 => 
#        # => recursion_order = len(has_vars) - nonrecursive variables
#        recursion_order = len(has_vars) - nonrecursive_variables
#        remove_rows = recursion_order - 1  # remove first (recursion order - 1) rows. Explain by yourself.
#        # print(remove_rows, has_vars, nonrecursive_variables, recursion_order)
#        print('--<< inside model2data end')
#        # 1/0
#        X = X[remove_rows:(remove_rows + number_of_terms), :]
#        Y = Y[remove_rows:(remove_rows + number_of_terms), :]
#        return X, Y

#    def model2diophant (model, X, Y):
#        """Turns model with data into the matrix and vector of diophantine equations.

#        It assumes polynomial model, i.e. sum of multiplied variables and constants.
#        Input:
#            - model: modelBox object? from model.py module
#            - X: data with non-target attributes (used in right-hand side of 
#                the equations)
#            - Y: column data with target attributes (right-hand side of the 
#                equations)
#        """
#        """
#        blueprint:
#        model -> 2models(c0*var + c1*var +..., vars + var + var + ...) ->  
#                -> manyModels( var1, var2, ..., varRight)
#                -> for every row in X: 
#                        A = [var1(X), var2(X), ..., var_n(X)], b = Y - other_sum_of_vars(X) 


#        """
#    #blueprint:
#    #1.) if type(expr) != Add naredi nekaj drugega ( uporabi algoritem pri #2.) )
#    #2.) if type(expr) == Add glej skico:
#    #   - (n**2, an_1, C2*n, C0*an_1*n, C1*an_3*n**2)  in naredi:
#    #   - funkcijo, ki vsakemu argumentu v expr.args priredi
#    #   - ( (1 (nima konstante), an_1 (expr), ..., (0 (ima konstanto), an_1*n (expr odstranimo konstanto), ...
#    #   - npr. (n**2, an_1, C2*n, C0*an_1*n, C1*an_3*n**2) ->  ((1, n**2), (1, an_1), (0, n), (0, an_1*n), (0, an_3*n**2))
#    #   - nato to pretvorimo v (1, 0, 0, 0) ... tj. index 0, da razdelimo v (n**2, an_1, ) in (n, an_1*n, an_3*n**2, )
#    #   - prvi tuple zdruzimo in koncamo z (n**2 + an_1, n, an_1*n, an_3*n**2)
#    #   - sedaj evalviramo:
#    #       - stolpec b = sp.lambdify(n**2 + an_1, lib="math")(X)-Y 
#    #       - stolpec A[:, 0] = sp.lambdify(an_1*n, lib="math")(X) 
#    #       - stolpec A[:, 1] = sp.lambdify(n, lib="math")(X) 
#    #       - ...
#    #   - sestavimo in vrnemo stolpce: A, b

#        print("->-> inside model2diophant() --- ")

#        expr = model.expr
#        # print(isinstance(expr, ))
#        print("Add", isinstance(expr, sp.Add))
#        print("Pow", isinstance(expr, sp.Pow))

#        def drop_constant(expr):
#            """Returns pair (bool, croped_expr), where bool logs the change
#            being done to the expression expr by cutting off the constant in 
#            summand `expr`. 

#            E.g.:
#                C1*expr -> (True, expr) or 
#                var -> (False, var)
#            """
#            return expr.has(*model.sym_params), expr.subs([(i, 1) for i in model.sym_params])



#        print("model.symbols aka. sym_vars", model.sym_vars, model.sym_params)

#        summands = expr.args if isinstance(expr, sp.Add) else (expr,)
#        filtered = [drop_constant(summand) for summand in summands]
#        with_constants = [summand[1] for summand in filtered if summand[0]]
#        without_constants = [summand[1] for summand in filtered if not summand[0]]
#        # b0 = [sp.Add(*(without_constants))]
#        b0 = [sp.Add(*without_constants)]
#        exprs = with_constants + b0  # = A_exprs | b_expr
#        print("\nall in: expr, summands, filtered, with_constants, without_constants, b0, exprs:", )
#        print( expr, summands, filtered, with_constants, without_constants, b0, exprs)
#        # lambds = [sp.lambdify(model.sym_vars, column, "numpy") for column in with_constants]

#        lambdas = [sp.lambdify(model.sym_vars, column, "sympy") for column in exprs]
#        # lambdas = [sp.lambdify(model.sym_vars, column, "numpy") for column in exprs]

#        # columns = [list(f(*X.T)) for f in lambdas]
#        # print(lambdas[0](X
#        # print('rows', [type(row), row.shape, len(row) for row in X])


#        # return 0
#        columns = [[f(*X[i_row, :]) for i_row in range(X.rows)] for f in lambdas]
#        print("Xp and Yp\n", X[4:14, :6])
#        print("evals", columns)
#        A_b = sp.Matrix(columns).T
#        # A_b = np.array(columns).T
#        print("A_b", A_b)
#        A = A_b[:, :-1]
#        b = Y - A_b[:, -1]
#        # b = Y - A_b[:, [-1]]  # if A_b is numpy
#        print('A b', A, b, A.shape, b.shape, type(A), type(b))
#        #NOT A SOLUTION: type(Matrix(np.array(m, dtype='float')).applyfunc(Integer)[3])

#        # print('X', X)




#            # for summand in summands:
#            #     print(type(summand), summand
#            #     print(f"drop_constant({summand})", drop_constant(summand))


#        A0 = sp.Matrix(
#            [[3, 0 ],
#            [0, 3],
#            [1, 0]])
#        b0 = sp.Matrix([6, 9, 2])
#        print("-<-< exiting model2diophant() --- ")

#        return A, b
#    # return 0  # end here

    for model in ED.models:
        print(model, type(model))
        print(model.expr, type(model.expr),)
        print(model.expr.func, model.expr.args) #.func, model.args)
        # X, Y = model2data(model, X, Y, number_of_terms)
        # print('model2data, X, Y, X.shape, Y.shape', X, Y, X.shape, Y.shape)
        # A, b = model2diophant(model, X, Y)
        # print('res A, b X Y', A, b, X, Y)
        # print('res shape A, b X Y', A.shape, b.shape, X.shape, Y.shape)
    print("returning 0 earlier")
    # return 0

    # 

    # EOf exact ed

    # 1/0
    seq_start = time.perf_counter()
    ED.fit_models()
    seq_cpu_time = time.perf_counter() - seq_start
    print(f"\nParameter fitting for sequence {seq_id} "
          f"took {seq_cpu_time} secconds.")
    print("\nFinal score:")
    for m in ED.models:
        # print(m)
        # print(m.__repr__())
        # print(type(m))
        print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    print("\nFinal score (sorted):")
    for m in ED.models.retrieve_best_models(-1):
        print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    return


# oeis_eq_disco(seq_id, is_direct, order)  # Run only one seq, e.g. the fibonaccis.
# oeis_eq_disco("A000045", is_direct, order)  # Run only one seq, e.g. the fibonaccis.
# Run eq. disco. on all oeis sequences:

start = time.perf_counter()
FIRST_ID = "A000000"
LAST_ID = "A246655"
# last_run = "A002378"

start_id = FIRST_ID
# start_id = "A000045"
end_id = LAST_ID
# end_id = "A000045"

# start_id = "A000041"
# end_id = "A000041"

CATALAN = "A000108"

selection = (
        "A000009", 
        "A000040", 
        "A000045", 
        "A000124", 
        # "A000108", 
        "A000219", 
        "A000292", 
        "A000720", 
        "A001045", 
        "A001097", 
        "A001481", 
        "A001615", 
        "A002572", 
        "A005230", 
        "A027642", 
        )
selection = ("A000045", )
# selection = (CATALAN, )
the_one_sequence = flags_dict.get("--seq_only", None)
selection = selection if the_one_sequence is None else (the_one_sequence,)

csv = csv.loc[:, (start_id <= csv.columns) & (csv.columns <= end_id)]
csv_ids = list(csv.columns)
csv_ids = selection
print(len(selection))


print("Running equation discovery for all oeis sequences, "
        "with these settings:\n"
        # f"=>> is_direct = {is_direct}\n"
        # f"=>> order of equation recursion = {order}\n"
        f"=>> sample_size = {sample_size}\n"
        f"=>> grammar's q and p = {q} and {p}\n"
        f"=>> grammar_template_name = {grammar_template_name}\n"
        # f"=>> generator_settings = {'{'}p_T: {p_T}, p_R: {p_R}{'}'}\n"
        f"=>> generator_settings = {generator_settings}\n"
        f"=>> optimizer = {optimizer}\n"
        f"=>> task_type = {task_type}\n"
        f"=>> timeout = {timeout}\n"
        f"=>> random_seed = {random_seed}\n"
        f"=>> lower_upper_bounds = {lower_upper_bounds}\n"
        f"=>> number of terms in every sequence saved in csv = {terms_count}\n"
        f"=>> number of terms in every sequence actually used = {number_of_terms}\n"
        f"=>> number of all considered sequences = {len(csv_ids)}\n"
        f"=>> list of considered sequences = {csv_ids}"
        )


for seq_id in csv_ids:
    # oeis_eq_disco(seq_id, is_direct, order)
    oeis_eq_disco(seq_id, number_of_terms=number_of_terms, max_order=max_order)
    print(f"\nTotal time consumed by now:{time.perf_counter()-start}\n")
cpu_time = time.perf_counter() - start
print(f"\nEquation discovery for all (chosen) OEIS sequences"
      f" took {cpu_time} secconds, i.e. {cpu_time/60} minutes"
      f" or {cpu_time/3600} hours.")

# def pretty_results(seq_name="fibonacci", is_direct=is_direct, order=order):
#     """Print results in prettier form."""
#     if seq_name =="fibonacci":
#         assert oeis == fibs
#     if seq_name=="fibonacci" and is_direct and order==0:  # i.e. direct fib
#         # is_fibonacci_direct = True
#         # if is_fibonacci_direct:
#         phi = (1+5**(1/2))/2
#         c0, c1 = 1/5**(1/2), np.log(phi)
#         print(f" m c0: {c0}", f"c1:{c1}")
#         model = ED.models[5]  # direct fib
#     elif seq_name=="fibonacci" and not is_direct and order != 0:  # i.e. rec fib
#         model = ED.models[-1]
#     elif seq_name=="primes" and not is_direct and order != 0:  # i.e. rec primes
#         model = ED.models[7]  # primes
#     else:    
#         model = ED.models[-1]  # in general to update
        
#     # an = model.lambdify()
#     an = model.lambdify(*np.round(model.params)) if order != 0 else model.lambdify(*model.params)
#     print("model:", model.get_full_expr())#, "[f(1), f(2), f(3)]:", an(1), an(2), an(3))

#     cache = oeis[:order]  # update this
#     # cache = list(oeis[:order])
#     for n in range(order, len(oeis)):
#         prepend = [n] if is_direct else []
#         append = cache[-order:] if (order != 0) else []
#         cache += [int(np.round(an(*(prepend+append))))]
#         # print(prepend, append, prepend + append, (prepend + append), cache, an)
#     res = cache
#     print(oeis)
#     print(res)
#     error = 0
#     for i, j in zip(res, oeis):
#         print(i,j, i-j, error)
#         error += abs(i-j)
#     print(error)
#     return
# # pretty_results(seq_name=seq_name, is_direct=is_direct, order=order)

