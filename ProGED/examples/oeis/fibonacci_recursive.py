import numpy as np
import pandas as pd
from scipy.optimize import brute, shgo, rosen, dual_annealing

from ProGED.equation_discoverer import EqDisco
# from ProGED.parameter_estimation import integer_brute_fit, DE_fit, shgo_fit, DAnnealing_fit

csv = pd.read_csv("oeis_selection.csv", nrows=50)
fibs = list(csv["A000045"])  # fibonacci = A000045
oeis = fibs
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
seq = np.array(oeis)
# seq = np.array(oeis[:40])
# print(seq.shape, seq.dtype, seq)

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
    indexes = np.fromfunction((lambda i,j:i+j), (n-order, order+1), dtype=int)
    first_column = indexes[:, [0]]
    if direct:
        return np.hstack((first_column, seq[indexes]))
    return seq[indexes]

#######main#settings#####################################
order, is_direct = 2, False  # recursive
# order, is_direct = 0, True  # direct
seq_name = "fibonacci"
data = grid(order, seq, is_direct)
#########################################################

variable_names_ = [f"an_{i}" for i in range(order, 0, -1)] + ["an"]
variable_names = ["n"]+variable_names_ if is_direct else variable_names_
variables = [f"'{an_i}'" for an_i in variable_names[:-1]]
# print(variable_names)
# print(variables)
# print(data.shape, type(data), data)

p_T = [0.4, 0.6]  # default settings, does nothing
p_R = [0.6, 0.4]
if (seq_name, order, is_direct) == ("fibonacci", 2, False):
    p_T = [0.4, 0.6]  # for rec fib
    p_R = [0.9, 0.1]

np.random.seed(0)
# seed 0 , size 20 (16)
# seed3 size 15 an-1 + an-2 + c3 rec 
ED = EqDisco(data = data,
            task = None,
            target_variable_index = -1,
            # variable_names=["an_2", "an_1", "an"],
            variable_names=variable_names,
            sample_size = 16,  # for recursive
            # sample_size = 10,  # for direct fib
            # sample_size = 1,
            verbosity = 0,
            # verbosity = 2,
            generator = "grammar", 
            generator_template_name = "polynomial",
            # generator_settings={"variables": ["'an_2'", "'an_1'"],
            generator_settings={"variables": variables,
                                 "p_T": p_T, "p_R": p_R
                                 },
            estimation_settings={"verbosity": 2,
             # "task_type": "algebraic",
             "task_type": "oeis",
            # "task_type": "oeis_recursive_error",  # bad idea
             "lower_upper_bounds": (-1000, 1000),  # najde (1001 pa ne) rec fib
            #  "lower_upper_bounds": (-100, 100),  # ne najde DE
            #  "lower_upper_bounds": (-25, 25),  # DA ne najde
            #  "lower_upper_bounds": (-11, 11),  # shgo limit
            #  "lower_upper_bounds": (-14, 14),  # DA dela
            # "lower_upper_bounds": (-2, 2),  # for direct
            # "lower_upper_bounds": (-4, 4),  # for direct fib
            # "lower_upper_bounds": (-5, 5), 
            # "lower_upper_bounds": (-8, 8),  # long enough for brute
            # "optimizer": DE_fit_metamodel,
            # "optimizer": DE_fit,
            # "optimizer": integer_brute_fit,
            # "optimizer": shgo_fit,
            # "optimizer": DAnnealing_fit,
            }
            )

ED.generate_models()
print(ED.models)
ED.fit_models()
print("\nFinal score:")
for m in ED.models:
    print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")

def pretty_results(seq_name="fibonacci", is_direct=is_direct, order=order):
    """Print results in prettier form."""
    if seq_name =="fibonacci":
        assert oeis == fibs
    if seq_name=="fibonacci" and is_direct and order==0:  # i.e. direct fib
        # is_fibonacci_direct = True
        # if is_fibonacci_direct:
        phi = (1+5**(1/2))/2
        c0, c1 = 1/5**(1/2), np.log(phi)
        print(f" m c0: {c0}", f"c1:{c1}")
        model = ED.models[5]  # direct fib
    elif seq_name=="fibonacci" and not is_direct and order != 0:  # i.e. rec fib
        model = ED.models[-1]
    elif seq_name=="primes" and not is_direct and order != 0:  # i.e. rec primes
        model = ED.models[7]  # primes
    # model = ED.models[-1]  # in general to update
        
    # an = model.lambdify()
    an = model.lambdify(*np.round(model.params)) if order != 0 else model.lambdify(*model.params)

    cache = oeis[:order]  # update this
    # cache = list(oeis[:order])
    for n in range(order, len(oeis)):
        prepend = [n] if is_direct else []
        append = cache[-order:] if (order != 0) else []
        cache += [int(np.round(an(*(prepend+append))))]
        # print(prepend, append, prepend + append, (prepend + append), cache, an)
    res = cache
    print(oeis)
    print(res)
    error = 0
    for i, j in zip(res, oeis):
        print(i,j, i-j, error)
        error += abs(i-j)
    print(error)
    return
pretty_results(seq_name="fibonacci", is_direct=is_direct, order=order)

# print( "\n\n\n -------------output.txt bounds--------------\n\n\n")
# for i in range(12, 100):
#     print(i, "=i")
#     np.random.seed(0)
#     ED = EqDisco(
#         data = data,
#         task = None,
#         target_variable_index = -1,
#         variable_names=["an_2", "an_1", "an"],
#         sample_size = 16,
#         verbosity = 0,
#         generator = "grammar",
#         generator_template_name = "polynomial",
#         generator_settings={"variables": ["'an_2'", "'an_1'"], "p_T": p_T, "p_R": p_R},
#         estimation_settings={"verbosity": 0,
#             "task_type": "oeis",
#             "lower_upper_bounds": (-i, i),
#             }
#         )
#     ED.generate_models()
#     ED.fit_models()
#     m = ED.models[-1] 
#     print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
