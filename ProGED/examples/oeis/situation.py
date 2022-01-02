
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



np.random.seed(0)
has_titles = 1
csv = pd.read_csv('oeis_selection.csv')[has_titles:]

# selection.py:
def abs_max(seq):
    return max([abs(int(term)) for term in seq])

ab = [abs_max(csv[col]) for col in csv.columns]
print(ab)
poz = [i for i in ab if i < 0 ]
notbig = [i for i in ab if i < 10**16 ]
print(poz, len(poz))
print('notbig', len(notbig), notbig)
print('maxi', [f"{maxi:e}" for maxi in notbig])
print(max(ab) > 10**16)
print(f"{max(ab):e}")



