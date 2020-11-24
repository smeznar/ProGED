import numpy as np
from odes import example_tB_data # import datasets T, X and Y

T, Xs, Ys, _, a = example_tB_data() # n,t1,t2,B,a)
X = np.array([Xs]).T; Y = np.array([Ys]).T
print(T.shape, X.shape, Y.shape)