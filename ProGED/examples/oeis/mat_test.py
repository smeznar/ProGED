import numpy as np


def f(x):
    return 2.0 * (x + 0.3)

np.random.seed(1)

X = np.linspace(-1, 1, 1000).reshape(-1,1)
Y = f(X).reshape(-1,1)
data = np.hstack((X,Y))


from ProGED import EqDisco

#ED = EqDisco(data=data, sample_size=100, verbosity=1)
ED = EqDisco(data=data, sample_size=100, verbosity=1, variable_names = ["x", "f"],
            target_variable_index = -1,
)

print(ED.generate_models())
print(ED.fit_models())
print(ED.get_results())
print(ED.get_stats())
