from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, rand
space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
    },
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
            [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    ])
import hyperopt.pyll.stochastic
# for i in range(10):
#     print(hyperopt.pyll.stochastic.sample(space))

lower_bound = -100
upper_bound = 50
p0 = (13.45, 15.1, 16.834)
# proged_space = {'C'+str(i): hp.randint('C_in'+str(i), lower_bound, upper_bound)
proged_space = [hp.randint('C'+str(i), lower_bound, upper_bound)
                    for i in range(len(p0))]
prspace = proged_space
# prspace = {'nek': hp.randint('lab', 4)}
# prspace = hp.randint('lab', 4)
# print(prspace)
# for i in range(10):
#     print(hyperopt.pyll.stochastic.sample(prspace))
def objectiv(args):
    return (args[0]**2 + args[1]**2)**(1/2)
    # return args**2
    # return args['nek']**2
best = fmin(fn=objectiv, algo=rand.suggest, timeout=3, space=prspace)#, max_evals=2000)
# best = fmin(fn=objectiv, algo=tpe.suggest, space=prspace, max_evals=100)
print(best)

# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
# best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

# print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
# print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}