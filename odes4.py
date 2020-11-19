#%% dataset
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #,  odeint
from scipy.interpolate import interp1d


# # go-away message: odeint works perfectly.
# %% 

# def ode_plot(ts, Xs, dx):
#     Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts)
#     plt.plot(ts,Xs,"r-")
#     plt.plot(ts, Xode.y[0],'k--')



np.random.seed(2)

from generate import generate_models    
from generators.grammar import GeneratorGrammar
from pyDOE import lhs






def ode1d(model, params, T, X_data, y0):
    if T.shape[0] != X_data.shape[0]: 
        raise IndexError("Number of samples in T and X does not match.")
    X = interp1d(T, X_data, kind='cubic')  # testiral, zgleda da dela.
    # def dy_dt(t, y):  # \frac{dy}{dt}
    #     # model.evaluate(np.array([[y]+X(t)]), *params)  # if X is vector(array)
    #     # b = np.array([[y]+[X(t)q]])
    #     # b = np.array([[y[0], X(t)]])
    #     # print(b, y,X(t),y.shape, X(t).shape, "in dy_dt")
    #     # b = np.array([[1,2]])
    #     # return 1
    #     return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    lamb_expr = sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
    def dy_dt(t, y):  # \frac{dy}{dt}
        # return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
        return lamb_expr(*np.array([[y[0], X(t)]]).T)
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T)
    # plt.plot(T, Y, "r-")
    # plt.plot(T, Yode.y[0],'k--')
    return Yode.y[0]
    # return dy_dt(11,3)


# from parameter_estimation import *

def testf (x):
    return 3*x[:,0]*x[:,1]**2 + 0.5

X = lhs(2, 10)*5
y = testf(X)


grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                            T -> 'C' [0.6] | T "*" V [0.4]
                            V -> 'x' [0.5] | 'y' [0.5]""")
symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
N = 10
models = generate_models(grammar, symbols, strategy_parameters = {"N":1})
m = models[-1]
print(m)


x1 = X[:,0]
ts = np.linspace(10, 50, X.shape[0])
Xt = interp1d(ts, x1, kind='cubic')


odey = ode1d(m, m.params, ts, x1, y[0])
odey2 = ode1d(m, m.params, ts, x1, y[0])
# plt.plot(ts,X[:,1],"r-")
# plt.plot(ts, odey,'k--')

print(ts.shape)

    
def ode(model, params, T, X_data, y0):
    """Solves ode defined by model.
        Input specs:
        - y0 is array (1-dim) of initial value of vector function y(t)
        i.e. y0 = y(T[0]) = [y1(T[0]), y2(T[0]), y3(T[0]),...].
        - X_data is 2-dim array (matrix) i.e. X = [X[0,:], X[1,:],...].
        - T is (1-dim) array, i.e. of shape (N,)
    """
    if T.shape[0] != X_data.shape[0]: 
        raise IndexError("Number of samples in T and X does not match.")
    ### 1-dim version currently: ###
    # X = interp1d(T, X_data, kind='cubic') # orig
    # Xt = interp1d(T, X_data.T[0], kind='cubic')  
    X = interp1d(T, X_data.T[0], kind='cubic')  # 1 -dim
    # if X(T[0]).ndim == 0:  # in case X_data is one dimensional array
    ### 1-dim version currently: ###
    # X = lambda t: np.array([Xt(t)])  # convert 0-dim. number to 1-dim. array
        # print("Opazil, da je X 0-dim")
    lamb_expr = sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
    def dy_dt(t, y):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
        # return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
        # b = np.array([np.concatenate((y,X(t)))]) # popravi
        # 1-dim interpol:
        b = np.array([np.concatenate((y,np.array([X(t)])))]) # popravi
        # print("b:", b)
        # return lamb_expr(*np.array([[y[0], X(t)]]).T)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
        return lamb_expr(*b.T)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T)

    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T, atol=0) # spremeni v y0
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), method='LSODA', t_eval=T, atol=0) 
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), method='LSODA', t_eval=T) 
    print(f"Status: {Yode.status}, Success: {Yode.success}, message: {Yode.message}.")
    return Yode.y[0]

X1 = np.array([x1]).T
print(x1.shape, x1.ndim)
print(X1.shape, X1.ndim)
ode2 = ode(m, m.params, ts, X1, np.array([y[0]]))
ode3 = ode(m, m.params, ts, X1, np.array([y[0]]))

plt.plot(ts,X[:,1],"r-")
plt.plot(ts, odey,'g-')
plt.plot(ts, ode2,"k--")
def simerr(y1,y2): return np.mean((y1-y2)**2)
print(np.mean((ode2-odey)**2))
print(simerr(ode2,odey))
print(simerr(ode2,ode3))
print(simerr(odey2,odey))


# print(X)
# print(x1, "x1")
# print(x1.T,)
# print(X[:,0])
# print("po X")
# X = interp1d(ts, x1, kind='cubic')  # testiral, zgleda da dela.
# # if X(T[0]).ndim == 0:  # in case X_data is one dimensional array
# # plt.plot(ts,x1,"r-")
# # plt.plot(ts,X(ts),"k--")

# Xt = lambda t: np.array([X(t)])
# # plt.plot(ts,Xt(ts)[0],"b--")
# print(x1)
# print(X(30))
# print(Xt(30))
# print(Xt(ts))

# # # izgled X in y dataset
# a = np.array([np.array([j*10+i for i in range(j,j+4)]) for j in range(1,6)])
# print(a)
# # X2 = a[:,2:3] # vsi data set imputi tako X kot y bodo te oblike.
# # # tj. oblike shape=(N,D) = [[x1,..,xD],..,[x1_N,..,xD_N]]
# # print(X2)
# X2 = a[:,2]
# X3 = a[:,[2]]
# print(X2)
# print(X3)
# # X2 = a[:,(2-1):2]
# # print(X2)
# # print(a[:,0:1])
# # print(a[:,[0]])
# print(np.array([X2]))
# print(np.array([X2]).T)
# # print(a[0,:])
# # print(np.linspace(1,50,4), np.linspace(1,50,4).shape, np.linspace(1,50,4).ndim)

