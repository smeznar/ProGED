#%% dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #,  odeint
from scipy.interpolate import interp1d


# # go-away message: odeint works perfectly.
# %% 
# # results:
# # print(Xs-Xode[:,0])
# # print("napaka ODE: %.12f" % sum((Xs-Xode[:,0])**2) )
# # print(f"Napaka ode: {sum((Xs-Xode[:,0])**2)}")
# # print(Xs.shape, Xode.shape, Xode[:,0].shape)
# # print((Xs-Xode[:,0]).shape)


# # %%
# # urejeni exampli:

# # diff eq no.1

# def ode_plot(ts, Xs, dx):
#     Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts)
#     plt.plot(ts,Xs,"r-")
#     plt.plot(ts, Xode.y[0],'k--')
#     error = sum((Xs-Xode.y[0])**2)
#     print(f"Napaka ode: {error}")
#     return error
# # print(ode_plot(ts, Xs_c, Ys_c, dx_c))

# def example_c_data(
#     n = 1000,
#     t_start = 0.45,
#     t_end = 0.87,
#     C = 3.75,
#     cy = 10.34,
#     a = 0.4
#     ):
#     ts = np.linspace(t_start, t_end, n)
#     Xs_c = C*np.exp(a*ts) - cy/a
#     Ys_c = np.ones(ts.shape[0])*cy
#     return ts, Xs_c, Ys_c, cy, a

# def example1c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
#     ts, Xs_c, _, cy, a = example_c_data(n,t1,t2,C,cy,a)
#     def dx_c(t,x):
#         return a*x + cy
#     return ode_plot(ts, Xs_c, dx_c)
# # print(example1c())

# def example2c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
#     ts, Xs_c, Ys_c, _, a = example_c_data(n,t1,t2,C,cy,a)
#     def y_index(t, Ys, ts):
#         """vrne vrednost v stolpcu y, ki pripada casu t"""
#         find_index = ts==t # = [False, False, True, False, ...]
#         index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
#         return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
#     def dx_c(t, x):
#         return a*x + y_index(t, Ys_c, ts)
#     return ode_plot(ts, Xs_c, dx_c)
# # print(example2c())

# def example3c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
#     ts, Xs_c, Ys_c, _, a = example_c_data(n,t1,t2,C,cy,a)
#     y = interp1d(ts, Ys_c, kind='cubic')
#     def dx_c(t, x):
#         return a*x + y(t)
#     return ode_plot(ts, Xs_c, dx_c)
# # print(example3c())

# # print(example1c())
# # print(example2c())
# # print(example3c())



# # diff eq no.2

# def example_tB_data(
#     n = 1000,
#     t_start = 0.45,
#     t_end = 0.87,
#     B = -2.56, # B = 0,
#     a = 0.4
#     ):
#     ts = np.linspace(t_start, t_end, n)
#     Xs = (ts+B)*np.exp(a*ts)
#     Ys = np.exp(a*ts)
#     return ts, Xs, Ys, B, a

# def example4tB(n=1000, t1=0.45, t2=0.87, B=-2.56, a=0.4):
#     ts, Xs, Ys, _, a = example_tB_data(n,t1,t2,B,a)
#     def y_index(t, Ys, ts):
#         """vrne vrednost v stolpcu y, ki pripada casu t"""
#         find_index = ts==t # = [False, False, True, False, ...]
#         index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
#         return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
#     def dx_index(t, x):
#         return a*x + y_index(t, Ys, ts)
#     return ode_plot(ts, Xs, dx_index)
# # print(example4tB())

# def example5tB(n=1000, t1=0.45, t2=0.87, B=-2.56, a=0.4):
#     ts, Xs, Ys, _, a = example_tB_data(n,t1,t2,B,a)
#     y = interp1d(ts, Ys, kind='cubic')
#     def dx(t, x):
#         return a*x + y(t)
#     return ode_plot(ts, Xs, dx)
# # print(example5tB())


# # diff eq no.3

# def example_ab_data(
#     n = 1000,
#     t_start = 0.45,
#     t_end = 0.87,
#     C = 4.21,
#     b = 4, # b=-15.76,
#     a = 0.4
#     ):
#     ts = np.linspace(t_start, t_end, n)
#     Xs = (1/(b-a))*np.exp(b*ts) + C*np.exp(a*ts)
#     Ys = np.exp(b*ts)
#     return ts, Xs, Ys, a

# def example6ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=4, a=0.4):
#     ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
#     def y_index(t, Ys, ts):
#         """vrne vrednost v stolpcu y, ki pripada casu t"""
#         find_index = ts==t # = [False, False, True, False, ...]
#         index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
#         return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
#     def dx_index(t, x):
#         return a*x + y_index(t, Ys, ts)
#     return ode_plot(ts, Xs, dx_index)
# # print(example6ab())  # pomankljivost indexa
# # print(example6ab(b=-15.76))  # delujoc
# # print(example6ab(b=4))  # to je default nastavitev b-ja.

# def example7ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=4, a=0.4):
#     ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
#     y = interp1d(ts, Ys, kind='cubic')
#     def dx(t, x):
#         return a*x + y(t)
#     return ode_plot(ts, Xs, dx)
# # print(example7ab())
# # print(example7ab(b=4)) # = original (interpol premaga index)
# # print(example7ab(b=-30, a=-40))  # primer ko "ne deluje"
# # # dodatni primeri (isti) za arhiv:
# # # # print(example7ab(n=1000, t1=0.45, t2=0.87, C=-1*24.21, b=-34.56, a=-40.4))  
# # # # print(example7ab(n=1000, t1=0.45, t2=0.87, C=-1.21, b=-23.56, a=-34.4))

# print(example6ab())
# # print(example7ab(t1=0.95, t2=0.87))


# ## begin implement:     # # # #

# def model_ode_error (model, params, T, X, Y):
#     """Defines mean squared error as the error metric.
#         Input:
#         - T je 
#     """
#     # testY = model.evaluate(X, *params)
#     odeY = ode(model, params, T, X, y0=Y[0])
#     res = np.mean((Y-odeY)**2)
#     if np.isnan(res) or np.isinf(res) or not np.isreal(res):
# #        print(model.expr, model.params, model.sym_params, model.sym_vars)
#         return 10**9
#     return res

# def ode(model, params, T, X, y0):
#     # Check shape(T)[0] = shape(X)[0]: 
#     #     "DimError: number of samples in T and X does not match.
#     # Nepotrebno:
#     # # Check T = sort(T) and T[0]<T[-1] : 
#     # # "Times error: ODEs require ascending order of times column T."

#     X = interpNd(T, X, kind='cubic')
#     def dy_dt(t, y):  # \frac{dy}{dt}
#         # \dot{y} = a*y + X(t)
#         model.evaluate(["y"]+X(t))  # =[y,X(t)] =[y,X1(t),X2(t),...] 
#         # return a*y + X(t)

#     Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T)
#     # plt.plot(T,Y,"r-")
#     # plt.plot(T, Yode.y[0],'k--')
#     return Yode.y[0]

# def ode1d(model, params, T, X, y0):
#     # Check shape(T)[0] = shape(X)[0]: 
#     #     "DimError: number of samples in T and X does not match.
#     # Nepotrebno:
#     # # Check T = sort(T) and T[0]<T[-1] : 
#     # # "Times error: ODEs require ascending order of times column T."

#     X = interp1d(T, X, kind='cubic')  # testiral, zgleda da dela.
#     def dy_dt(t, y):  # \frac{dy}{dt}
#         # \dot{y} = a*y + X(t)
#         # model.evaluate(np.array([[y]+X(t)]), *params)  # if X is vector(array)
#         model.evaluate(np.array([[y]+[X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
#         # return a*y + X(t)

#     Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T)
#     # plt.plot(T,Y,"r-")
#     # plt.plot(T, Yode.y[0],'k--')
#     return Yode.y[0]

# # for real:

# def ode1d(model, params, T, X, y0):
#     if T.shape[0] != X.shape[0]: 
#         raise IndexError("Number of samples in T and X does not match.")
#     X = interp1d(T, X, kind='cubic')  # testiral, zgleda da dela.
#     def dy_dt(t, y):  # \frac{dy}{dt}
#         # model.evaluate(np.array([[y]+X(t)]), *params)  # if X is vector(array)
#         return model.evaluate(np.array([[y]+[X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
#     Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T)
#     # plt.plot(T, Y, "r-")
#     # plt.plot(T, Yode.y[0],'k--')
#     return Yode.y[0]

# def model_ode_error (model, params, T, X, Y):
#     """Defines mean squared error of solution to differential equation
#     as the error metric.
#         Input:
#         - T is column of times at which samples in X and Y happen.
#         - X are columns that do not contain variables that are derived.
#         - Y is column containing variable that is derived.
#     """
#     odeY = ode1d(model, params, T, X, y0=Y[0])
#     res = np.mean((Y-odeY)**2)
#     if np.isnan(res) or np.isinf(res) or not np.isreal(res):
# #        print(model.expr, model.params, model.sym_params, model.sym_vars)
#         return 10**9
#     return res





# # error = np.mean((Xs-Xode.y[0])**2)
# # error1 = np.mean(Xs-Xode.y[0])
# # print(f"Napaka ode: {error} in {error1}")



# # Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts)
# # # plt.plot(ts,Xs,"r-")
# # # plt.plot(ts, Xode.y[0],'k--')
# # error = sum((Xs-Xode.y[0])**2)
# # print(f"Napaka ode: {error}")

# # %%

def ode1d(model, params, T, X_data, y0):
    if T.shape[0] != X_data.shape[0]: 
        raise IndexError("Number of samples in T and X does not match.")
    X = interp1d(T, X_data, kind='cubic')  # testiral, zgleda da dela.
    def dy_dt(t, y):  # \frac{dy}{dt}
        # model.evaluate(np.array([[y]+X(t)]), *params)  # if X is vector(array)
        # b = np.array([[y]+[X(t)q]])
        # b = np.array([[y[0], X(t)]])
        # print(b, y,X(t),y.shape, X(t).shape, "in dy_dt")
        # b = np.array([[1,2]])
        # return 1
        return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T)
    # plt.plot(T, Y, "r-")
    # plt.plot(T, Yode.y[0],'k--')
    return Yode.y[0]
    # return dy_dt(11,3)



np.random.seed(2)
import sympy as sp
from generate import generate_models    
from generators.grammar import GeneratorGrammar
from pyDOE import lhs

from parameter_estimation import *
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
X = interp1d(ts, x1, kind='cubic')
odey = ode1d(m, m.params, ts, x1, y[0])
plt.plot(ts,X[:,1],"r-")
plt.plot(ts, odey,'k--')
print(ts.shape)

    
def ode(model, params, T, X_data, y0):
    """Solves ode defined by model.
        Input worth mentioning:
        - y0 is array (1-dim) of initial value of vector function y(t)
        i.e. y0 = y(T[0]) = [y1(T[0]), y2(T[0]), y3(T[0]),...]."""
    if T.shape[0] != X_data.shape[0]: 
        raise IndexError("Number of samples in T and X does not match.")
    X = interp1d(T, X_data, kind='cubic')  # testiral, zgleda da dela.
    if X(T[0]).ndim == 0:  # in case X_data is one dimensional array
            X = lambda t: X(t).reshape(1)  # convert 0-dim. number to 1-dim. array
            print("Opazil, da je X 0-dim")
    lamb_expr = sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
    def dy_dt(t, y):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
        # return model.evaluate(np.array([[y[0], X(t)]]), *params)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
        b = np.array([np.concatenate((y,X(t)))]) # popravi
        print("b:", b)
        # return lamb_expr(*np.array([[y[0], X(t)]]).T)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
        return lamb_expr(*b.T)  # =[y,X(t)] =[y,X1(t),X2(t),...] 
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T)
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), t_eval=T, atol=0) # spremeni v y0
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), method='LSODA', t_eval=T, atol=0) 
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), np.array([y0]), method='LSODA', t_eval=T) 
    print(f"Status: {Yode.status}, Success: {Yode.success}, message: {Yode.message}.")
    return Yode.y[0]

# X = interp1d(T, X_data, kind='cubic') 