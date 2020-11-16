#%% dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d

#%% 
n= 1000
ts = np.linspace(0.45,0.87,n)
a = 0.4
cy = 10.34
C = 3.75
B = -2.56
Xs_c = C*np.exp(a*ts) - cy/a
Ys_c = np.ones(ts.shape[0])*cy
# Xs = (ts+B)*np.exp(a*ts)
# Ys = np.exp(a*ts)
x0_c = Xs_c[0]
# print(Xs, Ys)
# plt.plot(ts,Xs)

def dx(x,t):
    # return a*X + Ys[t] 
    return a*x + cy 
Xode_c = odeint(dx, x0_c, ts)
plt.plot(ts,Xs_c,"r-")
plt.plot(ts, Xode_c,'b--')

# go-away message: odeint works perfectly.
# %% 
# results:
# print(Xs-Xode[:,0])
print("napaka ODE: %.12f" % sum((Xs-Xode[:,0])**2) )
print(f"Napaka ode: {sum((Xs-Xode[:,0])**2)}")
print(Xs.shape, Xode.shape, Xode[:,0].shape)
print((Xs-Xode[:,0]).shape)


#%%
from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
# %%


# %%

# print(ts[0:10])
# tq = ts[4]
# print(tq)
# print((ts==tq)[:10])
# print(ts[ts==tq])
# print((ts==tq).argmax())

# a=[1,2,3]
# a.index
n= 1000
ts = np.linspace(0.45,0.87,n)
a = 0.4
B = -2.56
# B = 0
Xs = (ts+B)*np.exp(a*ts)
x0 = Xs[0]
Ys = np.exp(a*ts)
Ts = ts # stolpec casov
def y(t, Ys):
    """vrne vrednost v stolpcu y, ki pripada casu t"""
    find_index = Ts==t # = [False, False, True, False, ...]
    index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
    return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
def dx(x,t):
    return a*x + y(t,Ys)
    # return a*x + cy 
Xode_y = odeint(dx, x0, ts)
plt.plot(ts,Xs,"r-")
plt.plot(ts, Xode_y,'k--')

# %%
# solve_ivp cell:
n= 1000
ts = np.linspace(0.45,0.87,n)
C = 3.75
cy = 10.34
a = 0.4
B = -2.56
# B = 0
Xs_c = C*np.exp(a*ts) - cy/a
Ys_c = np.ones(ts.shape[0])*cy
Xs = (ts+B)*np.exp(a*ts)
# Xs = Xs_c
x0 = Xs[:1]
Ys = np.exp(a*ts)
y = interp1d(ts, Ys, kind='cubic')
# plt.plot(ts,Ys,"r-")
# plt.plot(ts, y(ts),'k:')

def y_index(t, Ys, ts):
    """vrne vrednost v stolpcu y, ki pripada casu t"""
    find_index = ts==t # = [False, False, True, False, ...]
    index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
    return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
def dx(t, x):
    # return a*x + cy
    # return a*x + y_index(t, Ys_c, ts)
    # return a*x + y_index(t, Ys, ts)
    return a*x + y(t)
# n=-1
# ts = ts[:n]
# print(tsa)
# Xode_y = odeint(dx, x0, ts)
# Xode_y = solve_ivp(dx, (ts[0],ts[-1]), x0, t_eval = ts)
Xode_y = solve_ivp(dx, (ts[0],ts[-1]), x0, t_eval = ts)
# plt.plot(ts,Xs[:n],"r-")
plt.plot(ts,Xs,"r-")
len(Xode_y.t), len(Xode_y.y)
# len(Xode_y.sol(ts))
# print(Xode_y.t_events, Xode_y.y_events)
# print(ts[0],ts[n-1])
# print(Xs[0],Xs[n-1])
plt.plot(ts, Xode_y.y[0],'k--')
# print(Xode_y)
# print("ode.t:", Xode_y.t, "ode.y:",  Xode_y.y)
# print(Xs[:n])
# print(Xs[:n]-Xode_y.y[0])
print(x0)


# %%
# urejeni exampli:

# diff eq no.1

def ode_plot(ts, Xs, dx):
    Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts)
    plt.plot(ts,Xs,"r-")
    plt.plot(ts, Xode.y[0],'k--')
    error = sum((Xs-Xode.y[0])**2)
    print(f"Napaka ode: {error}")
    return error
# print(ode_plot(ts, Xs_c, Ys_c, dx_c))

def example_c_data(
    n = 1000,
    t_start = 0.45,
    t_end = 0.87,
    C = 3.75,
    cy = 10.34,
    a = 0.4
    ):
    ts = np.linspace(t_start, t_end, n)
    Xs_c = C*np.exp(a*ts) - cy/a
    Ys_c = np.ones(ts.shape[0])*cy
    return ts, Xs_c, Ys_c, cy, a

def example1c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
    ts, Xs_c, _, cy, a = example_c_data(n,t1,t2,C,cy,a)
    def dx_c(t,x):
        return a*x + cy
    return ode_plot(ts, Xs_c, dx_c)
# print(example1c())

def example2c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
    ts, Xs_c, Ys_c, _, a = example_c_data(n,t1,t2,C,cy,a)
    def y_index(t, Ys, ts):
        """vrne vrednost v stolpcu y, ki pripada casu t"""
        find_index = ts==t # = [False, False, True, False, ...]
        index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
        return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
    def dx_c(t, x):
        return a*x + y_index(t, Ys_c, ts)
    return ode_plot(ts, Xs_c, dx_c)
# print(example2c())

def example3c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
    ts, Xs_c, Ys_c, _, a = example_c_data(n,t1,t2,C,cy,a)
    y = interp1d(ts, Ys_c, kind='cubic')
    def dx_c(t, x):
        return a*x + y(t)
    return ode_plot(ts, Xs_c, dx_c)
# print(example3c())

# print(example1c())
# print(example2c())
# print(example3c())



# diff eq no.2

def example_tB_data(
    n = 1000,
    t_start = 0.45,
    t_end = 0.87,
    B = -2.56, # B = 0,
    a = 0.4
    ):
    ts = np.linspace(t_start, t_end, n)
    Xs = (ts+B)*np.exp(a*ts)
    Ys = np.exp(a*ts)
    return ts, Xs, Ys, B, a

def example4tB(n=1000, t1=0.45, t2=0.87, B=-2.56, a=0.4):
    ts, Xs, Ys, _, a = example_tB_data(n,t1,t2,B,a)
    def y_index(t, Ys, ts):
        """vrne vrednost v stolpcu y, ki pripada casu t"""
        find_index = ts==t # = [False, False, True, False, ...]
        index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
        return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
    def dx_index(t, x):
        return a*x + y_index(t, Ys, ts)
    return ode_plot(ts, Xs, dx_index)
# print(example4tB())

def example5tB(n=1000, t1=0.45, t2=0.87, B=-2.56, a=0.4):
    ts, Xs, Ys, _, a = example_tB_data(n,t1,t2,B,a)
    y = interp1d(ts, Ys, kind='cubic')
    def dx(t, x):
        return a*x + y(t)
    return ode_plot(ts, Xs, dx)
# print(example5tB())


# diff eq no.3

def example_ab_data(
    n = 1000,
    t_start = 0.45,
    t_end = 0.87,
    C = 4.21,
    b = 4, # b=-15.76,
    a = 0.4
    ):
    ts = np.linspace(t_start, t_end, n)
    Xs = (1/(b-a))*np.exp(b*ts) + C*np.exp(a*ts)
    Ys = np.exp(b*ts)
    return ts, Xs, Ys, a

def example6ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=4, a=0.4):
    ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
    def y_index(t, Ys, ts):
        """vrne vrednost v stolpcu y, ki pripada casu t"""
        find_index = ts==t # = [False, False, True, False, ...]
        index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
        return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
    def dx_index(t, x):
        return a*x + y_index(t, Ys, ts)
    return ode_plot(ts, Xs, dx_index)
# print(example6ab())  # pomankljivost indexa
# print(example6ab(b=-15.76))  # delujoc
# print(example6ab(b=4))  # to je default nastavitev b-ja.

def example7ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=4, a=0.4):
    ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
    y = interp1d(ts, Ys, kind='cubic')
    def dx(t, x):
        return a*x + y(t)
    return ode_plot(ts, Xs, dx)
# print(example7ab())
# print(example7ab(b=4)) # = original (interpol premaga index)
# print(example7ab(b=-30, a=-40))  # primer ko "ne deluje"
# # dodatni primeri (isti) za arhiv:
# # # print(example7ab(n=1000, t1=0.45, t2=0.87, C=-1*24.21, b=-34.56, a=-40.4))  
# # # print(example7ab(n=1000, t1=0.45, t2=0.87, C=-1.21, b=-23.56, a=-34.4))


# print(example7ab())
# print(example6ab())
n=1000; t1=0.45; t2=0.87; b=-30; a=-40
# b=-23.56; a=-34.4
# print(example7ab(n, t1, t2, b=b, a=a))
ts, Xs, Ys, a = example_ab_data(n, t_start=t1, t_end=t2, b=b, a=a)
# plt.plot(ts, Ys, "g-")
y = interp1d(ts, Ys, kind='cubic')
# plt.plot(ts, y(ts), "k--")
def dx(t, x):
    return a*x + y(t)
plt.plot(ts,Xs,"r-")
Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts) # neuspesno
# plt.plot(ts, Xode.y[0],'w--')
plt.plot(ts, Xode.y[0],'b--')
# resitev z atol:
Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts, atol=10**(-9))
# Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts, atol=0)
plt.plot(ts, Xode.y[0],'k--')
# plt.plot(ts,Xs,"r-")
# plt.plot(ts, Xode.y[0],'k--')
error = sum((Xs-Xode.y[0])**2)
error1 = sum(Xs-Xode.y[0])
print(f"Napaka ode: {error} in {error1}")


# def model_error (model, params, X, Y):
#     """Defines mean squared error as the error metric."""
#     testY = model.evaluate(X, *params)
#     res = np.mean((Y-testY)**2)
#     if np.isnan(res) or np.isinf(res) or not np.isreal(res):
# #        print(model.expr, model.params, model.sym_params, model.sym_vars)
#         return 10**9
#     return res