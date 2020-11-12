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
Xs = Xs_c
x0 = Xs[:1]
print(x0, Xs[:4])
# print(np.array(Xs[0]), np.array(Xs[0]).ndim)
# print(type(Xs), Xs.shape, Xs.ndim)
# print(x0, type(x0), x0.shape, x0.ndim)
Ys = np.exp(a*ts)

y = interp1d(ts, Ys, kind='cubic')
# plt.plot(ts,Ys,"r-")
# plt.plot(ts, y(ts),'k:')

def dx(t,x):
    # return a*x + y(t)
    return a*x + cy
n=10
tsa = ts[:n]
# print(tsa)
# Xode_y = odeint(dx, x0, ts)
# Xode_y = solve_ivp(dx, (ts[0],ts[-1]), x0, t_eval = ts)
Xode_y = solve_ivp(dx, (ts[0],ts[-1]), x0, t_eval = tsa)
# plt.plot(ts,Xs,"r-")
len(Xode_y.t), len(Xode_y.y)
# len(Xode_y.sol(ts))
# print(Xode_y.t_events, Xode_y.y_events)
print(ts[0],ts[n-1])
print(Xs[0],Xs[n-1])
# plt.plot(ts, Xode_y,'k--')
# print(Xode_y)
print("ode.t:", Xode_y.t, "ode.y:",  Xode_y.y)
print(Xs[:n])
print(x0)




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
