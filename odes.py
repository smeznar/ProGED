#%% dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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
B = 0
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
