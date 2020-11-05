# dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#%% 
n= 1000
ts = np.linspace(0.45,0.87,n)
a = 0.4
cy = 10.34
C = 3.75
Xs = C*np.exp(a*ts) - cy/a
Ys = np.ones(ts.shape[0])*cy
x0 = Xs[0]
print(Xs, Ys)
# plt.plot(ts,Xs)

def dx(x,t):
    # return a*X + Ys[t] 
    return a*x + cy 
Xode = odeint(dx, x0, ts)
plt.plot(ts,Xs,"r-")
plt.plot(ts, Xode,'b--')

# go-away message: odeint works perfectly.
# %% 
# results:
# print(Xs-Xode[:,0])
print("napaka ODE: %.12f" % sum((Xs-Xode[:,0])**2) )
print(f"Napaka ode: {sum((Xs-Xode[:,0])**2)}")
print(Xs.shape, Xode.shape, Xode[:,0].shape)
print((Xs-Xode[:,0]).shape)





# %%
