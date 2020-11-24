import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
n= 1000
ts = np.linspace(0.45,0.87,n)
a = 0.4
# B = -2.56
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
# plt.plot(ts,Xs,"r-")
# plt.plot(ts, Xode_y,'k--')

# primer 1: 
cy = 10.34
C = 3.75
Xs_c = C*np.exp(a*ts) - cy/a
Ys_c = np.ones(ts.shape[0])*cy
x0_c = Xs_c[0]
def dx_c(x,t):
    # return a*x + y(t,Ys)
    return a*x + cy 
Xode_c = odeint(dx_c, x0_c, ts)
# plt.plot(ts,Xs_c,"r-")
# plt.plot(ts, Xode_c,'k--')



#########################################################

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

