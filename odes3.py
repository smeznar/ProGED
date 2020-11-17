#%% 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp # nic vec funkcija `odeint`
from scipy.interpolate import interp1d # interpolacija spline
#%% 

def example_ab_data(
    n = 1000,
    t_start = 0.45,
    t_end = 0.87,
    C = 4.21,
    b = -30, 
    # b=4,
    a = -40 
    # a=0.4
    ):
    ts = np.linspace(t_start, t_end, n)
    Xs = (1/(b-a))*np.exp(b*ts) + C*np.exp(a*ts)
    Ys = np.exp(b*ts)
    return ts, Xs, Ys, a
ts, Xs, Ys, a = example_ab_data()

y = interp1d(ts, Ys, kind='cubic')  # interpolacija spline (cubic)
def dx(t, x):
    return a*x + y(t)

Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts) # brez tolerance -> neuspesno
plt.plot(ts,Xs,"r-")  # analiticna (na roke) resitev
plt.plot(ts, Xode.y[0],'b--')  # ne izgleda kot resitev

## resitev z atol absolutno toleranco:
Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts, atol=10**(-9))
plt.plot(ts, Xode.y[0],'k--')  # izgleda kot resitev
print(f"Napaka ode: {sum((Xs-Xode.y[0])**2)} in {sum(Xs-Xode.y[0])}")
#%%












# neuspesno z indexacijo (brez interpolaicje):

def example6ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=-30, a=-40):
    ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
    
    def y_index(t, Ys, ts):
        """vrne vrednost v stolpcu y, ki pripada casu t"""
        find_index = ts==t # = [False, False, True, False, ...]
        index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
        return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
    
    def dx_index(t, x):
        return a*x + y_index(t, Ys, ts)
    
    Xode2 = solve_ivp(dx_index, (ts[0],ts[-1]), Xs[:1], t_eval=ts)#, atol=10**(-13))
    
    plt.plot(ts,Xs,"r-") # analiticna resitev
    plt.plot(ts, Xode2.y[0],'b--')  # numericna resitev
    print(f"Napaka ode: {sum((Xs-Xode2.y[0])**2)} in {sum(Xs-Xode2.y[0])}")
    return None

example6ab(b=4, a=0.4) # Se vidi odstopanje (index ne deluje).
# example6ab() # ( =nepomembno: ) isti primer kot pri interpolaciji za primerjavo