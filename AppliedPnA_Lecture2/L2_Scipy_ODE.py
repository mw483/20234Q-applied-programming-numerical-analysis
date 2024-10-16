import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# define function for ODE
def func_dydt(y, t):
    dydt = -y**2 + np.sin(y) + np.exp(2*y + 6*y**2)
    return dydt

t_list = np.linspace(0.0, 10.0, 100)
y_init = 1.0  # initial value
y_list = odeint(func_dydt, y_init, t_list)

# visualization
fig, ax = plt.subplots()
ax.plot(t_list, y_list)
plt.show()
