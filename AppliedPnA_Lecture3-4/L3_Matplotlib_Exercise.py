import matplotlib.pyplot as plt
import random

import numpy as np

x = np.linspace(-2*np.pi, 2*np.pi, 100)
r = random.randint(0,1)
y1 = np.cos(x)
y2 = np.cos(x) + 0.5*(r-0.5)

plt.plot(x,y1, label = "cos(x)")
plt.plot(x, y2, label = "cos(x) + 0.5*(r-0.5)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

