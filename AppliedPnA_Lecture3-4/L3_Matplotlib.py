import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.exp(x)

plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("f = exp(x)")
plt.show()

