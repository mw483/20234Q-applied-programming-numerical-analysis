# Gradient Descent
# 1. Start with an initial position
# 2. Calculate gradients at the position
# 3. Stop if the gradients is enough low
# 4. Update position using the gradients
# 5. Go back to step 2

# Find the peak of the function below:
# f(x1,x2) = 2*(cos(2**x1 - x2**2 + 1)) + e**((x1**2 + x2**2)/6)
# search optimum x1 and x2 within
# -2 < x1 < 2
# -3 < x2 < 3
# Try various initial conditions and learning rates
# Try large delta for numerical differential
# Compare with analytical differential case

import numpy as np
import matplotlib.pyplot as plt

h = 0.01
n = 100

def F(x1, x2):
    return 2*np.cos(2**x1 - x2**2 + 1) + np.e**((x1**2 + x2**2)/6)

def dF_numerical(x1, x2):
    dF_numerical_x1 = (F(x1+h, x2)-F(x1-h, x2))/(2*h)
    dF_numerical_x2 = (F(x1,x2+h)-F(x1,x2-h))/(2*h)
    return [dF_numerical_x1, dF_numerical_x2]

def dF_analytical(x1,x2):
    dF_analytical_x1 = (1/3)*x1*np.e**((x1**2 + x2**2)/6) - (2**(x1+1)) * np.log(2) * np.sin(2**x1 - x2**2 + 1)
    dF_analytical_x2 = (1/3)*x2*np.e**((x1**2 + x2**2)/6) + 4*x2*np.sin(2**x1 - x2**2 + 1)
    return [dF_analytical_x1, dF_analytical_x2]

def optimum(x1, x2, lr):
    point_x1 = []
    point_x2 = []
    for _ in range(n):
        res = dF_numerical(x1,x2)
        x1 -= res[0] * lr
        x2 -= res[1] * lr
        point_x1.append(x1)
        point_x2.append(x2)
    return [point_x1, point_x2]

x1min = -2
x1max = 2
x2min = -3
x2max = 3
no_of_grid = 100

x1_array = np.linspace(x1min, x1max, no_of_grid) # array for x1-axis
x2_array = np.linspace(x2min, x2max, no_of_grid) # array for x2-axis
# Grid generation
X1, X2 = np.meshgrid(x1_array, x2_array) # Value on each grid point

def plot(case, x1, x2, lr):
    plt.contourf(X1, X2, F(X1,X2))

    points = optimum(x1, x2, lr)
    points_x1 = points[0]
    points_x2 = points[1]

    # plt.plot(x1, x2, marker='o', color='red', linestyle='none')
    plt.scatter(points_x1, points_x2, color='red')
    plt.title(f'LR = {lr}, initial x1 = {x1}, initial x2 = {x2}, delta = {h}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.show()

cases = [(-0.5, 1.0, 0.1), (-0.5, 1.5, 0.05), (-0.5, 1.0, 0.55)]

for i, case in enumerate(cases):
    x1, x2, lr = case
    plot(case, x1, x2, lr)
