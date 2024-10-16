import numpy as np
from scipy import integrate

# Define the function to integrate
def func(x):
    return np.exp(-x**2)  # function: exp(-x^2)

# Perform numerical integration using quad
result, error = integrate.quad(func, -10, 10)  # Integrate x^2 from 0 to 4

print("Result of the integration:", result)
print("Error of the integration:", error)