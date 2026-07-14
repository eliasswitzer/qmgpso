import numpy as np

def sphere_function(x):
        return np.sum(x**2)
    
def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c*x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.exp(1)

def rastrigin_function(x):
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))