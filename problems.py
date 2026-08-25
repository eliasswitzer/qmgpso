import numpy as np
from abc import ABC, abstractmethod

class DynamicProblem(ABC):
    """Abstract class for dynamic problems"""
    def __init__(self, dims, num_objectives, search_bounds, is_minimization):
        self.dims = dims
        self.num_objectives = num_objectives
        self.search_bounds = search_bounds
        if isinstance(is_minimization, bool): # is_minimization needs to be a list of booleans for each objective
            self.is_minimization = [is_minimization] * num_objectives
        else:
            self.is_minimization = list(is_minimization)
        self.iteration = 0

    def advance(self):
        """Advances iteration of the problem"""
        self.iteration += 1

    @abstractmethod
    def has_changed(self) -> bool:
        """Returns True if an environmental change has occured"""
        pass

    @abstractmethod
    def handle_change(self):
        """Performs internal environment change according to the problem"""
        pass

    @abstractmethod
    def evaluate(self, x):
        """Evaluates a position vector and returns a numpy array with fitness values for each objective"""
        pass

    @abstractmethod
    def true_pareto_front(self, num_points=1000):
        """Returns a NumPy array containing objective vectors uniformly sampled from the true Pareto-optimal front of the problem at its current point in time"""
        pass

class FDA1(DynamicProblem):
    """
        A benchmark multi-objective problem proposed by Farina et al., 2004
        Type I Dynamic Problem (POS changes but POF remains static)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=20, num_objectives=2, search_bounds=[(0.0, 1.0)] + [(-1.0, 1.0) for _ in range(20)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step/environment index value (number of times the environment has changed so far)

    def has_changed(self) -> bool:
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)
    
    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T) # change environments

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        x1 = x_clamped[0]
        x_rest = x_clamped[1:]

        G = np.sin(0.5 * np.pi * self.t)
        g = 1.0 + np.sum((x_rest - G)**2)

        f1 = x1
        f2 = 1.0 - np.sqrt(f1/max(g,1e-9)) # prevents division by 0

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        f1 = np.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])

class ZJZ(DynamicProblem):
    """
        A benchmark multi-objective problem proposed by Zhou et al., 2006
        Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=20, num_objectives=2, search_bounds=[(0.0, 1.0)] + [(-1.0, 2.0) for _ in range(20)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self) -> bool:
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)
    
    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        x1 = x_clamped[0]
        x_rest = x_clamped[1:]

        G = np.sin(0.5 * np.pi * self.t)
        H = 1.5 + G

        g = 1.0 + np.sum((x_rest + G - (x1**H))**2)

        f1 = x1
        f2 = 1.0 - (f1/max(g,1e-9))**H # prevents division by 0

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        G = np.sin(0.5 * np.pi * self.t)
        H = 1.5 + G
        f1 = np.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - f1**H
        return np.column_stack([f1, f2])

class FDA2(DynamicProblem):
    """
        A benchmark multi-objective problem proposed by Camara et al., 2010
        Type III Dynamic Problem (POF changes but POS remains static)
    """
    def __init__(self, tau_T=10, n_T=10, z=5):
        super().__init__(dims=31, num_objectives=2, search_bounds=[(0.0, 1.0)] + [(-1.0, 1.0) for _ in range(30)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value
        self.z = z # problem parameter

    def has_changed(self) -> bool:
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)
    
    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        x1 = x[0]
        x2 = x[1:16]
        x3 = x[16:]

        H = self.z ** (-np.cos(np.pi *  self.t / 4))
        f1 = x1

        g = 1 + np.sum(x2**2)
        h = 1 - (np.clip(f1/max(g,1e-9), 0.0, 1.0))**(H + np.sum((x3-(H/2))**2))

        f2 = g*h

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        H = self.z ** (-np.cos(np.pi *  self.t / 4))
        f1 = np.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - f1**H
        return np.column_stack([f1, f2])


class F5(DynamicProblem):
    """
    A benchmark problem proposed by Zhou et al., 2014
    Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(0.0, 5.0) for _ in range(10)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self) -> bool:
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)
    
    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        n = self.dims
        x1 = x[0]

        H = 1.25 + 0.75 * np.sin(np.pi * self.t)
        a = 2 * np.cos(np.pi * self.t) + 2
        b = 2 * np.sin(2 * np.pi * self.t) + 2

        f1_sum, f2_sum = 0.0, 0.0
        for i in range(2, n+1):
            xi = x[i - 1]
            yi = xi - b - 1 + np.abs(x1-a)**(H+i/n)
            if i % 2 == 1: # odd i -> I1 -> f1
                f1_sum += yi ** 2
            else: # even i -> I2 -> f2
                f2_sum += yi ** 2

        f1 = np.abs(x1-a)**H + f1_sum
        f2 = np.abs(x1-a-1)**H + f2_sum

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        H = 1.25 + 0.75 * np.sin(np.pi * self.t)
        s = np.linspace(0.0, 1.0, num_points) # parameterize s=x1-a gives f1=s**H and f2=(1-s)**H
        f1 = s**H
        f2 = (1.0 - s)**H
        return np.column_stack([f1, f2])

class F6(DynamicProblem):
    """
    A benchmark problem proposed by Zhou et al., 2014
    Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(0.0, 5.0) for _ in range(10)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self) -> bool:
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)
    
    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        n = self.dims
        x1 = x[0]

        H = 1.25 + 0.75 * np.sin(np.pi * self.t)
        a = 2 * np.cos(1.5 * np.pi * self.t) * np.sin(0.5 * np.pi * self.t) + 2
        b = 2 * np.cos(1.5 * np.pi * self.t) * np.cos(0.5 * np.pi * self.t) + 2

        f1_sum, f2_sum = 0.0, 0.0
        for i in range(2, n+1):
            xi = x[i - 1]
            yi = xi - b - 1 + np.abs(x1-a)**(H+i/n)
            if i % 2 == 1: # odd i -> I1 -> f1
                f1_sum += yi ** 2
            else: # even i -> I2 -> f2
                f2_sum += yi ** 2

        f1 = np.abs(x1-a)**H + f1_sum
        f2 = np.abs(x1-a-1)**H + f2_sum

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
            H = 1.25 + 0.75 * np.sin(np.pi * self.t)
            s = np.linspace(0.0, 1.0, num_points) # parameterize s=x1-a gives f1=s**H and f2=(1-s)**H
            f1 = s**H
            f2 = (1.0 - s)**H
            return np.column_stack([f1, f2])

class F7(DynamicProblem):
    """
    A benchmark problem proposed by Zhou et al., 2014
    Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(0.0, 5.0) for _ in range(10)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self) -> bool:
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)
    
    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        n = self.dims
        x1 = x[0]

        H = 1.25 + 0.75 * np.sin(np.pi * self.t)
        a = 1.7 * (1 - np.sin(np.pi * self.t)) * np.sin(np.pi * self.t) + 3.4
        b = 1.4 * (1 - np.sin(np.pi * self.t)) * np.cos(np.pi * self.t) + 2.1

        f1_sum, f2_sum = 0.0, 0.0
        for i in range(2, n+1):
            xi = x[i - 1]
            yi = xi - b - 1 + np.abs(x1-a)**(H+i/n)
            if i % 2 == 1: # odd i -> I1 -> f1
                f1_sum += yi ** 2
            else: # even i -> I2 -> f2
                f2_sum += yi ** 2

        f1 = np.abs(x1-a)**H + f1_sum
        f2 = np.abs(x1-a-1)**H + f2_sum

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
            H = 1.25 + 0.75 * np.sin(np.pi * self.t)
            s = np.linspace(0.0, 1.0, num_points) # parameterize s=x1-a gives f1=s**H and f2=(1-s)**H
            f1 = s**H
            f2 = (1.0 - s)**H
            return np.column_stack([f1, f2])

class FDA4(DynamicProblem):
    """
    A triobjective benchmark problem proposed by Farina et al., 2004
    Type 1 Dynamic Problem (POS changes but POF remains static)
    """
    def __init__(self, tau_T=10, n_T=10, M=3):
        self.M = M
        n = M + 9
        super().__init__(dims=n, num_objectives=M, search_bounds=[(0.0, 1.0) for _ in range(n)], is_minimization=[True] * M)

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self):
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)

    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        M = self.M
        xI = x_clamped[:M - 1]
        xII = x_clamped[M-1:]

        G = np.abs(np.sin(0.5 * np.pi * self.t))
        g = np.sum((xII - G)**2)

        f = np.zeros(M)
        f[0] = (1.0 + g) * np.prod(np.cos(xI * np.pi / 2.0))
        for k in range(2, M):
            f[k-1] = (1.0 + g) * np.prod(np.cos(xI[:M-k] * np.pi / 2.0)) * np.sin(xI[M-k] * np.pi / 2.0)
        f[M-1] = (1.0 + g) * np.sin(xI[0] * np.pi / 2.0)

        return f

    def true_pareto_front(self, num_points=1000):
        M = self.M
        if M == 3:
            side = int(np.ceil(np.sqrt(num_points)))
            x1 = np.linspace(0.0, 1.0, side)
            x2 = np.linspace(0.0, 1.0, side)
            X1, X2 = np.meshgrid(x1, x2)
            X1, X2 = X1.ravel(), X2.ravel()
            f1 = np.cos(X1 * np.pi / 2.0) * np.cos(X2 * np.pi / 2.0)
            f2 = np.cos(X1 * np.pi / 2.0) * np.sin(X2 * np.pi / 2.0)
            f3 = np.sin(X1 * np.pi / 2.0)
            return np.column_stack([f1, f2, f3])
        else:
            pts = np.abs(np.random.normal(size=(num_points, M)))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            return pts

class DIMP1(DynamicProblem):
    """
    A benchmark problem proposed by Koo et al., 2010
    Type I Dynamic Problem (POS changes but POF remains static)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(0.0, 1.0)] + [(-1.0, 1.0) for _ in range(9)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self):
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)

    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def _G(self, i):
        n = self.dims
        return np.sin(0.5 * np.pi * self.t + 2.0 * np.pi * (i / (n+1))**2)

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        n = self.dims
        x1 = x_clamped[0]

        g = 1.0
        for i in range(2, n+1):
            xi = x_clamped[i-1]
            g += (xi - self._G(i))**2

        f1 = x1
        f2 = g * (1.0 - (f1 / max(g, 1e-9))**2)

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        f1 = np.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - f1**2
        return np.column_stack([f1, f2])

class DF4(DynamicProblem):
    """
    A benchmark problem proposed by Jiang et al., 2017 (CEC2018 competition)
    Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(-2.0, 2.0) for _ in range(10)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self):
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)

    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        n = self.dims
        x1 = x_clamped[0]

        a = np.sin(0.5 * np.pi * self.t)
        b = 1.0 + np.abs(np.cos(0.5 * np.pi * self.t))
        H = 1.5 + a

        g = 1.0
        for i in range(2, n+1):
            xi = x_clamped[i-1]
            g += (xi - a * (x1**2) / i)**2

        f1 = g * np.abs(x1 - a)**H
        f2 = g * np.abs(x1 - a - b)**H

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        a = np.sin(0.5 * np.pi * self.t)
        b = 1.0 + np.abs(np.cos(0.5 * np.pi * self.t))
        H = 1.5 + a
        f1 = np.linspace(0.0, b**H, num_points)
        f2 = (b - f1**(1.0 / H))**H
        return np.column_stack([f1, f2])

class DF5(DynamicProblem):
    """
    A benchmark problem proposed by Jiang et al., 2017 (CEC2018 competition)
    Type III Dynamic Problem (POF changes but POS remains static)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(0.0, 1.0)] + [(-1.0, 1.0) for _ in range(9)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self):
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)

    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        x1 = x_clamped[0]
        x_rest = x_clamped[1:]

        G = np.sin(0.5 * np.pi * self.t)
        w_t = np.floor(10.0 * G)

        g = 1.0 + np.sum((x_rest - G)**2)

        f1 = g * (x1 + 0.02 * np.sin(w_t * np.pi * x1))
        f2 = g * (1.0 - x1 + 0.02 * np.sin(w_t * np.pi * x1))

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        G = np.sin(0.5 * np.pi * self.t)
        w_t = np.floor(10.0 * G)
        x1 = np.linspace(0.0, 1.0, num_points)
        f1 = x1 + 0.02 * np.sin(w_t * np.pi * x1)
        f2 = 1.0 - x1 + 0.02 * np.sin(w_t * np.pi * x1)
        return np.column_stack([f1, f2])

class DF6(DynamicProblem):
    """
    A benchmark problem proposed by Jiang et al., 2017 (CEC2018 competition)
    Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, tau_T=10, n_T=10):
        super().__init__(dims=10, num_objectives=2, search_bounds=[(0.0, 1.0)] + [(-1.0, 1.0) for _ in range(9)], is_minimization=[True, True])

        self.tau_T = tau_T # frequency of change
        self.n_T = n_T # severity of change
        self.t = 0.0 # time-step value

    def has_changed(self):
        return self.iteration > 0 and (self.iteration % self.tau_T == 0)

    def handle_change(self):
        self.t = (1.0 / self.n_T) * np.floor(self.iteration / self.tau_T)

    def evaluate(self, x):
        # Clip values to ensure they stay within bounds
        lows = np.array([b[0] for b in self.search_bounds])
        highs = np.array([b[1] for b in self.search_bounds])
        x_clamped = np.clip(x, lows, highs)

        x1 = x_clamped[0]
        G = np.sin(0.5 * np.pi * self.t)
        alpha_t = 0.2 + 2.8 * np.abs(G)

        y = x_clamped[1:] - G
        g = 1.0 + np.sum(np.abs(G) * y**2 - 10.0 * np.cos(2.0 * np.pi * y) + 10.0)

        f1 = g * (x1 + 0.1 * np.sin(3.0 * np.pi * x1))**alpha_t
        f2 = g * (1.0 - x1 + 0.1 * np.sin(3.0 * np.pi * x1))**alpha_t

        return np.array([f1, f2])

    def true_pareto_front(self, num_points=1000):
        G = np.sin(0.5 * np.pi * self.t)
        alpha_t = 0.2 + 2.8 * np.abs(G)
        x1 = np.linspace(0.0, 1.0, num_points)
        f1 = (x1 + 0.1 * np.sin(3.0 * np.pi * x1))**alpha_t
        f2 = (1.0 - x1 + 0.1 * np.sin(3.0 * np.pi * x1))**alpha_t
        return np.column_stack([f1, f2])