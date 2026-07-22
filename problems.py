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

class FDA1(DynamicProblem):
    """
        A benchmark multi-objective problem proposed by Farina et al., 2004
        Type I Dynamic Problem (POS changes but POF remains static)
    """
    def __init__(self, dims=20, tau_T=10, n_T=10):
        search_bounds = [(0.0, 1.0)] + [(-1.0, 1.0) for _ in range(dims-1)]
        super().__init__(dims=dims, num_objectives=2, search_bounds=search_bounds, is_minimization=[True, True])

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
        g = 1.0 + np.sum((x_rest - G)**2)

        f1 = x1
        f2 = 1.0 - np.sqrt(f1/max(g,1e-9)) # prevents division by 0

        return np.array([f1, f2])

class ZJZ(DynamicProblem):
    """
        A benchmark multi-objective problem proposed by Zhou et al., 2006
        Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, dims=20, tau_T=10, n_T=10):
        search_bounds = [(0.0, 1.0)] + [(-1.0, 2.0) for _ in range(dims-1)]
        super().__init__(dims=dims, num_objectives=2, search_bounds=search_bounds, is_minimization=[True, True])

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

class F5(DynamicProblem):
    pass

class F6(DynamicProblem):
    pass

class F7(DynamicProblem):
    """
    A benchmark problem proposed by Zhou et al., 2014
    Type II Dynamic Problem (Both POS and POF change)
    """
    def __init__(self, dims=10, tau_T=10, n_T=10):
        search_bounds = [(0.0, 5.0) for _ in range(dims)]
        super().__init__(dims=dims, num_objectives=2, search_bounds=search_bounds, is_minimization=[True, True])

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

        n = self.dims
        x1 = x_clamped[0]

        H = 1.25 + 0.75 * np.sin(np.pi * (self.t / self.n_T))
        a = 1.7 * (1 - np.sin(np.pi * self.t)) * np.sin(np.pi * self.t) + 3.4
        b = 1.4 * (1 - np.sin(np.pi * self.t)) * np.cos(np.pi * self.t) + 2.1

        f1_sum, f2_sum = 0.0, 0.0
        for i in range(2, n+1):
            xi = x_clamped[i - 1]
            yi = xi - b - 1 + np.abs(x1-a)**(H+i/n)
            if i % 2 == 1: # odd i -> I1 -> f1
                f1_sum += yi ** 2
            else: # even i -> I2 -> f2
                f2_sum += yi ** 2

        f1 = np.abs(x1-a)**H + f1_sum
        f2 = np.abs(x1-a-1)**H + f2_sum

        return np.array([f1, f2])