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
    """A benchmark multi-objective problem proposed by Farina et al., 2004"""
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
        f2 = 1 - np.sqrt(f1/max(g,1e-9)) # prevents division by 0

        return np.array([f1, f2])

class ZJZ(DynamicProblem):
    pass

class F5(DynamicProblem):
    pass

class F6(DynamicProblem):
    pass

class F7(DynamicProblem):
    pass