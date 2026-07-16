import numpy as np
from abc import ABC, abstractmethod

class DynamicProblem(ABC):
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

class MovingPeaksBenchmark(DynamicProblem):
    def __init__(self, dims=5, num_peaks=1, pos_bounds=(0,100), height_bounds=(30.0,70.0), width_bounds=(1.0,12.0), change_interval=200, change_severity=1.0, height_severity=1.0, width_severity=0.05, lambda_corr=0):
        super().__init__(dims=dims, num_objectives=1, search_bounds=[pos_bounds for _ in range(dims)], is_minimization=False)

        self.pos_bounds = pos_bounds
        self.height_bounds = height_bounds
        self.width_bounds = width_bounds
        self.s = change_severity
        self.h = height_severity
        self.w = width_severity
        self.l = lambda_corr
        self.change_interval = change_interval

        # Initialize peaks
        self.num_peaks = num_peaks
        self.peak_positions = np.random.uniform(pos_bounds[0], pos_bounds[1], (num_peaks, dims))
        self.peak_heights = np.random.uniform(height_bounds[0], height_bounds[1], num_peaks)
        self.peak_widths = np.random.uniform(width_bounds[0], width_bounds[1], num_peaks)

        # Initialize random peak velocities with length s
        velocities_raw = np.random.uniform(-1,1,(num_peaks, dims))
        self.velocities = (velocities_raw / np.linalg.norm(velocities_raw, axis=1, keepdims=True)) * self.s

    def has_changed(self) -> bool:
        return self.iteration > 0 and self.iteration % self.change_interval == 0
    
    def handle_change(self):
        """Updates peak positions, heights, and widths according to formula"""
        for i in range(self.num_peaks):
            # Generate random vector and calculate new position based on it and previous velocity
            r = np.random.uniform(-1, 1, self.dims)
            r = (r / np.linalg.norm(r)) * self.s

            new_velocity = (self.s / np.abs(r + self.velocities[i])) * (((1-self.l)*r) + (self.l*self.velocities[i]))
            self.velocities[i] = new_velocity

            new_position = self.peak_positions[i] + self.velocities[i]

            # Boundary handling: check if the peak left the domain and if so, negate velocity
            for d in range(self.dims):
                if new_position[d] < self.pos_bounds[0] or new_position[d] > self.pos_bounds[1]:
                    self.velocities[i][d] = -self.velocities[i][d]
                    new_position[d] = self.peak_positions[i][d] + self.velocities[i][d]

            self.peak_positions[i] = new_position

            # Vary peak heights and widths
            sigma = np.random.normal(0,1)
            self.peak_heights[i] += sigma * self.h
            self.peak_widths[i] += sigma * self.w
            self.peak_heights[i] = np.clip(self.peak_heights[i], self.height_bounds[0], self.height_bounds[1])
            self.peak_widths[i] = np.clip(self.peak_widths[i], self.width_bounds[0], self.width_bounds[1])

    def evaluate(self, x):
        fitness_vals = []
        for i in range(self.num_peaks):
            fitness_vals.append(self.peak_heights[i] / (1 + (self.peak_widths[i] * np.sum((x - self.peak_positions[i]) ** 2))))
        return np.array([np.max(fitness_vals)])

class FDA1(DynamicProblem):
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