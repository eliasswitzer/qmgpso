import numpy as np

class MovingPeaksBenchmark:
    def __init__(self, dims=5, num_peaks=1, pos_bounds=(0,100), height_bounds=(30.0,70.0), width_bounds=(1.0,12.0), change_severity=1.0, height_severity=1.0, width_severity=0.05, lambda_corr=0):
        self.dims = dims
        self.num_peaks = num_peaks
        self.pos_bounds = pos_bounds
        self.height_bounds = height_bounds
        self.width_bounds = width_bounds
        self.s = change_severity
        self.h = height_severity
        self.w = width_severity
        self.l = lambda_corr

        # Initialize peaks
        self.peak_positions = np.random.uniform(pos_bounds[0], pos_bounds[1], (num_peaks, dims))
        self.peak_heights = np.random.uniform(height_bounds[0], height_bounds[1], num_peaks)
        self.peak_widths = np.random.uniform(width_bounds[0], width_bounds[1], num_peaks)

        # Initialize random peak velocities with length s
        velocities_raw = np.random.uniform(-1,1,(num_peaks, dims))
        self.velocities = (velocities_raw / np.linalg.norm(velocities_raw, axis=1, keepdims=True)) * self.s

    def evaluate(self, x):
        fitness_vals = []
        for i in range(self.num_peaks):
            fitness_vals.append(self.peak_heights[i] / (1 + (self.peak_widths[i] * np.sum((x - self.peak_positions[i]) ** 2))))
        return np.max(fitness_vals)
    
    def change_environment(self):
        """
        Updates peak positions, heights, and widths according to formula
        """
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