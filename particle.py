import numpy as np

class Particle:
    def __init__(self, search_bounds, is_quantum):
        self.search_bounds = search_bounds
        num_dims = len(search_bounds)

        self.position = np.array([np.random.uniform(low, high) for low, high in search_bounds])
        self.velocity = np.zeros(len(search_bounds))

        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

        self.is_quantum = is_quantum

    def get_position(self):
        return self.position