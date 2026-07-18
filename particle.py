import numpy as np

class Particle:
    """
    Particle class for PSO
    """
    def __init__(self, search_bounds, is_quantum, num_objectives=1):
        self.search_bounds = search_bounds
        self.is_quantum = is_quantum
        self.num_objectives = num_objectives

        self.position = np.array([np.random.uniform(low, high) for low, high in search_bounds])
        self.velocity = np.zeros(len(search_bounds))

        self.best_position = self.position.copy()
        self.best_fitness = np.array([float('inf')] * num_objectives)

        self.l = np.random.uniform(0,1)

    def get_position(self):
        return self.position