import numpy as np
import random

def dominates(fitness_a, fitness_b, is_minimization):
    "Returns True if fitness a Pareto-dominates fitness b, where a dominates b if a is at least as good as b in every objective and strictly better in at least one"
    at_least_as_good = True
    strictly_better = False
    for a, b, minimize in zip(fitness_a, fitness_b, is_minimization):
        if minimize:
            better, worse = a < b, a > b
        else:
            better, worse = a > b, a < b
        if worse:
            at_least_as_good = False
        if better:
            strictly_better = True
    return at_least_as_good and strictly_better

def crowding_distance(fitnesses):
    """Computes crowding distance for each entry in fitnesses and returns a list containing distances for each solution in archive"""
    fitnesses = np.array(fitnesses)
    n, num_objectives = fitnesses.shape
    distances = np.zeros(n)

    if n <= 2:
        return np.full(n, np.inf)
    
    for m in range(num_objectives): 
        order = np.argsort(fitnesses[:, m]) # Sort solutions in ascending order
        f_min = fitnesses[order[0], m]
        f_max = fitnesses[order[-1], m]

        distances[order[0]] = np.inf # Infinite distance assigned to boundary points
        distances[order[-1]] = np.inf

        if f_max == f_min:
            continue

        for i in range(1, n-1): # for all intermediate points, calculate the difference between the objective values of the two adjacent neighbors and divide by total range of objective function and sum together distance for each objective function
            prev_value = fitnesses[order[i-1], m]
            next_value = fitnesses[order[i+1], m]
            distances[order[i]] += (next_value - prev_value) / (f_max - f_min)

    return distances


class Archive:
    """Stores solutions and ensures that all stored solutions are non-dominated (i.e. each solution represents a good trade-off between objectives)"""
    def __init__(self, is_minimization, max_size=None):
        self.is_minimization = is_minimization
        self.max_size = max_size
        self.positions = []
        self.fitnesses = []

    def add_solution(self, position, fitness):
        """
        Adds a new solution to the archive and uses bounded archive maintenance to update it
        """
        # Reject solution if an existing archive member dominates it or is similar to it
        for existing_fitness in self.fitnesses:
            if dominates(existing_fitness, fitness, self.is_minimization):
                return
            if np.array_equal(existing_fitness, fitness):
                return
            
        # New solution is added, and thus drop any existing members that it dominates
        keep_positions, keep_fitnesses = [], []
        for pos, fit in zip(self.positions, self.fitnesses):
            if not dominates(fitness, fit, self.is_minimization):
                keep_positions.append(pos)
                keep_fitnesses.append(fit)
        self.positions, self.fitnesses = keep_positions, keep_fitnesses

        self.positions.append(np.array(position).copy())
        self.fitnesses.append(np.array(fitness).copy())

        # If archive size now exeeds the archive size limit, then remove most crowded solution from the archive
        if self.max_size is not None and len(self.positions) > self.max_size:
            distances = crowding_distance(self.fitnesses)
            drop_idx = int(np.argmin(distances)) # min distance indicates most crowded
            del self.positions[drop_idx]
            del self.fitnesses[drop_idx]

    def random_member(self):
        """Return the position of a random member of the archive (used in tournament selection)"""
        if not self.positions:
            return None
        return self.positions[random.randrange(len(self.positions))]
    
    def tournament_select(self, tournament_size=3):
        """
        Picks an archive guide via tournament selection and which has the largest crowding distance
        """
        n = len(self.positions)
        # Defaults if there are not enough stored solutions for 3-tourament selection
        if n == 0:
            return None
        if n == 1:
            return self.positions[0] 
        
        distances = crowding_distance(self.fitnesses)
        k = min(tournament_size, n) # chooses 2 or 3 based on amount of solutions in archive
        candidate_indices = random.sample(range(n), k) # choose random indices
        best_idx = max(candidate_indices, key=lambda idx: distances[idx])
        return self.positions[best_idx]

    def __len__(self):
        return len(self.positions)
