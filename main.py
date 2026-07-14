import numpy as np

from pso import PSO
from particle import Particle
from functions import sphere_function, rosenbrock_function, ackley_function, rastrigin_function
from mpb import MovingPeaksBenchmark
from visualizations import plot_fitness, plot_diversity

np.random.seed(50)

dims = 5
mbp = MovingPeaksBenchmark(dims=dims, num_peaks=1, pos_bounds=(0,100))
mbp_bounds = [(0, 100) for _ in range(dims)]

pso = PSO(num_particles=30, search_bounds=mbp_bounds, objective=mbp.evaluate, is_minimization=False, quantum_proportion=0.5)

best_position, history = pso.optimize(num_iterations=1000, neighborhood_size=3, dynamic_env=mbp, change_interval=200)

plot_fitness(history)
plot_diversity(history)

print("Best architecture found: ", best_position)
print(mbp.peak_positions)
print(mbp.peak_heights)
print(mbp.peak_widths)