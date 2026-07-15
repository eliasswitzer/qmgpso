import numpy as np

from pso import PSO
from particle import Particle
from functions import sphere_function, rosenbrock_function, ackley_function, rastrigin_function
from mpb import MovingPeaksBenchmark
from visualizations import plot_fitness, plot_diversity

np.random.seed(100)

dims = 5
mbp = MovingPeaksBenchmark(dims=dims, num_peaks=1, pos_bounds=(0,100))
mbp_bounds = [(0, 100) for _ in range(dims)]

num_iterations = 1000
change_interval = 200

pso = PSO(num_particles=30, search_bounds=mbp_bounds, objective=mbp.evaluate, is_minimization=False, quantum_proportion=0.5, quantum_radius=2)

# Initialize PSO particle positions and attractors
pso.initialize()

# Main PSO Loop
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    # --- Environment Change Trigger ---
    if iteration > 0 and iteration % change_interval == 0:
        print(f"Environment has changed!")
        mbp.change_environment()

    pso.step()

plot_fitness(pso.history)
plot_diversity(pso.history)

print("Best architecture found:", pso.global_best_position)
print("Peak Positions:", mbp.peak_positions)
print("Peak Heights", mbp.peak_heights)
print("Peak Widths", mbp.peak_widths)