import numpy as np
import random

from qmgpso import QMGPSO
from problems import MovingPeaksBenchmark, FDA1, ZJZ

seed = 100
np.random.seed(seed)
random.seed(seed)

dims = 5
# problem = MovingPeaksBenchmark(dims=dims, num_peaks=1, pos_bounds=(0,100))
problem = FDA1(dims=dims, tau_T=25, n_T=10)

num_iterations = 1000

qmgpso = QMGPSO(num_particles=30, search_bounds=problem.search_bounds, objective=problem.evaluate, num_objectives=problem.num_objectives, is_minimization=problem.is_minimization, quantum_proportion=0.5, quantum_radius=2)

# Initialize PSO particle positions and attractors
qmgpso.initialize()

# Main PSO Loop
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    qmgpso.step()
    problem.advance()

    if problem.has_changed():
        print("Environment has changed!")
        problem.handle_change()