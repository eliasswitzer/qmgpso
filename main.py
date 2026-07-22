import numpy as np
import random

from qmgpso import QMGPSO
from problems import FDA1, ZJZ, F7
from visualizations import plot_pareto_front, plot_archive_size, plot_pareto_front_history

# Set Random Seed
seed = 100
np.random.seed(seed)
random.seed(seed)

# Initialize DMOP (Dynamic Multi Objective Problem)
dims = 20
problem =ZJZ(dims=dims, tau_T=25 , n_T=10)
num_iterations = 1000

qmgpso = QMGPSO(num_particles=100, search_bounds=problem.search_bounds, objective=problem.evaluate, num_objectives=problem.num_objectives, is_minimization=problem.is_minimization, quantum_proportion=0, quantum_radius=0.5, archive_strategy="hd", archive_size=100)

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
        qmgpso.handle_change()

# Show plots
plot_pareto_front(qmgpso.archive)
plot_archive_size(qmgpso.history)
plot_pareto_front_history(qmgpso.history, num_snapshots=1000, tau_T=50)

# TODO: METRICS (accuracy, stability, etc.) and more VISUALS