import numpy as np
import random
import argparse

from qmgpso import QMGPSO
from problems import FDA1, ZJZ, F5, F6, F7
from visualizations import plot_pareto_front, plot_archive_size, plot_pareto_front_history

# PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', metavar="seed", type=int, required=True, help="The random seed for reproducibility.")

# Problem Parameters
parser.add_argument('-p', '--problem', metavar="problem", type=str, required=True, choices=["FDA1", "ZJZ", "F5", "F6", "F7"], help="The dynamic multi-objective benchmark problem for optimization.")
parser.add_argument("-d", "--dims", metavar="dims", type=int, required=True, help="The number of dimensions/decision variables for set problem.")
parser.add_argument("-i", "--iterations", metavar="iterations", type=int, required=True, help="The number of iterations to run the PSO algorithm/problem optimization for.")
parser.add_argument('-tt', "--tau_t", metavar="tau_t", type=int, required=True, help="The change frequency parameter. i.e. how many iterations the algorithm will will run before the environment changes.")
parser.add_argument('-nt', "--n_t", metavar="n_t", type=int, required=True, help="The change severity parameter. i.e. how much the environment changes by.")

# PSO Parameters
parser.add_argument("-np", "--particles", metavar="num_particles", type=int, required=False, default=100, help="The number of particles to run the QMGPSO algorithm with. Default=100")
parser.add_argument('-w', metavar='w', type=float, required=False, default=0.6, help="The inertia term weight for the QMGPSO algorithm. Default=0.6")
parser.add_argument('-c1', metavar='c1', type=float, required=False, default=0.1, help="The cognitive acceleration coefficient for the QMGPSO algorithm. Default=0.1")
parser.add_argument('-c2', metavar='c2', type=float, required=False, default=0.02, help="The social acceleration coefficient for the QMGPSO algorithm. Default=0.02")
parser.add_argument('-c3', metavar='c3', type=float, required=False, default=1.8, help="The archive-pull acceleration coefficient for the QMGPSO algorithm. Default=1.8")
parser.add_argument('-ns', '--neighborhood_size', metavar='neighborhood_size', type=int, required=False, default=3, help="The neighborhood size for QMGPSO swarms. Default=3")

# QPSO Parameters
parser.add_argument('-qp', '--quantum_proportion', metavar="quantum_proportion", type=float, required=False, default=0.5, help="The proportion of total particles to designate as quantum particles (will update using proportion sampling and radius). Default=0.5")
parser.add_argument("-qr", '--quantum_radius', metavar="quantum_radius", type=float, required=False, default=0.5, help="The problem-dependent radius parameter for the distribution used in quantum particle position update. Default=0.5")

# MGPSO Parameters
parser.add_argument('-st', '--archive_strategy', metavar="archive_strategy", type=str, required=False, default="hd", choices=["cl", "re", "h2", "h5", "h10", "hd"], help="The archive management strategy used to update archive solutions upon environment change. Default=hd")
parser.add_argument('-as', "--archive_size", metavar="archive_size", type=int, required=False, default=100, help="The max size for the bounded archive. Default=100")

args = parser.parse_args()
print(f"Args: {args}")

# Set Random Seed
np.random.seed(args.seed)
random.seed(args.seed)

# Initialize DMOP (Dynamic Multi Objective Problem)
if args.problem == "FDA1":
    problem = FDA1(dims=args.dims, tau_T=args.tau_t , n_T=args.n_t)
elif args.problem == "ZJZ":
    problem = ZJZ(dims=args.dims, tau_T=args.tau_t , n_T=args.n_t)
elif args.problem == "F5":
    problem = F5(dims=args.dims, tau_T=args.tau_t , n_T=args.n_t)
elif args.problem == "F6":
    problem = F6(dims=args.dims, tau_T=args.tau_t , n_T=args.n_t)
elif args.problem == "F7":
    problem = F7(dims=args.dims, tau_T=args.tau_t , n_T=args.n_t)


qmgpso = QMGPSO(num_particles=args.particles, search_bounds=problem.search_bounds, objective=problem.evaluate, num_objectives=problem.num_objectives, is_minimization=problem.is_minimization, w=args.w, c1=args.c1, c2=args.c2, c3=args.c3, neighborhood_size=args.neighborhood_size, quantum_proportion=args.quantum_proportion, quantum_radius=args.quantum_radius, archive_strategy=args.archive_strategy, archive_size=args.archive_size)
qmgpso.initialize()

# Main PSO Loop
for iteration in range(args.iterations):
    print(f"Iteration {iteration + 1}/{args.iterations}")

    qmgpso.step()
    problem.advance()

    if problem.has_changed():
        print("Environment has changed!")
        problem.handle_change()
        qmgpso.handle_change()

# Show plots
plot_pareto_front(qmgpso.archive)
plot_archive_size(qmgpso.history)
plot_pareto_front_history(qmgpso.history, num_snapshots=1000, tau_T=args.tau_t)

# TODO: METRICS (accuracy, stability, etc.) and more VISUALS