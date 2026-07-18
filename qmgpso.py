from qpso import QPSO
from archive import Archive

class QMGPSO:
    """
    Quantum Multi-Guide Particle Swarm Optimization, incoporating one QPSO subswarm per objective function and an archive enforcing Pareto-dominance for dynamic multi-objective optimization
    """
    def __init__(self, num_particles, search_bounds, objective, num_objectives, is_minimization, w=0.6, c1=0.1, c2=0.02, c3=1.8, neighborhood_size=3, quantum_proportion=0.5, quantum_radius=1.0, archive_size=None):
        self.objective = objective
        self.num_objectives = num_objectives
        if isinstance(is_minimization, bool):
            self.is_minimization = [is_minimization] * num_objectives
        else:
            self.is_minimization = list(is_minimization)

        self.archive = Archive(is_minimization=self.is_minimization, max_size=archive_size)

        # Split number of particles by number of objectives (accounts for uneven splits)
        p = num_particles // num_objectives
        remainder = num_particles % num_objectives
        particles_per_swarm = [p + (1 if m < remainder else 0) for m in range(num_objectives)]

        # Create each subswarm
        self.subswarms = []
        for m in range(num_objectives):
            def obj_m(x, m=m): # Separate each objective function by passing in mini-function that returns corresponding objective value
                return self.objective(x)[m]
            
            self.subswarms.append(QPSO(num_particles=particles_per_swarm[m], search_bounds=search_bounds, objective=obj_m, w=w, c1=c1, c2=c2, c3=c3, neighborhood_size=neighborhood_size, is_minimization=self.is_minimization[m], quantum_proportion=quantum_proportion, quantum_radius=quantum_radius))

        self.history = {
            'archive_size': [],
            'archive_snapshots': []
        }

    def initialize(self):
        """Initialize each subswarm (calls subswarm initialize method to set initial positions, velocities, etc.) and add initial positions to archive"""
        for subswarm in self.subswarms:
            subswarm.initialize()
        self.update_archive()

    def step(self):
        """Do one optimization step for each subswarm and subsequently update archive"""
        for subswarm in self.subswarms:
            subswarm.step(archive=self.archive)
        self.update_archive()
        self.history['archive_size'].append(len(self.archive))
        self.history['archive_snapshots'].append([f.copy() for f in self.archive.fitnesses])

    def update_archive(self):
        """Add current positions to archive"""
        for subswarm in self.subswarms:
            for particle in subswarm.particles:
                full_fitness = self.objective(particle.position)
                self.archive.add_solution(particle.best_position, full_fitness)