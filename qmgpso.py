from qpso import QPSO
from archive import Archive
import numpy as np

class QMGPSO:
    """
    Quantum Multi-Guide Particle Swarm Optimization, incoporating one QPSO subswarm per objective function and an archive enforcing Pareto-dominance for dynamic multi-objective optimization
    """
    def __init__(self, num_particles, search_bounds, objective, num_objectives, is_minimization, w=0.6, c1=0.1, c2=0.02, c3=1.8, neighborhood_size=3, quantum_proportion=0.5, quantum_radius=1.0, quantum_strategy="adaptive", quantum_guide="t", pcx_sigma1=0.1, pcx_sigma2=0.3, pcx_num_parents=3, archive_strategy="hd", archive_size=None, stability_guided=False):
        self.objective = objective
        self.num_objectives = num_objectives
        self.search_bounds = search_bounds
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
            
            self.subswarms.append(QPSO(num_particles=particles_per_swarm[m], search_bounds=search_bounds, objective=obj_m, w=w, c1=c1, c2=c2, c3=c3, neighborhood_size=neighborhood_size, is_minimization=self.is_minimization[m], quantum_proportion=quantum_proportion, quantum_radius=quantum_radius, quantum_strategy=quantum_strategy, quantum_guide=quantum_guide, pcx_sigma1=pcx_sigma1, pcx_sigma2=pcx_sigma2, pcx_num_parents=pcx_num_parents, stability_guided=stability_guided))

        self.history = {
            'archive_size': [],
            'archive_snapshots': []
        }

        self.archive_strategy=archive_strategy

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
        if len(self.archive.fitnesses) > 0:
            snapshot = np.array([f.copy() for f in self.archive.fitnesses])
        else:
            snapshot = np.empty((0, self.num_objectives))
        self.history['archive_snapshots'].append(snapshot)
        

    def update_archive(self):
        """Add current positions to archive"""
        for subswarm in self.subswarms:
            for particle in subswarm.particles:
                full_fitness = self.objective(particle.position)
                self.archive.add_solution(particle.position, full_fitness)

    def handle_change(self):
        """Applies chosen archive management approach upon environment change"""
        if self.archive_strategy == "cl":
            self.archive.clear()
        elif self.archive_strategy == "re":
            self.archive.reevaluate(self.objective)
        elif self.archive_strategy == "h2":
            self.archive.local_search(self.objective, self.search_bounds, step_size=0.02)
        elif self.archive_strategy == "h5":
            self.archive.local_search(self.objective, self.search_bounds, step_size=0.05)
        elif self.archive_strategy == "h10":
            self.archive.local_search(self.objective, self.search_bounds, step_size=0.10)
        elif self.archive_strategy == "hd":
            self.archive.local_search(self.objective, self.search_bounds)
        else:
            pass

        for subswarm in self.subswarms:
            subswarm.handle_change()