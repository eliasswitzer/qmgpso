from qpso import QPSO
from archive import Archive

class QMGPSO:
    def __init__(self, num_particles, search_bounds, objective, num_objectives, is_minimization, w=0.6, c1=0.1, c2=0.02, c3=1.8, neighborhood_size=3, quantum_proportion=0.5, quantum_radius=1.0, archive_size=None):
        self.objective = objective
        self.num_objectives = num_objectives
        if isinstance(is_minimization, bool):
            self.is_minimization = [is_minimization] * num_objectives
        else:
            self.is_minimization = list(is_minimization)

        self.archive = Archive(is_minimization=self.is_minimization, max_size=archive_size)

        self.subswarms = []
        # TODO: split number of particles by number of objectives
        for m in range(num_objectives):
            def obj_m(x, m=m):
                return self.objective(x)[m]
            
            self.subswarms.append(QPSO(num_particles=num_particles, search_bounds=search_bounds, objective=obj_m, w=w, c1=c1, c2=c2, c3=c3, neighborhood_size=neighborhood_size, is_minimization=self.is_minimization[m], quantum_proportion=quantum_proportion, quantum_radius=quantum_radius))

    def initialize(self):
        for subswarm in self.subswarms:
            subswarm.initialize()
        self.update_archive()

    def step(self):
        for subswarm in self.subswarms:
            subswarm.step(archive=self.archive)
        self.update_archive()

    def update_archive(self):
        for subswarm in self.subswarms:
            for particle in subswarm.particles:
                full_fitness = self.objective(particle.best_position)
                self.archive.add_solution(particle.best_position, full_fitness)