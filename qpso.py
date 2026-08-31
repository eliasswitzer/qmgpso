import numpy as np
import operator
import random

from particle import Particle

class QPSO:
    """
    Runs single-objective QPSO
    """
    def __init__(self, num_particles, search_bounds, objective, w=0.729844, c1=0.1, c2=0.02, c3=1.8, neighborhood_size=3, is_minimization=True, quantum_proportion=0.5, quantum_radius=1.0, quantum_strategy="adaptive", quantum_guide="t", pcx_sigma1=0.1, pcx_sigma2=0.3, pcx_num_parents=3, stability_guided=False):
        self.iteration = 0
        self.particles = []
        num_quantum = int(num_particles * quantum_proportion)
        num_neutral = num_particles - num_quantum
        for i in range(num_neutral): # initialize neutral particles
            self.particles.append(Particle(search_bounds=search_bounds, is_quantum=False))
        for i in range(num_quantum): # initialize quantum particles
            self.particles.append(Particle(search_bounds=search_bounds, is_quantum=True))

        self.search_bounds = search_bounds
        self.objective = objective
        self.is_minimization = is_minimization

        if self.is_minimization:
            self.better_than = operator.lt
            self.global_best_fitness = float('inf')
        else:
            self.better_than = operator.gt
            self.global_best_fitness = float('-inf')

        self.global_best_position = None
        self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3
        self.neighborhood_size = neighborhood_size

        # Initialize original QPSO parameter
        self.quantum_radius = quantum_radius

        # Quantum particle strategy configuration
        self.quantum_strategy = quantum_strategy
        self.quantum_guide = quantum_guide

        # PCX QPSO parameters
        self.pcx_sigma1 = pcx_sigma1
        self.pcx_sigma2 = pcx_sigma2
        self.pcx_num_parents = pcx_num_parents

        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

        self.stability_guided = stability_guided

    def initialize(self):
        """Initializes n particles, their position, velocity and personal best fitnesses"""
        # Initial Evaluation
        # print("Evaluating Initial Population")
        for i in range(len(self.particles)):
            # print(f"Particle {i+1}")
            position = self.particles[i].get_position()

            fitness = self.objective(position)
            # print(f"Fitness: {fitness:.4f} | Position: {position}")

            # Set Initial Personal Best
            self.particles[i].best_fitness = fitness
            self.particles[i].best_position = self.particles[i].position.copy()

            # Set Initial Global Best
            if self.better_than(fitness, self.global_best_fitness):
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_fitness = fitness

    def _get_neighborhood_best(self, i):
        """Returns the nbest position for particle i"""
        neighbor_indices = [(i + j) % len(self.particles) for j in range(-(self.neighborhood_size // 2), (self.neighborhood_size // 2) + 1)]
        
        # Find which local particle has the best personal fitness
        best_neighbor_idx = neighbor_indices[0]
        best_neighbor_fitness = float('inf') if self.is_minimization else float('-inf')
        for idx in neighbor_indices:
            if self.better_than(self.particles[idx].best_fitness, best_neighbor_fitness):
                best_neighbor_fitness = self.particles[idx].best_fitness
                best_neighbor_idx = idx
        return self.particles[best_neighbor_idx].best_position

    def _calculate_diversity(self, positions):
        """Computes the average Euclidean distance of a set of positions to their centroid, used as the self-adaptive quantum cloud radius"""
        if positions is None or len(positions) == 0:
            return 0.0
        positions = np.array(positions)
        mean_position = positions.mean(axis=0)
        distances = np.sqrt(np.sum((positions - mean_position) ** 2, axis=1))
        return float(np.mean(distances))

    def _calculate_adaptive_radius(self, archive):
        """Computes r_cloud for self-adaptive QPSO. When the quantum guide comes from the archive, diversity is computed over archive members. Otherwise, r_cloud is the maximum of the neutral and quantum-particle diversities of this subswarm"""
        if self.quantum_guide in ("r", "t"):
            if archive is not None and len(archive) > 0:
                return self._calculate_diversity(archive.positions)
            return self._calculate_diversity([p.position for p in self.particles]) # fallback if the archive is empty (very first iteration)

        neutral_positions = [p.position for p in self.particles if not p.is_quantum]
        quantum_positions = [p.position for p in self.particles if p.is_quantum]
        return max(self._calculate_diversity(neutral_positions), self._calculate_diversity(quantum_positions))

    def _get_quantum_guide(self, i, archive, tournament_size):
        """Returns the guide position quantum particle i should be sampled/mutated around, based on quantum guide parameter"""
        if self.quantum_guide == "n":
            return self._get_neighborhood_best(i)

        if archive is None or len(archive) == 0:
            # no archive members available yet, fall back to nbest
            return self._get_neighborhood_best(i)

        if self.quantum_guide == "r":
            guide = archive.random_member()
        else:
            guide = archive.tournament_select(tournament_size=tournament_size)

        return guide if guide is not None else self._get_neighborhood_best(i)

    def _get_pcx_parents(self, i, archive):
        """Selects the mutation parent and the remaining nu-1 parents used by the PCX operator, based on quantum guide. For nbest, parents are drawn from this subswarm's particles, and for random and tournament parents are drawn from the bounded archive"""
        num_other_needed = self.pcx_num_parents - 1
        if self.quantum_guide == "n":
            mutation_parent = self._get_neighborhood_best(i)
            n = len(self.particles)
            k = min(num_other_needed, max(n-1, 0))
            other_indices = set()
            while len(other_indices) < k:
                idx = random.randrange(n)
                if idx != i:
                    other_indices.add(idx)
            other_parents = [self.particles[idx].best_position for idx in other_indices]
            return mutation_parent, other_parents

        if archive is None or len(archive) == 0:
            return self._get_neighborhood_best(i), []

        mutation_parent = archive.random_member() if self.quantum_guide == "r" else archive.tournament_select()
        pool_size = len(archive.positions)
        k = min(num_other_needed, max(pool_size - 1, 0))
        other_parents = []
        seen = {id(mutation_parent)}
        while len(other_parents) < k:
            candidate = archive.positions[random.randrange(pool_size)]
            if id(candidate) not in seen:
                other_parents.append(candidate)
                seen.add(id(candidate))
        return mutation_parent, other_parents
            
    def _orthonormal_basis_perpendicular_to(self, direction, k, dims):
        """Generates up to k random orthonormal vectors perpendicular to direction"""
        k = min(k, dims - 1)
        if k <= 0:
            return []
        random_matrix = np.random.normal(0, 1, size=(dims, k))
        if np.linalg.norm(direction) > 1e-12:
            random_matrix -= np.outer(direction, direction @ random_matrix)
        q, _ = np.linalg.qr(random_matrix)
        return [q[:, i] for i in range(q.shape[1])]

    def _parent_centric_crossover(self, mutation_parent, other_parents):
        """Generates a new quantum particle position using the parent-centric crossover (PCX) operator. Requires at least 2 total parents to define a meaningful search direction; otherwise falls back to a small Gaussian perturbvation around the mutation parent"""
        dims = len(self.search_bounds)
        mutation_parent = np.array(mutation_parent, dtype=float)

        if not other_parents:
            return mutation_parent + np.random.normal(0, self.pcx_sigma1, size=dims)

        other_parents = np.array(other_parents, dtype=float)
        parents = np.vstack([mutation_parent, other_parents])
        mean_position = parents.mean(axis=0)

        d = mutation_parent - mean_position
        D = np.linalg.norm(d)
        e_l = d / D if D > 1e-12 else np.zeros(dims) # prevent division by 0

        # Average perpendicular distance of the other parents to the line through mutation parent along d
        diff = other_parents - mutation_parent
        proj = np.outer(diff @ e_l, e_l)
        perpendicular = diff - proj
        D_bar = float(np.mean(np.linalg.norm(perpendicular, axis=1)))

        basis = self._orthonormal_basis_perpendicular_to(e_l, len(other_parents), dims)
        if basis:
            basis_matrix = np.column_stack(basis)
            coefficients = np.random.normal(0, self.pcx_sigma2, size=basis_matrix.shape[1])
            perpendicular_term = basis_matrix @ (coefficients * D_bar)
        else:
            perpendicular_term = np.zeros(dims)

        offspring = mutation_parent + (self.pcx_sigma1 * np.random.normal(0, 1) * d) + perpendicular_term
        return offspring

    def _sample_stable_control_parameters(self, l, max_attempts=10):
        for _ in range(max_attempts):
            w = np.random.uniform(low=0, high=1)
            c1 = np.random.uniform(low=0, high=2)
            c2 = np.random.uniform(low=0, high=2)
            c3 = np.random.uniform(low=0, high=2)
            if self._check_stability(w, l, c1, c2, c3):
                return w, c1, c2, c3
        return self.w, self.c1, self.c2, self.c3 # reuse current parameters if no parameter values generated satisfy the stability conditions


    def _check_stability(self, w, l, c1, c2, c3):
        rho = c1 + l*c2 + (1-l)*c3
        upper_bound = (4*(1-w**2)) / (1-w+((c1**2+(l**2)*(c2**2)+((1-l)**2)*(c3**2))*(1+w)/(3*rho**2)))
        if 0 < rho < upper_bound and abs(w) < 1:
            return True
        return False

    
    def step(self, archive=None, tournament_size=3):
        """Runs one optimization step, where the algorithm first reevaluates best positions if an environment change occurs, then updates velocities and positions. Includes neutral and quantum particle updates."""

        r_cloud = self._calculate_adaptive_radius(archive) if self.quantum_strategy == "adaptive" else self.quantum_radius # pre-compute r_cloud for adaptive radius QPSO so it isn't recomputed per quantum particle
        
        for i in range(len(self.particles)):
            # Neutral particle update (standard velocity-position update)
            if self.particles[i].is_quantum == False:
                
                local_best_position = self._get_neighborhood_best(i)

                # Generate w, c1, c2 and c3 if using stability-guided QMGPSO (falls back to default parameters if it cannot generate stable parameters on the first iteration)
                if self.stability_guided:
                    self.w, self.c1, self.c2, self.c3 = self._sample_stable_control_parameters(self.particles[i].l)

                # Update Velocity
                r1 = np.random.rand(len(self.particles[i].position))
                r2 = np.random.rand(len(self.particles[i].position))
                r3 = np.random.rand(len(self.particles[i].position))

                cognitive = self.c1 * r1 * (self.particles[i].best_position - self.particles[i].position)
                social = self.particles[i].l * self.c2 * r2 * (local_best_position - self.particles[i].position) # uses lbest

                if archive is not None and len(archive) > 0:
                    archive_guide = archive.tournament_select(tournament_size=tournament_size)
                    archive_pull = (1 - self.particles[i].l) * self.c3 * r3 * (archive_guide - self.particles[i].position)
                else:
                    archive_pull = 0.0

                self.particles[i].velocity = (self.w * self.particles[i].velocity) + cognitive + social + archive_pull

                # Update Position
                self.particles[i].position = self.particles[i].position + self.particles[i].velocity

            # Quantum particle update
            else:
                if self.quantum_strategy == "original":
                    y_hat = self._get_neighborhood_best(i)
                    dims = len(self.search_bounds)
                    self.particles[i].position = np.random.uniform(low=(y_hat-r_cloud), high=(y_hat+r_cloud), size=dims)
                elif self.quantum_strategy == "adaptive":
                    guide_position = self._get_quantum_guide(i, archive, tournament_size)
                    dims = len(self.search_bounds)
                    self.particles[i].position = np.random.normal(loc=guide_position, scale=max(r_cloud, 1e-12), size=dims) # ensure r_cloud cannot reach 0
                elif self.quantum_strategy == "pcx":
                    mutation_parent, other_parents = self._get_pcx_parents(i, archive)
                    self.particles[i].position = self._parent_centric_crossover(mutation_parent, other_parents)

            # Clip to Search Bounds
            low = np.array([b[0] for b in self.search_bounds])
            high = np.array([b[1] for b in self.search_bounds])
            self.particles[i].position = np.clip(self.particles[i].position, low, high)

            position = self.particles[i].get_position()
            fitness = self.objective(position)
            # print(f"Particle {i+1} | Fitness: {fitness:.4f} | Position: {position}")

            # Update Personal Best
            if self.better_than(fitness, self.particles[i].best_fitness):
                self.particles[i].best_fitness = fitness
                self.particles[i].best_position = self.particles[i].position.copy()

        # Track Global Best
        for particle in self.particles:
            if self.better_than(particle.best_fitness, self.global_best_fitness):
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.copy()

        self.history['best_fitness'].append(self.global_best_fitness)
        self.history['avg_fitness'].append(np.mean([p.best_fitness for p in self.particles]))
        
        positions = np.array([p.position for p in self.particles])
        swarm_spread = np.mean(np.std(positions, axis=0))
        self.history['diversity'].append(swarm_spread)

        self.iteration += 1

    def handle_change(self):
        """Handles parameter resampling for environment changes"""
        for p in self.particles:
            p.l = np.random.uniform(0,1) # resample balance coefficient upon environment change
            if p.is_quantum:
                p.position = np.array([np.random.uniform(low, high) for low, high in self.search_bounds]) # Quantum particles are re-initialized uniformly at random within the search domain, since they do not use their best position in the position update calculation
            p.best_position = p.position.copy() # as an environment change reaction strategy, all particles pbest values are reset to the current particle's position
            p.best_fitness = self.objective(p.position) # reevaluate particle fitness on environment change
