import numpy as np
import operator
import random

from particle import Particle

class PSO:
    def __init__(self, num_particles, search_bounds, objective, w=0.729844, c1=1.496180, c2=1.496180, neighborhood_size=3, is_minimization=True, quantum_proportion=0.5, quantum_radius=1.0):
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

        self.v_max = np.array([(high - low) * 0.2 for low, high in search_bounds])
        self.v_min = -self.v_max

        self.global_best_position = None
        self.w, self.c1, self.c2 = w, c1, c2
        self.neighborhood_size = neighborhood_size

        self.quantum_radius = quantum_radius

        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

    def initialize(self):
        # Initial Evaluation
        print("Evaluating Initial Population")
        for i in range(len(self.particles)):
            print(f"Particle {i+1}")
            position = self.particles[i].get_position()

            fitness = self.objective(position)
            print(f"Fitness: {fitness:.4f} | Position: {position}")

            # Set Initial Personal Best
            self.particles[i].best_fitness = fitness
            self.particles[i].best_position = self.particles[i].position.copy()

            # Set Initial Global Best
            if self.better_than(fitness, self.global_best_fitness):
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_fitness = fitness

    def step(self):
        # Reactive change detection
        if self.iteration > 0 and self.global_best_position is not None:
            # Evaluate objective function at last known global best position
            current_global_fitness = self.objective(self.global_best_position)

            # Check if value is different from last iteration
            if abs(current_global_fitness - self.global_best_fitness) > 1e-8:
                # If so, re-evaluate personal bests of each particle
                self.global_best_fitness = float('inf') if self.is_minimization else float('-inf')
                for particle in self.particles:
                    particle.best_fitness = self.objective(particle.best_position)
                    if self.better_than(particle.best_fitness, self.global_best_fitness):
                        self.global_best_position = particle.best_position.copy()
                        self.global_best_fitness = particle.best_fitness
        
        # Update velocities using local best
        for i in range(len(self.particles)):
            # Neutral particle update (standard velocity-position update)
            if self.particles[i].is_quantum == False:
                # Find neighbor indices of current particle
                neighbor_indices = [(i + j) % len(self.particles) for j in range(-(self.neighborhood_size // 2), (self.neighborhood_size // 2) + 1)]

                # Find which local particle has the best personal fitness
                best_neighbor_idx = neighbor_indices[0]
                best_neighbor_fitness = float('inf') if self.is_minimization else float('-inf')
                for idx in neighbor_indices:
                    if self.better_than(self.particles[idx].best_fitness, best_neighbor_fitness):
                        best_neighbor_fitness = self.particles[idx].best_fitness
                        best_neighbor_idx = idx
                local_best_position = self.particles[best_neighbor_idx].best_position

                # Update Velocity
                r1 = np.random.rand(len(self.particles[i].position))
                r2 = np.random.rand(len(self.particles[i].position))

                cognitive = self.c1 * r1 * (self.particles[i].best_position - self.particles[i].position)
                social = self.c2 * r2 * (local_best_position - self.particles[i].position) # uses lbest

                new_velocity = (self.w * self.particles[i].velocity + cognitive + social)
                
                self.particles[i].velocity = np.clip(new_velocity, self.v_min, self.v_max) # Apply velocity clamping

                # Update Position
                self.particles[i].position = self.particles[i].position + self.particles[i].velocity

            # Quantum particle update
            else:
                y_hat = self.global_best_position.copy()
                r_cloud = self.quantum_radius
                dims = len(self.search_bounds)
                self.particles[i].position = np.random.uniform(low=(y_hat-r_cloud), high=(y_hat+r_cloud), size=dims)

            position = self.particles[i].get_position()
            fitness = self.objective(position)
            print(f"Particle {i+1} | Fitness: {fitness:.4f} | Position: {position}")

            # Clip to Search Bounds
            low = np.array([b[0] for b in self.search_bounds])
            high = np.array([b[1] for b in self.search_bounds])
            self.particles[i].position = np.clip(self.particles[i].position, low, high)

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
