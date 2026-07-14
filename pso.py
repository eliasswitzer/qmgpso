import numpy as np
import operator
import random

from particle import Particle

class PSO:
    def __init__(self, num_particles, search_bounds, objective, w=0.729844, c1=1.496180, c2=1.496180, is_minimization=True):
        
        self.particles = []
        for i in range(num_particles):
            self.particles.append(Particle(search_bounds=search_bounds))

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

    def optimize(self, num_iterations, patience, neighborhood_size, dynamic_env=None, change_interval=200):
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

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
                
        # Main PSO Loop
        no_improvement_count = 0
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            current_best_fitness = self.global_best_fitness

            # Reactive change detection
            if iteration > 0 and self.global_best_position is not None:
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

            # Environment Change Trigger
            if dynamic_env is not None and iteration > 0 and iteration % change_interval == 0: #TODO: expand this to include above code and turn above code into sentinel check
                print(f"Environment has changed!")
                dynamic_env.change_environment()
            
            # Update velocities using local best
            for i in range(len(self.particles)):
                # Find neighbor indices of current particle
                neighbor_indices = [(i + j) % len(self.particles) for j in range(-(neighborhood_size // 2), (neighborhood_size // 2) + 1)]

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

            history['best_fitness'].append(self.global_best_fitness)
            history['avg_fitness'].append(np.mean([p.best_fitness for p in self.particles]))
            
            positions = np.array([p.position for p in self.particles])
            swarm_spread = np.mean(np.std(positions, axis=0))
            history['diversity'].append(swarm_spread)

            # Early Stopping (fitness stagnation)
            if (self.global_best_fitness - current_best_fitness) < 1e-4:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= patience:
                print("Fitness improvement has stagnated, stopping early!")
                break

            # Early Stopping (particle distance)
            # if swarm_spread < 1e-2:
            #     print("Swarm has converged, stopping early!")
            #     break

        best_position = self.global_best_position
        return best_position, history
