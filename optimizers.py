import numpy as np

class AdamOptimizer:
    def __init__(self, surface_function, gradient_function, curr_pt, num_iterations=100, lambda_v=.99, lambda_s=.9999, epsilon=1e10-8, alpha=10):
        self.surface_function = surface_function
        self.gradient_function = gradient_function
        self.curr_pt = curr_pt
        self.num_iterations = num_iterations
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.epsilon = epsilon
        self.alpha = alpha
        self.path = []

    def optimize(self):
        print("Beggining Optimizations")
        v_curr = np.zeros_like(self.curr_pt)
        s_curr = np.zeros_like(self.curr_pt)
        step = 0

        for i in range(num_iterations):
            print("Current Iteration: ", str(i+1))
            print("Computing Gradient")
            gradient = self.gradient_function(self.curr_pt[0], self.curr_pt[1])
            print("Current Gradient: ", gradient)

            v_next = (self.lambda_v * v_curr) + (1 - self.lambda_v)*gradient

            s_next = (self.lambda_s * s_curr) + (1 - self.lambda_s)*(gradient**2)

            step += 1

            corrected_v = v_next / (1 - self.lambda_v**step)

            corrected_s = s_next / (1 - self.lambda_s**step)

            self.curr_pt = self.curr_pt - (alpha*(corrected_v))/(epsilon + np.sqrt(corrected_s))

            v_curr = v_next

            s_curr = s_next

            path.append((self.curr_pt[0], self.curr_pt[1], self.surface_function(self.curr_pt[0], self.curr_pt[1])))
        
        return self.path

class SimulatedAnnealingOptimizer:
    def __init__(self, surface_function, gradient_function, curr_pt, max_iterations=1000, temperature=10, min_temp=.001, cooling_rate=.9):
        self.surface_function = surface_function
        self.gradient_function = gradient_function
        self.curr_pt = curr_pt

        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.path = []

    def optimize(self):

        for idx in range(max_iterations):
            new_position = self.solution_transition(self.curr_pt, self.temperature)
            new_energy = self.surface_function(self.curr_pt[0], self.curr_pt[1])
            alpha = self.acceptance_probability_computation(curr_energy, new_energy, self.temperature)

            if new_energy < curr_energy: 
                self.curr_pt = new_position
                curr_energy = new_energy
                self.path.append((self.curr_pt[0], self.curr_pt[1], new_energy))

            elif np.random.rand() < alpha:
                self.curr_pt = new_position
                curr_energy = new_energy
                self.path.append((self.curr_pt[0], self.curr_pt[1], new_energy))

            self.temperature *= self.cooling_rate

            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)
            if self.temperature < self.min_temp:
                break

        return self.path

    def solution_transition(self):
        new_position = self.curr_pt + np.random.normal(0, self.temperature, size = len(curr_pt))
        return new_position

    def acceptance_probability_computation(self, curr_energy, new_energy):
        if new_energy < curr_energy:
            return 1.0
        else:
            return np.exp((curr_energy - new_energy) / self.temperature )

class ParticleSwarmOptimizer:
    def __init__(self, surface_function, gradient_function, num_particles, dimensions, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.surface_function = surface_function
        self.gradient_function = gradient_function

        self.num_particles = num_particles         # Number of particles in the swarm
        self.dimensions = dimensions               # Number of dimensions in the search space
        self.max_iter = max_iter                   # Maximum number of iterations
        self.w = w                                 # Inertia weight (controls velocity)
        self.c1 = c1                               # Cognitive coefficient (individual learning factor)
        self.c2 = c2                               # Social coefficient (swarm learning factor)

        self.paths = [[] for _ in range(self.num_particles)]
        self.values = [[] for _ in range(self.num_particles)]
 
        self.positions = np.random.uniform(-100, 100, (num_particles, dimensions))  
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))  
        
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.surface_function(self.personal_best_positions[p][0], self.personal_best_positions[p][1], self.personal_best_positions[p][2]) for p in range(self.num_particles)])

        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_velocity(self, particle_idx):
        r1 = np.random.random(self.dimensions)  
        r2 = np.random.random(self.dimensions)  
        
        cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[particle_idx])
        
        new_velocity = self.w * self.velocities[particle_idx] + cognitive_velocity + social_velocity
        return new_velocity

    def update_position(self, particle_idx):
        new_position = self.positions[particle_idx] + self.velocities[particle_idx]
        return new_position

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                new_velocity = self.update_velocity(i)
                self.velocities[i] = new_velocity

                # Update position
                new_position = self.update_position(i)
                self.positions[i] = new_position
                
                print("Current Position for Agent ", i, ":", new_position)
                # Evaluate new fitness
                fitness = self.surface_function(new_position[0], new_position[1], new_position[2])
                self.paths[i].append(self.positions[i].copy())
                self.values[i].append(fitness)

                print("Current Fitness for Agent ", i, ":", fitness)

                # Update personal best if necessary
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

            # Update global best
            min_fitness_idx = np.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                self.global_best_position = self.personal_best_positions[min_fitness_idx]

            # Print progress
            #if iteration % 10 == 0:
            print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")
       
        print(self.paths)
        print(self.values)

        return self.global_best_position, self.global_best_fitness, self.paths, self.values

class SwarmBasedAnnealingOptimizer:
    def __init__(self, surface_function, gradient_function, num_particles, dimensions, max_iter, h=0.99): #, c1=1.5, c2=1.5):
        self.surface_function = surface_function
        self.gradient_function = gradient_function

        self.num_particles = num_particles         
        self.dimensions = dimensions               
        self.max_iter = max_iter                   
        self.h = h                                 
        #self.c1 = c1                               
        #self.c2 = c2                               

        self.paths = [[] for _ in range(self.num_particles)]
        self.values = [[] for _ in range(self.num_particles)]

        self.provisional_minimum = float('inf')

        # Initialize the swarm (random positions and velocities)
        self.positions = np.random.uniform(-100, 100, (self.num_particles, self.dimensions))  # Random initial positions
        self.masses = np.ones((1, self.num_particles))[0] * (1/self.num_particles)

        # Best known positions and their corresponding fitness values
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.surface_function(self.personal_best_positions[p][0], self.personal_best_positions[p][1], self.personal_best_positions[p][2]) for p in range(self.num_particles)])

        # Global best position (the best solution found by the swarm)
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_mass(self, particle_idx):
        # Update the velocity of each particle
        fitness = self.surface_function(self.positions[particle_idx][0], self.positions[particle_idx][1], self.positions[particle_idx][2])
        new_mass = self.masses[particle_idx] - (self.h*(fitness - self.provisional_minimum)*self.masses[particle_idx])
        new_mass = np.clip(new_mass, 1e-6, 1)
        return new_mass
        #return max(new_mass, 1e-6)

    def update_position(self, particle_idx, eta, iteration):
        
        gradient = self.gradient_function(self.positions[particle_idx][0], self.positions[particle_idx][1], self.positions[particle_idx][2])
        
        #inv_mass = (1/(self.masses[particle_idx]+1e-5))
        inv_mass = np.mean(self.masses)
        #print("Mass: ", inv_mass) 
        self.positions = np.clip(self.positions, -1e300, 1e-300)
        new_position = self.positions[particle_idx] - (self.h*gradient*self.surface_function(self.positions[particle_idx][0], self.positions[particle_idx][1], self.positions[particle_idx][2])) + (np.sqrt(2*self.h*inv_mass)*eta)
        #self.positions = np.clip(self.positions, -1e300, 1e-300)
        return new_position

    def provisional_min_computation(self):
        return np.sum(self.masses * np.array([self.surface_function(self.positions[y][0], self.positions[y][1], self.positions[y][2]) for y in range(self.num_particles)])) / np.sum(self.masses)

    def optimize(self):
        print("Beggining Optimization")
        self.provisional_minimum = self.provisional_min_computation()
        print("Provisional Minimum: ", self.provisional_minimum)
        
        #eta = np.random.normal(0, 1, (self.num_particles, self.dimensions)) 

        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                new_mass = self.update_mass(i)
                self.masses[i] = new_mass

            ### finished in second for here
            # Update position
            h = self.h * np.exp(-.99 * iteration)
            eta = np.random.normal(0, 1, (self.num_particles, self.dimensions)) * np.exp(-.99*iteration)

            #eta = np.random.normal(0, np.exp(-0.1 * iteration), (self.num_particles, self.dimensions))

            for i in range(self.num_particles):

                print("Initial Position for Agent ", i, ":", self.positions[i])

                new_position = self.update_position(i, eta[i], iteration)
                    
                self.positions[i] = new_position

                print("Current Position for Agent ", i, ":", new_position)

                # Evaluate new fitness
                fitness = self.surface_function(new_position[0], new_position[1], new_position[2])
                self.paths[i].append(self.positions[i].copy())
                self.values[i].append(fitness)

                print("Current Fitness for Agent ", i, ":", fitness)

                # Update personal best if necessary
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                min_fitness_idx = np.argmin(self.personal_best_fitness)
                if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                    self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                    self.global_best_position = self.personal_best_positions[min_fitness_idx]

            self.provisional_minimum = self.provisional_min_computation()
            print("Provisional Minimum: ", self.provisional_minimum)

            print(self.masses)

            # Print progress
            print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")
        print("Completed Optimization")
        #min_idx = np.argmin([self.surface_function(self.positions[y][0], self.positions[y][1], self.positions[y][2]) for y in range(self.num_particles)])
        #min_pos = self.positions[min_idx]
        #minima = self.surface_function(min_pos[0], min_pos[1], min_pos[2])
        return self.global_best_position, self.global_best_fitness, self.paths, self.values

