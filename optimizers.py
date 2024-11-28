import numpy as np
from bsp2 import *

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
    def __init__(self, surface_function, curr_pt=None, max_iterations=1000, temperature=10, min_temp=.001, cooling_rate=.9):
        self.surface_function = surface_function
        self.curr_pt = self.get_initial_pt(curr_pt)

        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.path = []

    def get_initial_pt(self, curr_pt):
        if not isinstance(curr_pt, np.ndarray):
            curr_pt = [0,0,0]
        else:
            curr_pt = curr_pt
        
        return curr_pt

    def optimize(self):

        print("Initial Point: ", self.curr_pt)
        update_ctr = 0
        curr_energy = self.surface_function(self.curr_pt[0], self.curr_pt[1], self.curr_pt[2])
        for idx in range(self.max_iterations):
            new_position = self.solution_transition(self.curr_pt)
            new_energy = self.surface_function(self.curr_pt[0], self.curr_pt[1], self.curr_pt[2])
            alpha = self.acceptance_probability_computation(curr_energy, new_energy)

            print("Acceptance Probability: ", alpha)

            if new_energy < curr_energy: 
                self.curr_pt = new_position
                curr_energy = new_energy
                self.path.append((self.curr_pt[0], self.curr_pt[1], self.curr_pt[2], new_energy))
                update_ctr = 0
            elif np.random.rand() > alpha:
                self.curr_pt = new_position
                curr_energy = new_energy
                self.path.append((self.curr_pt[0], self.curr_pt[1], self.curr_pt[2], new_energy))
                update_ctr = 0
            else:
                update_ctr += 1
            self.temperature *= self.cooling_rate

            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)
            if self.temperature < self.min_temp or update_ctr == 10:
                break

        return  self.curr_pt, curr_energy, self.path

    def solution_transition(self, curr_pt):
        new_position = curr_pt + np.random.normal(0, self.temperature, size = len(curr_pt))
        return new_position

    def acceptance_probability_computation(self, curr_energy, new_energy):
        if new_energy < curr_energy:
            return 1.0
        else:
            return np.exp(-((new_energy - curr_energy) / self.temperature) )

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

class HdFireflySimulatedAnnealingOptimizer:
    def __init__(self, surface_function,  dimensions, pop_test=100, hdfa_iterations=5, gamma=1, alpha=.2): 
        self.objective_computation = surface_function

        self.pop_test = pop_test 
        self.dimensions = dimensions
        self.hdfa_iterations = hdfa_iterations
        self.alpha = alpha
        self.gamma = gamma

        self.pop_positions = np.random.rand(self.pop_test, dimensions)
        self.pop_attractiveness = np.ones(self.pop_test)
        self.pop_fitness = np.zeros(self.pop_test)
        #print(self.pop_fitness)
        self.pop_alpha = np.zeros(self.pop_test)

        self.initialize_fitness()

        self.bsp_tree = self.initialize_bsp()
        
        #self.sort_pop_arrays()

    #def sort_pop_arrays(self):
    #    sorted_arrays = sorted(zip(self.pop_fitness, self.pop_positions, self.pop_attractiveness)
    #    self.pop_fitness, self.pop_positions, self.pop_attractiveness = zip(*sorted_arrays)

    def initialize_bsp(self):
        bsp = BSP(self.dimensions)
        bsp.build_tree(self.pop_positions, self.pop_fitness)
        return bsp

    def initialize_fitness(self):
        for idx in range(self.pop_test):
            self.pop_fitness[idx] = self.objective_computation(self.pop_positions[idx][0], self.pop_positions[idx][1], self.pop_positions[idx][2])

    def l2_norm(self, ff_idx_1, ff_idx_2):
        return np.sqrt(np.sum((self.pop_positions[ff_idx_1] - self.pop_positions[ff_idx_2])**2))

    def compute_attractiveness(self, ff_idx_1, ff_idx_2):
        norm = self.l2_norm(ff_idx_1, ff_idx_2)**2
        return self.pop_attractiveness[ff_idx_1] * np.exp(-self.gamma*norm)

    def update_position(self, new_attr, ff_idx_1, ff_idx_2):
        #attractiveness = self.compute_attractiveness(ff_idx_1, ff_idx_2)
        return self.pop_positions[ff_idx_1] + new_attr * (self.pop_positions[ff_idx_2] - self.pop_positions[ff_idx_1]) + self.alpha*(np.random.rand()-.5)

    def update_fitness(self, ff_idx_1):
        self.pop_fitness[ff_idx_1] = self.objective_computation(self.pop_positions[ff_idx_1][0], self.pop_positions[ff_idx_1][1], self.pop_positions[ff_idx_1][2])
    
    '''    
    def grow_bsp(self, points, fitness_scores):
        bsp = BSP(self.dimensions, self.bsp_tree)
        self.bsp_tree = bsp.grow_tree(points, fitness_scores)
        #self.bsp_tree.visualize_tree()

    def get_region_points(self, point, fitness):
        bsp = BSP(self.dimensions, self.bsp_tree)
        return bsp.find_region_points(point, fitness)
    '''

    def optimize(self):
        last_alpha = float('inf')
        maturity_condition = True
        nonincreasing_alpha_counter = 0
        min_position = None
        min_fitness = float('inf')
        hdfa_ctr = 0
        min_reg_fitness = float('inf')
        while hdfa_ctr < self.hdfa_iterations:
            print("in while")
            for idx1 in range(self.pop_test):
                for idx2 in range(self.pop_test):
                    print(idx1, " ", idx2)
                    print(self.pop_fitness[idx1], " ", self.pop_fitness[idx2])
                    #print("len of bsp tree: ", len(self.bsp_tree.tree.nodes))
                    #self.bsp_tree.visualize_tree()
                    if self.pop_fitness[idx1] < self.pop_fitness[idx2]:
                        new_attr = self.compute_attractiveness(idx1, idx2)
                        new_position = self.update_position(new_attr, idx1, idx2)                     
                        in_min_region, min_region, min_reg_fitness, new_graph, min_node, region_points = find_region_with_lowest_fitness(self.objective_computation, self.bsp_tree, new_attr, new_position, self.dimensions)
                        print("Min region: ", region_points)
                        new_fitness = self.objective_computation(min_region[0][0], min_region[0][1], min_region[0][2])
                        print("Potential New Position: ", new_position)
                        print("Potential New Position Error: ", new_fitness) 
                        #self.pop_positions[idx1] = new_position

                        if new_fitness < self.pop_fitness[idx1]:
                            #print("New Tree: ", self.bsp_tree)
                            self.pop_fitness[idx1] = new_fitness
                            self.pop_positions[idx1] = new_position
                            #region_points = self.get_region_points(new_position, new_fitness)
                            #print(region_points)
                            #self.bsp_tree(new_position, new_fitness)
                        
                        print("Curr Position: ", self.pop_positions[idx1])
                        print("Curr Fitness: ", self.pop_fitness[idx1])

                if maturity_condition:
                    if min_reg_fitness == float('inf'):
                        min_reg_fitness = self.pop_fitness[idx1]
                    self.pop_alpha[idx1] = np.abs(min_reg_fitness - min_fitness)
                    if idx1 > 1:
                        alpha_avg = np.average(self.pop_alpha[:idx1])
                    else:
                        alpha_avg = 1
                    print("Current Alpha Average: ", alpha_avg)
                    if alpha_avg <= 0 or nonincreasing_alpha_counter >= 10:
                        break
                    if last_alpha > alpha_avg:  
                        nonincreasing_alpha_counter += 1
                    else:
                        nonincreasing_alpha_counter = 0

                    last_alpha = alpha_avg
            hdfa_ctr += 1
        #print(in_min_region, " ", min_region, " ", min_reg_fitness)

        _, min_position, _ = self.bsp_tree.find_lowest_fitness_region()

        print("Final Min Position: ", min_position[0])

        sa = SimulatedAnnealingOptimizer(self.objective_computation, min_position[0], temperature=10, cooling_rate = .95)
        
        min_pt, min_fitness, path = sa.optimize()

        return min_pt, min_fitness, path

       

        

