import numpy as np
from bsp4 import *
import math
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.setrecursionlimit(3000)

class AdamOptimizer:
    def __init__(self, surface_function, gradient_function, curr_pt=[0,0,0], num_iterations=1000, lambda_v=.99, lambda_s=.9999, epsilon=1e10-8, alpha=10):
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
        min_position = self.curr_pt
        min_error = float("inf")

        for i in range(self.num_iterations):
            print("Current Iteration: ", str(i+1))
            print("Computing Gradient")
            gradient = self.gradient_function(self.curr_pt[0], self.curr_pt[1], self.curr_pt[2])
            print("Current Gradient: ", gradient)

            v_next = (self.lambda_v * v_curr) + (1 - self.lambda_v)*gradient
            s_next = (self.lambda_s * s_curr) + (1 - self.lambda_s)*(gradient**2)
            step += 1
            corrected_v = v_next / (1 - self.lambda_v**step)
            corrected_s = s_next / (1 - self.lambda_s**step)
            
            print("Current Position: ", self.curr_pt)
            self.curr_pt = self.curr_pt - (self.alpha*(corrected_v))/(self.epsilon + np.sqrt(corrected_s))
            print("Updated Position: ", self.curr_pt)
            curr_error = self.surface_function(self.curr_pt[0], self.curr_pt[1], self.curr_pt[2])

            v_curr = v_next
            s_curr = s_next

            self.path.append((self.curr_pt[0], self.curr_pt[1], self.curr_pt[2], self.surface_function(self.curr_pt[0], self.curr_pt[1], self.curr_pt[2])))
        
            if curr_error <= min_error:
                min_error = curr_error
                min_position = self.curr_pt
		
        print("ADAM Final Error: ", min_error)
        print("ADAM Final Position: ", min_position)
        self.plot(self.path, min_position)
        return min_position, min_error, self.path

    def plot(self, path, min_position):
        # Prepare the surface for plotting
        x_vals = np.linspace(-5, 5, 100)
        y_vals = np.linspace(-5, 5, 100)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        z_vals = np.vectorize(lambda x, y: self.surface_function(x, y, 0))(x_grid, y_grid)

        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=z_vals, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.8)])

        # Plot the optimization path
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        path_z = [point[3] for point in path]  # z values are the function values at each point

        fig.add_trace(go.Scatter3d(
            x=path_x, 
            y=path_y, 
            z=path_z, 
            mode='markers+lines', 
            marker=dict(size=1, color='red', opacity=0.7), 
            line=dict(width=1, color='red')
        ))
        '''
        # Mark the minima with a label
        fig.add_trace(go.Scatter3d(
            x=[min_position[0]],
            y=[min_position[1]],
            z=[self.surface_function(min_position[0], min_position[1], min_position[2])],
            mode='markers+text',
            marker=dict(size=10, color='green'),
            text=['Minimum'],
            textposition='top center'
        ))
	'''
        # Labels and layout
        fig.update_layout(
            title="Adam Optimizer",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Error',
            ),
            autosize=True
        )

        # Save the figure to an HTML file
        fig.write_html("adam_optimization_booth.html")
        fig.show()

#fix this, changed it to ctsp specific
class SimulatedAnnealingOptimizer:
    def __init__(self, ctsp_obj, curr_pt=None, init_error=None, max_iterations=1000, temperature=10, min_temp=.001, cooling_rate=.90, k=1):
        self.ctsp_obj = ctsp_obj
        self.curr_pt = self.get_initial_pt(curr_pt)
        self.init_error = init_error

        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.path = []

    def get_initial_pt(self, curr_pt):
        if not isinstance(curr_pt, np.ndarray):
            curr_pt = np.zeros(len(self.ctsp_obj.num_cities))
        else:
            curr_pt = curr_pt
        
        return curr_pt

    def optimize(self):

        print("Beggining Simulated Annealing Optimization")
        update_ctr = 0
        if self.init_error != None:
            curr_energy = self.init_error
        else:
            curr_energy = self.ctsp_obj.total_distance(self.curr_pt)
        for idx in range(self.max_iterations):
            new_position = self.solution_transition(self.curr_pt)
            new_energy = self.ctsp_obj.total_distance(new_position)
            print("Potential New Postition: ", new_position)
            print("Potential New Postition Error: ", new_energy)

            alpha = self.acceptance_probability_computation(curr_energy, new_energy)
            
            if new_energy < curr_energy: 
                self.curr_pt = new_position
                curr_energy = new_energy
                #self.path.append((self.curr_pt[0], self.curr_pt[1], self.curr_pt[2], new_energy))
                update_ctr = 0
            #elif np.random.rand() > alpha:
            elif np.random.rand() > (1-alpha):            
                self.curr_pt = new_position
                curr_energy = new_energy
                #self.path.append((self.curr_pt[0], self.curr_pt[1], self.curr_pt[2], new_energy))
                update_ctr = 0
            else:
                update_ctr += 1
            self.temperature *= self.cooling_rate

            print("Current Position: ", self.curr_pt)
            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)

            if self.temperature < self.min_temp or update_ctr == 10:
                print("Early Convergence, Breaking")
                break
        
        print("Final Error: ", curr_energy)
        print("Final Position: ", self.curr_pt)
        return  self.curr_pt, curr_energy, self.path


    def solution_transition(self, tour):
        new_position = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        new_position[i], new_position[j] = new_position[j], new_position[i]
        #new_position = np.random.permutation(len(tour)) 
        print(i, j)
        print(new_position)
        return new_position


    def acceptance_probability_computation(self, curr_energy, new_energy):
        if new_energy < curr_energy:
            return 1.0
        elif new_energy == curr_energy:
            return 0
        else:
            return np.exp(-((new_energy - curr_energy) / (self.temperature*.8)) )

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
        print("Completed Optimization")
        #min_idx = np.argmin([self.surface_function(self.positions[y][0], self.positions[y][1], self.positions[y][2]) for y in range(self.num_particles)])
        #min_pos = self.positions[min_idx]
        #minima = self.surface_function(min_pos[0], min_pos[1], min_pos[2])
        return self.global_best_position, self.global_best_fitness, self.paths, self.values

class HdFireflySimulatedAnnealingOptimizer:
    def __init__(self, surface_function,  dimensions, pop_test=100, hdfa_iterations=100, gamma=1, alpha=.2): 
        self.objective_computation = surface_function

        self.pop_test = pop_test 
        self.dimensions = dimensions
        self.hdfa_iterations = hdfa_iterations
        self.alpha = alpha
        self.gamma = gamma

        self.bounds = [-50, -50, -50, 50, 50, 50]
        self.max_depth = 3

        self.pop_positions = self.initialize_positions('initial', -45, 45)
        self.pop_attractiveness = np.ones(self.pop_test)
        self.pop_fitness_true = np.zeros(self.pop_test)
        self.pop_fitness_approx = np.random.uniform(-200, 200, pop_test)
        self.pop_alpha = np.ones(self.pop_test)

        self.firefly_positions_history = np.zeros((self.hdfa_iterations, self.pop_test, self.dimensions))
        self.minima_positions = [] 
        self.minima_fitness = []   
        self.fitness_history = []  


        #print("Init Po :", self.pop_positions)

        self.compute_fitness_true()

        self.bsp_tree = generate_bsp_tree(self.bounds, 0, self.max_depth)
        #print(self.bsp_tree) 
        '''
        root  = self.bsp_tree

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  # Equal scaling on all axes
    
        # Plot the BSP tree
        plot_bsp_tree(root, ax)
        plt.show()
        '''


    #def initialize_bsp(self):
    #    bsp = BSPTree(self.objective_computation)
    #    bsp.initialize_bsp_tree(-200, 200, 20)
   #     return bsp

    def compute_fitness_true(self):
        for idx in range(self.pop_test):
            #print("Initializing Firefly: ", idx)
            self.pop_fitness_true[idx] = self.objective_computation(self.pop_positions[idx][0], self.pop_positions[idx][1], self.pop_positions[idx][2])

    def compute_fitness_approx(self):
        for idx in range(self.pop_test):
            #print("Initializing Firefly: ", idx)
            self.pop_fitness_approx[idx] = self.objective_computation(self.pop_positions[idx][0], self.pop_positions[idx][1], self.pop_positions[idx][2])

    def initialize_positions(self, stage, low=None, high=None):
        if stage == 'initial':
            return low + (high - low) * np.random.rand(self.pop_test, self.dimensions)
        elif stage == 'finder_tracker':
            return self.finder_tracker_assignments()    
    
    def l2_norm(self, ff_idx_1, ff_idx_2):
        return np.sqrt(np.sum((self.pop_positions[ff_idx_1] - self.pop_positions[ff_idx_2])**2))

    def compute_attractiveness(self, idx1, idx2):
        #for idx1 in range(self.pop_test):        
            #for idx2 in range(self.pop_test):
        norm = self.l2_norm(idx1, idx2)#**2
        self.pop_attractiveness[idx1] = self.pop_attractiveness[idx1] * np.exp(-self.gamma*norm)
        return self.pop_attractiveness[idx1]        

    def compute_positions(self):
        for idx1 in range(self.pop_test):
            for idx2 in range(self.pop_test):
                if self.pop_fitness_true[idx1] < self.pop_fitness_true[idx2]:
                    new_attr = self.compute_attractiveness(idx1, idx2)
                    self.pop_positions[idx1] = self.pop_positions[idx1] + new_attr * (self.pop_positions[idx2] - self.pop_positions[idx1]) + self.pop_alpha[idx1]*(np.random.rand()-.5)

    def finder_tracker_assignments(self, tol=.5):
        print("Reassigning Finder Trackers")

        print("No Reassignments Needed")

    def optimize(self):
        print("Beggining Hd-Firefly-SA Optimization")
        last_alpha = float('inf')
        maturity_condition = False
        nonincreasing_alpha_counter = 0
        hdfa_ctr = 0
        min_reg_fitness = float('inf')
        new_fitness = float('inf')
        best_position = None
        best_fitness = float('inf')
        best_region = None
        while not maturity_condition:
            iteration_fitness = []        
            while hdfa_ctr < self.hdfa_iterations:
                print("Current HdFa Iteration: ", hdfa_ctr)

                #print("Initial Positions: ", self.pop_fitness_true[0])

                self.compute_positions()
                self.compute_fitness_true()
                                
                #print("Updated_poisitions: ", self.pop_fitness_approx[0])

                for idx in range(self.pop_test):
                    #s_best_region, fitness_approx, region = self.bsp_tree.compare_with_neighbors(self.pop_positions[idx])                
                    #root = self.bsp_tree
                    
                    #print("Position: ", self.bsp_tree.fitness)

                    region = find_region(self.bsp_tree, self.pop_positions[idx])
                    
                    if region is None:
                        continue

                    #print_bsp_tree(region)

                    #print('region: ', region.left) 

                    neighbors = find_neighbors(region, 5)

                    #print('neighbors: ', neighbors)

                    is_best_region = find_min_fitness_region(region, neighbors, self.pop_positions[idx])

                    print("is bnest region: ", is_best_region ==  region)

                    if is_best_region == region: 
                        new_fitness = self.objective_computation(self.pop_positions[idx][0], self.pop_positions[idx][1], self.pop_positions[idx][2])
                        print(new_fitness, " ", region.fitness)
                        if new_fitness < region.fitness:
                            #print("one alphaa ", idx, " :", self.pop_alpha[idx])
                            #self.pop_alpha[idx] = np.abs(new_fitness - is_best_region.fitness) 
                            region.fitness = new_fitness
                            #self.pop_fitness_true = new_fitness

                        #print("one alphaa ", idx, " :", self.pop_alpha[idx])
                        self.pop_alpha[idx] = np.abs(new_fitness - is_best_region.fitness)

                        region.split(new_fitness)
                        #self.pop_fitness_true[idx] = new_fitness
                        #print("new fitness, best fitness: ", new_fitness, " ", best_fitness)
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_position = self.pop_positions[idx]
                            best_region = region
                                     
                    else:
                        self.pop_fitness_true[idx] = region.fitness
                        
                    iteration_fitness.append(self.pop_fitness_true[idx])                    
                    #self.pop_alpha[idx] = np.abs(self.pop_fitness_true[idx] - is_best_region.fitness)    


                    print("alphaa ", idx, " :", self.pop_alpha[idx])
                #print("pop_fitness_true: ", self.pop_fitness_true)            
                #print("pop_fitness_approx: ", self.pop_fitness_approx)    
                alpha_avg = np.average(self.pop_alpha)    
                self.firefly_positions_history[hdfa_ctr] = self.pop_positions.copy()
                self.fitness_history.append(np.min(iteration_fitness))
                self.minima_positions.append(best_position)
                self.minima_fitness.append(best_fitness) 
                hdfa_ctr += 1
                print("Current Alpha Average: ", alpha_avg)



            print('HDFA Iteration: ', hdfa_ctr)
            print('Steps Without Increasing Alpha: ', nonincreasing_alpha_counter)

            if last_alpha >= alpha_avg:
                nonincreasing_alpha_counter += 1
            else:
                nonincreasing_alpha_counter = 0

            if nonincreasing_alpha_counter == 10 or alpha_avg == 0 or math.isnan(nonincreasing_alpha_counter):
                maturity_condition = True                    
                break                    
        
        best_region = find_min_fitness_region_recursive(self.bsp_tree)
        
        best_pt_est = find_bounding_box_center(best_region.bounds)

        print("Final Hd-FF Position Estimate: ", best_pt_est)
        print("Final Hd-FF Error: ", best_region.fitness)
        sa = SimulatedAnnealingOptimizer(self.objective_computation, best_pt_est, temperature=5, cooling_rate = .95)
        
        sa_min_pt, sa_min_fitness, sa_path = sa.optimize()

        print("Final SA Min Position: ", sa_min_pt)
        print("Final SA Error: ", sa_min_fitness)

        return best_position, best_fitness, self.firefly_positions_history, self.fitness_history, self.minima_positions, self.minima_fitness, sa_min_pt, sa_min_fitness, sa_path

