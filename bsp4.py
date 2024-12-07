import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class BSPNode:
    def __init__(self, bounds, axis=None, split_value=None, left=None, right=None, fitness=None, parent=None):
        self.bounds = bounds  # The bounding box for this node (x_min, y_min, z_min, x_max, y_max, z_max)
        self.axis = axis  # The axis along which the split is done (0: x, 1: y, 2: z)
        self.split_value = split_value  # The value at which the space is split (midpoint)
        self.left = left  # Left child
        self.right = right  # Right child
        self.fitness = fitness  # Random fitness value assigned to this node
        self.parent = parent  # Parent node (for updating relationships)

    def update_fitness(self, new_fitness):
        """Method to update the fitness of the node dynamically."""
        # For example, use the size of the region to update fitness (this is just an example).
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds
        #volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        self.fitness = new_fitness  #1 / (volume + 1)  # Smaller volumes get higher fitness values
        if self.parent:
            self.parent.update_fitness(new_fitness)  # Propagate fitness update upwards to the parent

    def split(self, new_fitness):
        """Split the current region into two new regions."""
        # Randomly choose an axis for the split
        self.axis = random.choice([0, 1, 2])  # Choose axis (0: x, 1: y, 2: z)
        
        # Calculate the midpoint for the split
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds
        split_value = (x_min + x_max) / 2 if self.axis == 0 else (y_min + y_max) / 2 if self.axis == 1 else (z_min + z_max) / 2
        self.split_value = split_value

        # Split bounds into two parts
        left_bounds = list(self.bounds)
        right_bounds = list(self.bounds)
        
        if self.axis == 0:
            left_bounds[3] = split_value  # Split the x_max
            right_bounds[0] = split_value  # Split the x_min
        elif self.axis == 1:
            left_bounds[4] = split_value  # Split the y_max
            right_bounds[1] = split_value  # Split the y_min
        else:
            left_bounds[5] = split_value  # Split the z_max
            right_bounds[2] = split_value  # Split the z_min
        
        # Create left and right children
        self.left = BSPNode(left_bounds, fitness=random.random(), parent=self)
        self.right = BSPNode(right_bounds, fitness=random.random(), parent=self)
        
        # Update the fitness of the current node after splitting
        self.update_fitness(new_fitness)

def generate_bsp_tree(bounds, depth, max_depth):
    if depth > max_depth:
        return None
    
    #print(f"Generating tree at depth {depth} with bounds {bounds}") 

    # Randomly choose an axis (0: x, 1: y, 2: z)
    axis = random.choice([0, 1, 2])
    
    # Calculate the midpoint along the chosen axis
    min_val, max_val = bounds[axis], bounds[axis + 3]
    split_value = (min_val + max_val) / 2
    
    # Split the bounds into two halves
    left_bounds = list(bounds)
    left_bounds[axis + 3] = split_value  # Update the max value along the split axis
    
    right_bounds = list(bounds)
    right_bounds[axis] = split_value  # Update the min value along the split axis
    
    # Create the current node with a random fitness value
    fitness = random.random()
    
    # Recursively generate left and right children
    left_node = generate_bsp_tree(left_bounds, depth + 1, max_depth)
    right_node = generate_bsp_tree(right_bounds, depth + 1, max_depth)
    
    return BSPNode(bounds, axis, split_value, left_node, right_node, fitness)

def find_region(node, point):
    """Recursively finds the region to which the given point belongs."""
    if node is None:
        return None
    
    x_min, y_min, z_min, x_max, y_max, z_max = node.bounds
    
    # Print debug information to verif  bounds
    #print(f"Checking point {point} against bounds {node.bounds}")
    
    # Check if the point is inside the current node's bounds
    x, y, z = point
    if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
        #print(f"Point {point} is inside bounds {node.bounds}")
        
        # If the point is inside the bounds, return the current node directly
        if node.left is None and node.right is None:
            return node  # This is a leaf node, so return this node
        
        # If not a leaf, we will go deeper into the tree depending on the axis
        if node.axis == 0:  # Split along x-axis
            if x < node.split_value:  # Check strictly less than split value
                return find_region(node.left, point)
            else:
                return find_region(node.right, point)
        elif node.axis == 1:  # Split along y-axis
            if y < node.split_value:  # Check strictly less than split value
                return find_region(node.left, point)
            else:
                return find_region(node.right, point)
        elif node.axis == 2:  # Split along z-axis
            if z < node.split_value:  # Check strictly less than split value
                return find_region(node.left, point)
            else:
                return find_region(node.right, point)
    
    return None

def find_neighbors(node, n):
    """Finds neighboring regions of a given node and returns them."""
    neighbors = []
    #print("Node: ", node.left)
    # Check if the node has left and right children and add them as neighbors
    if node is not None:
        if node.left is not None: 
            neighbors.append(node.left)
        if node.right is not None:
            neighbors.append(node.right)
    # Limit the number of neighbors to 'n'
    return neighbors[:n]

def find_min_fitness_region(node, neighbors, point):
    """Finds the region with the minimum fitness value from neighbors and checks if the point is inside it."""
    min_fitness = float('inf')
    best_region = None
    
    if neighbors == []:
        best_region = node

    for neighbor in neighbors:
        if neighbor.fitness < min_fitness and is_point_in_region(neighbor, point):
            min_fitness = neighbor.fitness
            best_region = neighbor
    
    return best_region

def is_point_in_region(region, point):
    """Checks if the point lies within the given region's bounds."""
    x, y, z = point
    x_min, y_min, z_min, x_max, y_max, z_max = region.bounds
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

def find_min_fitness_region_recursive(node):
    """Recursively traverses the BSP tree and finds the region with the minimum fitness value."""
    if node is None:
        return None
    
    # Initialize the best region as the current node
    min_fitness_region = node
    
    # Traverse the left subtree if it exists
    if node.left:
        left_region = find_min_fitness_region_recursive(node.left)
        if left_region and left_region.fitness < min_fitness_region.fitness:
            min_fitness_region = left_region
    
    # Traverse the right subtree if it exists
    if node.right:
        right_region = find_min_fitness_region_recursive(node.right)
        if right_region and right_region.fitness < min_fitness_region.fitness:
            min_fitness_region = right_region
    
    # Return the region with the minimum fitness value
    return min_fitness_region

def find_bounding_box_center(bounds):
    """Finds the center of a 3D bounding box.
    
    Args:
        bounds (tuple): A tuple representing the bounding box (x_min, y_min, z_min, x_max, y_max, z_max)
        
    Returns:
        tuple: A tuple representing the center of the bounding box (center_x, center_y, center_z)
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    
    # Calculate the center along each axis
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    
    return [center_x, center_y, center_z]

def plot_bsp_tree(node, ax):
    if node is None:
        return
    
    # Extract the bounds of the current region
    x_min, y_min, z_min, x_max, y_max, z_max = node.bounds
    
    # Plot the current region as a cuboid
    cuboid = [
        [(x_min, y_min, z_min), (x_min, y_min, z_max), (x_min, y_max, z_max), (x_min, y_max, z_min)],  # Left face
        [(x_max, y_min, z_min), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_max, y_max, z_min)],  # Right face
        [(x_min, y_min, z_min), (x_min, y_max, z_min), (x_max, y_max, z_min), (x_max, y_min, z_min)],  # Bottom face
        [(x_min, y_min, z_max), (x_min, y_max, z_max), (x_max, y_max, z_max), (x_max, y_min, z_max)],  # Top face
        [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_min, z_max), (x_min, y_min, z_max)],  # Front face
        [(x_min, y_max, z_min), (x_max, y_max, z_min), (x_max, y_max, z_max), (x_min, y_max, z_max)]   # Back face
    ]
    
    # Create the 3D polygons for each face
    for face in cuboid:
        poly = Poly3DCollection([face], color=(random.random(), random.random(), random.random()), alpha=0.1)
        ax.add_collection3d(poly)
    
    # Visualize the fitness value in the center of the region
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    ax.text(center[0], center[1], center[2], f'Fitness: {node.fitness:.2f}', fontsize=8, color='black', horizontalalignment='center')
    
    # Recursively plot left and right children
    plot_bsp_tree(node.left, ax)
    plot_bsp_tree(node.right, ax)
'''
def main():
    # Define the initial bounds (3D range)
    bounds = [-200, -200, -200, 200, 200, 200]
    
    # Define the maximum depth for splitting (controls the number of subdivisions)
    max_depth = 8
    
    # Generate the BSP tree
    root = generate_bsp_tree(bounds, 0, max_depth)
    
    # Define a point to search for
    
    point = [1, 1, 3]  # Example point
    
    # Find the region that the point belongs to
    region = find_region(root, point)
    if region is not None:
        print(f"Point {point} belongs to region with fitness value: {region.fitness}")
    else:
        print(f"Point {point} does not belong to any region.")
        return  # Stop execution if the point is not inside any region
    
    # Find neighbors for the region
    neighbors = find_neighbors(region, 3)  # Get 3 neighbors
    
    # Find the neighboring region with the minimum fitness value
    min_fitness_region = find_min_fitness_region(root, neighbors, point)
    
    # Check if the point is inside the region with minimum fitness
    if min_fitness_region:
        print(f"Point {point} is inside the region with minimum fitness: {min_fitness_region.fitness:.2f}")
    else:
        print(f"Point {point} is not inside any neighboring region with minimum fitness.")
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal scaling on all axes
    
    # Plot the BSP tree
    plot_bsp_tree(root, ax)
    
    # Add the point to the plot
    ax.scatter(point[0], point[1], point[2], color='red', s=100, marker='o', label=f"Point {point}")
    
    # Show the plot
    plt.legend()
    plt.show()

def print_bsp_tree(node, depth=0):
    if node is None:
        return
    print(f"{' ' * depth}Node at depth {depth} with bounds: {node.bounds}")
    print_bsp_tree(node.left, depth + 1)
    print_bsp_tree(node.right, depth + 1)

if __name__ == "__main__":
    main()
'''
