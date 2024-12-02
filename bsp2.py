import random

# Class for BSP Tree Node
class BSPNode:
    def __init__(self, position, fitness, left=None, right=None, split_dim=None, split_value=None, parent=None):
        self.position = position  # The solution (e.g., [x, f(x)])
        self.fitness = fitness    # Fitness of the solution (f(x))
        self.left = left          # Left child node
        self.right = right        # Right child node
        self.split_dim = split_dim  # Dimension on which the partition occurs (e.g., 0 for x-axis)
        self.split_value = split_value  # The value that splits the region
        self.parent = parent 

    def update_fitness(self, new_fitness):
        """Update the fitness of this node and propagate the change to parent nodes."""
        self.fitness = new_fitness
        
        # If there's a parent, propagate the change upwards
        if self.parent:
            self.parent.update_fitness(self.fitness)

# Class for BSP Tree
class BSPTree:
    def __init__(self, surface_function):
        self.root = None  # Initialize the BSP tree with no root
        self.surface_function = surface_function

    def insert(self, solution):
        # Insert a solution into the BSP tree
        if self.root is None:
            # If the tree is empty, create a root node
            self.root = BSPNode(solution[0], solution[1])
        else:
            # Start inserting from the root node
            self._insert_recursive(self.root, solution)

    def _insert_recursive(self, current_node, solution):
        if current_node.left and current_node.right:
            # Find the dimension (axis) with the largest difference
            dimension = self._find_max_diff_dimension(current_node.left.position, current_node.right.position)

            # Update the split dimension and value if necessary
            if current_node.split_dim != dimension or current_node.split_value != current_node.position[dimension]:
                current_node.split_dim = dimension
                current_node.split_value = current_node.position[dimension]

            # Compare the solution with left and right children
            if abs(current_node.left.position[dimension] - solution[0][dimension]) <= abs(current_node.right.position[dimension] - solution[0][dimension]):
                self._insert_recursive(current_node.left, solution)
            else:
                self._insert_recursive(current_node.right, solution)
        else:
            # If the node doesn't have children, insert it
            if not current_node.left:
                current_node.left = BSPNode(solution[0], solution[1])
                current_node.left.split_dim = self._find_max_diff_dimension(current_node.position, solution[0])
                current_node.left.split_value = solution[0][current_node.left.split_dim]
            elif not current_node.right:
                current_node.right = BSPNode(solution[0], solution[1])
                current_node.right.split_dim = self._find_max_diff_dimension(current_node.position, solution[0])
                current_node.right.split_value = solution[0][current_node.right.split_dim]



    def _find_max_diff_dimension(self, left_position, right_position):
        # Find the dimension with the maximum absolute difference between left and right positions
        return max(range(len(left_position)), key=lambda i: abs(left_position[i] - right_position[i]))

    def traverse(self, node):
        # In-order traversal of the BSP tree (for checking/visualization)
        if node:
            self.traverse(node.left)
            print(f"Position: {node.position}, Fitness: {node.fitness}")
            self.traverse(node.right)

    def extract_region(self, point):
        """
        Extract the region that a given point belongs to.
        This function traverses the BSP tree to find the leaf node where the point belongs.
        
        :param point: The 3D point [x, y, z]
        :return: The region (leaf node) where the point belongs.
        """
        return self._extract_recursive(self.root, point)

    def _extract_recursive(self, current_node, point):
        """
        Recursively traverse the BSP tree and find the region (leaf node) for the given point.
        :param current_node: The current node being checked
        :param point: The point to find the region for
        :return: The region node (leaf node) where the point belongs
        """
        if current_node is None:
            return None
        
        # If this node is a leaf node, return it
        if current_node.left is None and current_node.right is None:
            return current_node
        
        # Check the current node's split dimension and value
        split_dim = current_node.split_dim
        split_value = current_node.split_value
        
        # Traverse left or right based on the point's position
        #print("en extract recursive: ", current_node.left, " ", current_node.right, " ", split_dim, " ", split_value)
        if point[split_dim] < split_value:
            return self._extract_recursive(current_node.left, point)
        else:
            return self._extract_recursive(current_node.right, point)

    def compare_with_neighbors(self, point):
        """
        Compare the fitness of the region containing the point with its neighbors.
        
        :param point: The 3D point [x, y, z]
        :return: True if the region has the lowest fitness compared to its neighbors, otherwise False.
        """
        region = self.extract_region(point)
        if not region:
            return False, None, None
        
        # Get the fitness of the current region
        current_fitness = region.fitness

        # Check left and right neighbors
        left_fitness = region.left.fitness if region.left else float('inf')
        right_fitness = region.right.fitness if region.right else float('inf')

        is_best_fitness = current_fitness < left_fitness and current_fitness < right_fitness

        return is_best_fitness, current_fitness, region

    def fitness_approximation(self, solutions):
        for solution in solutions:
            in_best_region, region_fitness, region = self.compare_with_neighbors(solution[0])
            if in_best_region:
                new_fitness = self.surface_function(solution[0][0], solution[0][1], solution[0][2])
                if new_fitness < region_fitness:
                    region.fitness = new_fitness
                    self.bisect_region(region)

    def bisect_region(self, point):
        region = self.extract_region(point)
        if not region:
            return None

        # Assuming the best region has the lowest fitness, let's bisect it.
        # Find the split dimension (axis) to split the region
        split_dim = region.split_dim
        split_value = region.position[split_dim]

        # Create two new child nodes by splitting the region in the middle
        mid_point = (region.position[split_dim] + split_value) / 2

        left_node = BSPNode(region.position, region.fitness, split_dim=split_dim, split_value=mid_point, parent=region)
        right_node = BSPNode(region.position, region.fitness, split_dim=split_dim, split_value=mid_point, parent=region)

        # Attach the new child nodes to the current region
        region.left = left_node
        region.right = right_node

        # Update the fitness of the new regions
        left_node.update_fitness(region.fitness)
        right_node.update_fitness(region.fitness)


    def search_and_return_fitness(self, point):
        """
        Search the BSP tree for the region that contains the given point
        and return the fitness value of that region.
        
        :param point: The 3D point [x, y, z]
        :return: The fitness of the region the point belongs to
        """
        region = self.extract_region(point)  # Find the region the point belongs to
        if region:
            return region.fitness  # Return the fitness of the region
        else:
            return None  

    def initialize_bsp_tree(self, range_start, range_end, num_partitions):
        """
        Initializes the BSP tree by dividing a 3D space into user-defined sub-ranges,
        picking random points from each sub-range, evaluating them with the surface function,
        and creating a node for each evaluated point.
        
        :param range_start: The start of the range (list of 3 values [x_min, y_min, z_min])
        :param range_end: The end of the range (list of 3 values [x_max, y_max, z_max])
        :param num_partitions: The number of partitions to divide the range into
        :param surface_function: The function used to evaluate the fitness of each point
        """
        # Divide each dimension into the specified number of partitions
        partitioned_ranges = []
        #print("Ranges: ", range_start, " ", range_end, " ", num_partitions)
        #print( zip(range(range_start), range(range_end)))
        for dim_start, dim_end in zip(range(range_start, range_end), range(range_start + 1, range_end + 1)):
            #print("initializing bsp: ", dim_start, " ", dim_end)
            partitioned_ranges.append([dim_start + (dim_end - dim_start) * i / num_partitions for i in range(num_partitions + 1)])

        #print("Len of pr: ", len(partitioned_ranges))
        # Iterate over the partitioned ranges and pick random points
        for x in range(num_partitions):
            for y in range(num_partitions):
                for z in range(num_partitions):
                    # Pick a random point in each sub-region
                    rand_x = random.uniform(partitioned_ranges[0][x], partitioned_ranges[0][x+1])
                    rand_y = random.uniform(partitioned_ranges[1][y], partitioned_ranges[1][y+1])
                    rand_z = random.uniform(partitioned_ranges[2][z], partitioned_ranges[2][z+1])
                    point = [rand_x, rand_y, rand_z]
                    
                    # Evaluate the fitness of the point using the surface function
                    fitness = self.surface_function(point[0], point[1], point[2])
                    
                    # Insert the point into the BSP tree
                    self.insert([point, fitness])

'''
# Example of how to initialize and insert solutions into the BSP tree
def main():
    bsp_tree = BSPTree()

    # Define the 3D space range
    range_start = [-5, -5, -5]
    range_end = [5, 5, 5]

    # Define the number of partitions per dimension
    num_partitions = 5

    # Initialize the BSP tree by dividing the 3D space and evaluating fitness
    bsp_tree.initialize_bsp(range_start, range_end, num_partitions, surface_function)

    # Traverse the BSP tree to check its contents
    print("BSP Tree Traversal (In-order):")
    bsp_tree.traverse(bsp_tree.root)

if __name__ == "__main__":
    main()
'''
