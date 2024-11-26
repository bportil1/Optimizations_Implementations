import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BSP:
    def __init__(self, range_input, dimensions):
        """
        Initializes the BSP with the given range for all dimensions.
        `range_input` should be a tuple defining the range, like (-200, 200).
        The range will be applied to all dimensions.
        Example: range_input = (-200, 200) for 2D or 3D.
        """
        self.ranges = np.array([range_input] * dimensions)  # Default to 3D if only one range is provided
        self.dim = self.ranges.shape[0]

    def generate_points(self, num_points=100):
        """
        Generate random points within the given ranges.
        `num_points` defines the number of points to generate.
        """
        points = np.random.uniform(self.ranges[:, 0], self.ranges[:, 1], size=(num_points, self.dim))
        return points

    def generate_fitness_scores(self, num_points=100):
        """
        Generate random fitness scores for the given points.
        `num_points` defines the number of points.
        Fitness scores are random between 0 and 1.
        """
        return np.random.rand(num_points)

    def bisect(self, points, plane, fitness_scores):
        """
        Bisects a set of points using a hyperplane defined by `plane`.
        The plane is assumed to be an array that defines the dividing hyperplane.
        Fitness scores are also passed to update the nodes.
        """
        points_on_plane = []
        ahead = []
        behind = []
        fitness_ahead = []
        fitness_behind = []
        
        for i, point in enumerate(points):
            normal = plane[1]
            point_on_plane = np.dot(point - plane[0], normal)
            if np.isclose(point_on_plane, 0):
                points_on_plane.append(point)
            elif point_on_plane > 0:
                ahead.append(point)
                fitness_ahead.append(fitness_scores[i])
            else:
                behind.append(point)
                fitness_behind.append(fitness_scores[i])
        
        return (np.array(ahead), np.array(fitness_ahead)), (np.array(behind), np.array(fitness_behind)), np.array(points_on_plane)

    def select_dividing_plane(self, points):
        """
        Selects the dividing plane based on the median of the points.
        The plane is defined by the median point and the normal is aligned
        with the axis of greatest variance in the dataset.
        """
        variances = np.var(points, axis=0)
        dividing_axis = np.argmax(variances)  # Choose the axis with the greatest variance
        sorted_points = points[points[:, dividing_axis].argsort()]
        median_point = sorted_points[len(sorted_points) // 2]
        
        normal = np.zeros(self.dim)
        normal[dividing_axis] = 1  # The normal is along the chosen axis
        
        return median_point, normal

    def build_tree(self, points, fitness_scores):
        def bsp_helper(points, fitness_scores, plane):
            ahead, behind, on_plane = self.bisect(points, plane, fitness_scores)
            
            # Store node information: minimum fitness in region and points that contributed to it
            min_fitness = float('inf')
            min_fitness_points = []

            #points = np.array(points)
            #fitness_scores = np.array(fitness_scores)

            # For ahead and behind regions, find the minimum fitness points
            for region_points, region_fitness in [(ahead, fitness_scores[:len(ahead)]), (behind, fitness_scores[len(ahead):])]:
                print(region_fitness)
                region_points = np.array(region_points)
                region_fitness = np.array(region_fitness)
                min_fitness_in_region = min(region_fitness)
                #min_fitness_region = np.argmin(region_fitness)
                if min_fitness_in_region < min_fitness:
                    min_fitness = min_fitness_in_region
                    min_fitness_points = region_points[region_fitness==min_fitness]
            
            # Add the node to the graph
            node_id = id(plane)
            graph.add_node(node_id, plane=plane, min_fitness=min_fitness, min_fitness_points=min_fitness_points)
            
            if len(behind[0]) > 0:
                node_behind = bsp_helper(behind[0], behind[1], self.select_dividing_plane(behind[0]))
                graph.add_edge(node_id, node_behind, position=-1)
            
            if len(ahead[0]) > 0:
                node_ahead = bsp_helper(ahead[0], ahead[1], self.select_dividing_plane(ahead[0]))
                graph.add_edge(node_id, node_ahead, position=1)
            
            return node_id
        
        graph = nx.DiGraph()
        if len(points) > 0:
            starting_plane = self.select_dividing_plane(points)
        else:
            starting_plane = (np.zeros(self.dim), np.ones(self.dim))  # Default plane
        
        bsp_helper(points, fitness_scores, starting_plane)
        return nx.relabel.convert_node_labels_to_integers(graph)

    def traverse_dfs(self, tree, node_id=None, visited=None):
        if visited is None:
            visited = set()
        
        if node_id is None:
            node_id = list(tree.nodes)[0]  # Start with the first node
        
        visited.add(node_id)
        print(f"Visiting node {node_id}")
        
        neighbors = list(tree.neighbors(node_id))
        for neighbor in neighbors:
            if neighbor not in visited:
                self.traverse_dfs(tree, neighbor, visited)

    def traverse_bfs(self, tree, start_node=None):
        if start_node is None:
            start_node = list(tree.nodes)[0]  # Start from the first node
        
        visited = set()
        queue = [start_node]
        
        while queue:
            node_id = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                print(f"Visiting node {node_id}")
                queue.extend(list(tree.neighbors(node_id)))

    def points_caller(self, num_points=100):
        # Generate points and their corresponding fitness scores
        points = self.generate_points(num_points)
        fitness_scores = self.generate_fitness_scores(num_points)
        
        # Build the BSP tree
        tree = self.build_tree(points, fitness_scores)
        
        # Find the region with the lowest fitness score for each point
        for point in points:
            lowest_fitness_region, updated_tree = self.find_region_with_lowest_fitness(tree, point)
            print(f"Lowest fitness region for point {point}: {lowest_fitness_region}")
            # Optionally visualize or process the updated tree here
            return updated_tree

def find_region_with_lowest_fitness(tree, point, fitness):
	"""
	Traverse the BSP tree to find the region where the point lies,
	check if it is in the lowest fitness region, and if so, bisect the region.
	"""
	current_node = list(tree.nodes)[0]  # Start from the root of the tree
	lowest_fitness = float('inf')
	lowest_fitness_region = None
	point_fitness = fitness  # New random fitness score for demonstration

	while current_node is not None:
		# Get the plane defining the current node
		plane = tree.nodes[current_node]["plane"]
		min_fitness = tree.nodes[current_node]["min_fitness"]
		min_fitness_points = tree.nodes[current_node]["min_fitness_points"]
    
		# Update the lowest fitness if needed
		if min_fitness < lowest_fitness:
			lowest_fitness = min_fitness
			lowest_fitness_region = min_fitness_points
    
		# Check if the point is in the lowest fitness region
		if point in lowest_fitness_region:
			print(f"Point {point} is in the lowest fitness region, bisecting the region.")
			# Bisect the region
			new_plane = self.select_dividing_plane(lowest_fitness_region)
			new_node_id = id(new_plane)
			new_fitness_scores = np.random.rand(len(lowest_fitness_region))  # Assign random fitness scores
			ahead, behind, _ = self.bisect(lowest_fitness_region, new_plane, new_fitness_scores)
			graph = tree  # Use existing tree

			# Add the new nodes to the tree
			graph.add_node(new_node_id, plane=new_plane, min_fitness=min_fitness,  						min_fitness_points=lowest_fitness_region)
			graph.add_edge(current_node, new_node_id)

			# Return the updated tree
			return True, lowest_fitness_region, min_fitness, graph
    
		# Check where the point lies: ahead or behind the plane
		normal = plane[1]
		point_on_plane = np.dot(point - plane[0], normal)

		# Move to the appropriate subtree
		if point_on_plane > 0:
			neighbors = list(tree.neighbors(current_node))
			if neighbors:
				current_node = neighbors[1]  # Move to the "ahead" node
			else:
				break  # No more nodes, end the search
		elif point_on_plane < 0:
			neighbors = list(tree.neighbors(current_node))
			if neighbors:
				current_node = neighbors[0]  # Move to the "behind" node
			else:
				break  # No more nodes, end the search
		else:
			break  # Point is on the plane, we stop here

	print(f"Updating point {point} with fitness score: {point_fitness}")
	return False, lowest_fitness_region, lowest_fitness, tree

