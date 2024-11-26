import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BSP:
    def __init__(self, range_input, dimensions, tree=None):
        """
        Initializes the BSP with the given range for all dimensions.
        `range_input` should be a tuple defining the range, like (-200, 200).
        The range will be applied to all dimensions.
        Example: range_input = (-200, 200) for 2D or 3D.
        """
        self.ranges = np.array([range_input] * dimensions)  # Default to 3D if only one range is provided
        self.dim = dimensions
        self.fitness_scores_updated = None
        self.tree = tree

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
        
        self.fitness_scores_updated = []

        for i, point in enumerate(points):
            normal = plane[1]
            point_on_plane = np.dot(point - plane[0], normal)
            if np.isclose(point_on_plane, 0):
                points_on_plane.append(point)
            elif point_on_plane > 0:
                ahead.append(point)
                fitness_ahead.append(fitness_scores[i])
                self.fitness_scores_updated.append(fitness_scores[i])
            else:
                behind.append(point)
                fitness_behind.append(fitness_scores[i])
                self.fitness_scores_updated.append(fitness_scores[i])
        
        return (np.array(ahead), np.array(fitness_ahead)), (np.array(behind), np.array(fitness_behind)), np.array(points_on_plane)

    def build_tree(self, points, fitness_scores):
        def bsp_helper(points, fitness_scores, plane):
            ahead, behind, on_plane = self.bisect(points, plane, fitness_scores)
            node_id = id(plane)
        
            min_fitness = float('inf')
            min_fitness_points = []

            #print(ahead)
            #print(behind)
            #print(on_plane)
            #print(self.fitness_scores_updated)

            for region_points, region_fitness in [(ahead[0], ahead[1]), (behind[0], behind[1])]:
                print("Region Points: ", region_points)
                #print(region_points[1])
                print("Regions Fitnesses: ", region_fitness)
                if len(region_fitness) == 0:
                    continue

                min_fitness_in_region = min(region_fitness)
                if min_fitness_in_region < min_fitness:
                    min_fitness = min_fitness_in_region
                    min_fitness_points = region_points[np.asarray(region_fitness)==min_fitness]
                    print("Min Fitness Points: ", min_fitness_points)

                # Add the node to the graph
                graph.add_node(node_id, plane=plane, min_fitness=min_fitness, min_fitness_points=min_fitness_points)
            
                if len(behind[0]) > 0:
                    node_behind = bsp_helper(behind[0], behind[1], select_dividing_plane(behind[0], self.dim))
                    graph.add_edge(node_id, node_behind, position=-1)
            
                if len(ahead[0]) > 0:
                    node_ahead = bsp_helper(ahead[0], ahead[1], select_dividing_plane(ahead[0], self.dim))
                    graph.add_edge(node_id, node_ahead, position=1)
            
            return node_id
        
        graph = nx.DiGraph()
        if len(points) > 0:
            starting_plane = select_dividing_plane(points, self.dim)
        else:
            starting_plane = (np.zeros(self.dim), np.ones(self.dim))  # Default plane
        
        bsp_helper(points, fitness_scores, starting_plane)
        return nx.relabel.convert_node_labels_to_integers(graph)

    def grow_tree(self, points, fitness_scores):
        def bsp_helper(points, fitness_scores, plane):
            ahead, behind, on_plane = self.bisect(points, plane, fitness_scores)
            node_id = id(plane)
        
            min_fitness = float('inf')
            min_fitness_points = []

            #print(ahead)
            #print(behind)
            #print(on_plane)
            #print(self.fitness_scores_updated)

            for region_points, region_fitness in [(ahead[0], ahead[1]), (behind[0], behind[1])]:
                print("Region Points: ", region_points)
                #print(region_points[1])
                print("Regions Fitnesses: ", region_fitness)
                if len(region_fitness) == 0:
                    continue

                min_fitness_in_region = min(region_fitness)
                if min_fitness_in_region < min_fitness:
                    min_fitness = min_fitness_in_region
                    min_fitness_points = region_points[np.asarray(region_fitness)==min_fitness]
                    print("Min Fitness Points: ", min_fitness_points)

                # Add the node to the graph
                graph.add_node(node_id, plane=plane, min_fitness=min_fitness, min_fitness_points=min_fitness_points)
            
                if len(behind[0]) > 0:
                    node_behind = bsp_helper(behind[0], behind[1], select_dividing_plane(behind[0], self.dim))
                    graph.add_edge(node_id, node_behind, position=-1)
            
                if len(ahead[0]) > 0:
                    node_ahead = bsp_helper(ahead[0], ahead[1], select_dividing_plane(ahead[0], self.dim))
                    graph.add_edge(node_id, node_ahead, position=1)
            
            return node_id
        
        graph = self.tree
        if len(points) > 0:
            starting_plane = select_dividing_plane(points, self.dim)
        else:
            starting_plane = (np.zeros(self.dim), np.ones(self.dim))  # Default plane
        
        bsp_helper(points, fitness_scores, starting_plane)
        return nx.relabel.convert_node_labels_to_integers(graph)

def select_dividing_plane(points, dimensions):
    """
    Selects the dividing plane based on the median of the points.
    The plane is defined by the median point and the normal is aligned
    with the axis of greatest variance in the dataset.
    """
    variances = np.var(points, axis=0)
    dividing_axis = np.argmax(variances)  # Choose the axis with the greatest variance
    #sorted_points = points[points[:, dividing_axis].argsort()]
    print("before sort: ", points)
    sorted_points = points[points[:, dividing_axis].argsort()]
    print("after sort: ", sorted_points)    
    median_point = sorted_points[len(sorted_points) // 2]

    normal = np.zeros(dimensions)
    normal[dividing_axis] = 1  # The normal is along the chosen axis

    return median_point, normal

def find_region_with_lowest_fitness(objective_computation, tree, point, fitness, dimensions):
    """
    Traverse the BSP tree to find the region where the point lies,
    check if it is in the lowest fitness region, and if so, bisect the region.
    """
    current_node = list(tree.nodes)[0]  # Start from the root of the tree
    lowest_fitness = float('inf')
    lowest_fitness_region = None

    while current_node is not None or current_node == {}:
        print('Current Node: ', current_node)
        print('Current Node Attributes: ', tree.nodes[current_node])
        print('Current Neighbors: ', list(tree.neighbors(current_node)))
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
            #new_plane = select_dividing_plane(lowest_fitness_region, dimensions)
            #new_node_id = id(new_plane)
            new_fitness_scores = objective_computation(lowest_fitness_region[0][0], lowest_fitness_region[0][1], lowest_fitness_region[0][2]) 

            bsp = BSP((-200,200), dimensions, tree)
            
            graph = bsp.grow_tree(lowest_fitness_region, new_fitness_scores)

            #ahead, behind, _ = bsp.bisect(lowest_fitness_region, new_plane, new_fitness_scores)
            #graph = tree  # Use existing tree

            # Add the new nodes to the tree
            #graph.add_node(new_node_id, plane=new_plane, min_fitness=min_fitness, min_fitness_points=lowest_fitness_region)
            #graph.add_edge(current_node, new_node_id)

            # Return the updated tree
            return True, lowest_fitness_region, min_fitness, graph

        normal = plane[1]
        point_on_plane = np.dot(point - plane[0], normal)

        # Move to the appropriate subtree
        if point_on_plane > 0:
            neighbors = list(tree.neighbors(current_node))
            #print(neighbors)
            if len(neighbors) == 1:
                current_node = neighbors[0]
            elif len(neighbors) > 1:
                current_node = neighbors[1]# Move to the "ahead" node
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

        if tree.nodes[current_node] == {}:
            break

    #print(f"Updating point {point} with fitness score: {point_fitness}")
    return False, lowest_fitness_region, lowest_fitness, tree



