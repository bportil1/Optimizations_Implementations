import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import isclose
import math

from optimization_functions import *
from optimizers_tsp import *

import plotly.express as px
import plotly.graph_objects as go

class tests():
    def __init__(self):
        self.surface_fcn = Surface()

    def adam_test(self):
        adam = AdamOptimizer(self.surface_fcn.booth, self.surface_fcn.booth_gradient, alpha=10000000)
        adam.optimize()

    def sa_test(self):
        sa = SimulatedAnnealingOptimizer(self.surface_fcn.booth )
        minima, lowest_val, path = sa.optimize()
        
        print("Best Postition: ", minima)
        print("Minima: ", lowest_val)

        self.annealing_plot(minima, lowest_val, path)        

    def pso_test(self):
        pso = ParticleSwarmOptimizer(self.surface_fcn.ackley, self.surface_fcn.ackley_gradient, 100, 3, 1000) 
        minima, lowest_val, paths, values = pso.optimize()
        self.swarm_plot(minima, lowest_val, paths, values)

    def sba_test(self):
        sba = SwarmBasedAnnealingOptimizer(self.surface_fcn.ackley, self.surface_fcn.ackley_gradient, 100, 3, 200)
        minima, lowest_val, paths, values = sba.optimize()
        print("Best Position: ", minima)
        print("Minima: ", lowest_val)
        #print(values)
        self.swarm_plot(minima, lowest_val, paths, values)

    def hdffsa_test(self):
        hdffsa = HdFireflySimulatedAnnealingOptimizer(self.surface_fcn.ackley, 3, 100)
        hdff_min_pt, hdff_min_fitness, hdff_positions_history, hdff_fitness_history, hdff_minima_positions, hdff_minima_fitness, sa_min_pt, sa_min_fitness, sa_path = hdffsa.optimize()

        self.plot_firefly_movements_3d_plotly(hdff_positions_history)
        self.plot_error_history(hdff_fitness_history)

        print("Best Position: ", sa_min_pt)
        print("Minima :", sa_min_fitness)

        self.annealing_plot(sa_min_pt, sa_min_fitness, sa_path)

    def annealing_plot(self, minima, lowest_val, path):
        x = np.linspace(-5, 5, 1000)
        y = np.linspace(-5, 5, 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.surface_fcn.ackley(X, Y, 0)  # For simplicity, use z=0 for surface visualization

        # Create a 3D surface plot using Plotly
        fig = go.Figure()

        # Add surface plot
        fig.add_trace(go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.6,
            showscale=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[point[0] for point in path],
            y=[point[1] for point in path],
            z=[point[2] for point in path],
            mode='markers+lines',
            marker=dict(size=1, color='red', opacity=0.7),
            line=dict(width=1, color='red')
        ))

        # Highlight the minimum found
        fig.add_trace(go.Scatter3d(
            x=[minima[0]], y=[minima[1]], z=[minima[2]],
            mode='markers',
            marker=dict(
                size=10, color='blue', symbol='diamond', line=dict(width=2, color='black')
            ),
            name='Global Minimum'
        ))

        fig.update_layout(
            title="Hd-FFSA Optimization",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        fig.write_html("sa_optimization_booth.html")
        # Show the plot
        fig.show()

    def swarm_plot(self, minima, lowest_val, paths, values, ):
        x = np.linspace(-100, 100, 1000)
        y = np.linspace(-100, 100, 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.surface_fcn.ackley(X, Y, 0)  # For simplicity, use z=0 for surface visualization

        # Create a 3D surface plot using Plotly
        fig = go.Figure()

        # Add surface plot
        fig.add_trace(go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.6,
            showscale=False
        ))
       
        '''
        color_scale = 'Jet'  # You can change this color scale
        for i, path in enumerate(paths):
            path = np.array(path)
            num_steps = path.shape[0]

            # Assign colors to each path step based on their iteration (steps)
            colors = np.linspace(0, 1, num_steps)
            fig.add_trace(go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    #color=colors,  # Color the path steps from start to end
                    #colorscale=color_scale,  # Define color scale
                    cmin=0, cmax=1,
                    line=dict(width=2)
                ),
                line=dict(width=4, colorscale=color_scale),  # Thicker lines
                name=f'Agent {i} Path'
            ))
            '''
        # Highlight the minimum found
        fig.add_trace(go.Scatter3d(
            x=[minima[0]], y=[minima[1]], z=[minima[2]],
            mode='markers',
            marker=dict(
                size=10, color='blue', symbol='diamond', line=dict(width=2, color='black')
            ),
            name='Global Minimum'
        ))

        fig.update_layout(
            title="Swarm-based Simulated Annealing Optimization",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        # Show the plot
        fig.write_html("sba_optimization_ackley.html")
        fig.show()

        plt.figure(figsize=(8, 6))
        for value in values:
            plt.plot(value, label="Global Best Fitness")
        plt.title("Global Best Fitness Over Iterations")
        plt.xlim(0, len(values[0]))
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Value")
        plt.grid(True)
        plt.legend()
        plt.show()
           
    def plot_firefly_movements_3d_plotly(self, firefly_positions_history):
        x = np.linspace(-50, 50, 1000)
        y = np.linspace(-600, 25, 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.surface_fcn.booth(X, Y, 0)  # For simplicity, use z=0 for surface visualization

        fig = go.Figure()


        fig.add_trace(go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.6,
            showscale=False
        ))


        iterations = firefly_positions_history.shape[0]
        num_fireflies = firefly_positions_history.shape[1]


        # Plot the trajectory for each firefly in 3D
        for i in range(num_fireflies):
            # Extract the positions for this firefly at each iteration
            firefly_trajectory = firefly_positions_history[:, i, :]

            # Create a 3D line plot for the firefly's trajectory
            fig.add_trace(go.Scatter3d(
                x=firefly_trajectory[:, 0],  # X coordinates
                y=firefly_trajectory[:, 1],  # Y coordinates
                z=firefly_trajectory[:, 2],  # Z coordinates
                mode='lines+markers',        # Display both lines and markers
                name=f'Firefly {i+1}',        # Name for the firefly
                marker=dict(size=6)          # Marker size for visibility
            ))

        # Update layout for the plot
        fig.update_layout(
            title='Fireflies Movement Over Time (3D)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            showlegend=True
        )   
        fig.write_html("hdffsa_optimization_ackley.html")
        # Show the plot
        fig.show()

    def plot_error_history(self, fitness_history):
        # Create a line plot for the fitness history
        fig = go.Figure()

        # Plot fitness (error) over HDFA iterations
        fig.add_trace(go.Scatter(
            x=list(range(len(fitness_history))),  # x-axis: HDFA iterations
            y=fitness_history,                    # y-axis: fitness (error)
            mode='lines+markers',                 # Show both line and markers
            name='Fitness/Error',                 # Name for the plot
            marker=dict(size=6)                   # Marker size for visibility
        ))

        # Update layout for the plot
        fig.update_layout(
            title='Fitness/Error Over HDFA Iterations',
            xaxis_title='Iteration',
            yaxis_title='Fitness/Error',
            showlegend=True
        )

        # Show the plot
        fig.show()


if __name__ == "__main__":
    test = tests()
    #test.adam_test()
    #test.sa_test()
    #test.pso_test()
    #test.sba_test()
    test.hdffsa_test()
