import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import isclose
import math

from optimization_functions import *
from optimizers import *

import plotly.express as px
import plotly.graph_objects as go

class tests():
    def __init__(self):
        self.surface_fcn = Surface()

    def sa_test(self):
        sa = SimulatedAnnealingOptimizer(self.surface_fcn.rastrigin )
        minima, lowest_val, path = sa.optimize()
        
        print("Best Postition: ", minima)
        print("Minima: ", lowest_val)

        self.annealing_plot(minima, lowest_val, path)
        

    def pso_test(self):
        pso = ParticleSwarmOptimizer(self.surface_fcn.ackley, self.surface_fcn.ackley_gradient, 30, 3, 10) 
        minima, lowest_val, paths, values = pso.optimize()
        self.swarm_plot(minima, lowest_val, paths, values)

    def sba_test(self):
        sba = SwarmBasedAnnealingOptimizer(self.surface_fcn.booth, self.surface_fcn.booth_gradient, 100, 3, 200)
        minima, lowest_val, paths, values = sba.optimize()
        print("Best Position: ", minima)
        print("Minima: ", lowest_val)
        #print(values)
        self.swarm_plot(minima, lowest_val, paths, values)

    def hdffsa_test(self):
        hdffsa = HdFireflySimulatedAnnealingOptimizer(self.surface_fcn.griewank, 3, 30, 10)
        minima, lowest_val, path = hdffsa.optimize()

        print("Best Position: ", minima)
        print("Minima :", lowest_val)

        self.annealing_plot(minima, lowest_val, path)

    def annealing_plot(self, minima, lowest_val, path):
        x = np.linspace(-50, 50, 1000)
        y = np.linspace(-600, 50, 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.surface_fcn.griewank(X, Y, 0)  # For simplicity, use z=0 for surface visualization

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

        # Highlight the minimum found
        fig.add_trace(go.Scatter3d(
            x=[minima[0]], y=[minima[1]], z=[minima[2]],
            mode='markers',
            marker=dict(
                size=10, color='red', symbol='diamond', line=dict(width=2, color='black')
            ),
            name='Global Minimum'
        ))

        fig.update_layout(
            title="3D Simulated Annealing",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        # Show the plot
        fig.show()

    def swarm_plot(self, minima, lowest_val, paths, values, ):
        x = np.linspace(-100, 100, 1000)
        y = np.linspace(-100, 100, 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.surface_fcn.booth(X, Y, 0)  # For simplicity, use z=0 for surface visualization

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
                    color=colors,  # Color the path steps from start to end
                    colorscale=color_scale,  # Define color scale
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
                size=10, color='red', symbol='diamond', line=dict(width=2, color='black')
            ),
            name='Global Minimum'
        ))

        fig.update_layout(
            title="3D Particle Swarm Optimization Paths",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        # Show the plot
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
            
if __name__ == "__main__":
    test = tests()
    #test.sa_test()
    #test.pso_test()
    test.sba_test()
    #test.hdffsa_test()
