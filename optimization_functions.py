import numpy as np

class Surface:
    # Rastrigin Function and Gradient (3D version)
    def rastrigin(self, x, y, z, A=10):
        x = np.clip(x, -500, 500)
        y = np.clip(y, -500, 500)
        z = np.clip(z, -500, 500)
        value = A * 3 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y)) + (z**2 - A * np.cos(2 * np.pi * z))
        return value

    def rastrigin_gradient(self, x, y, z, A=10):
        grad_x = 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)
        grad_y = 2 * y + 2 * A * np.pi * np.sin(2 * np.pi * y)
        grad_z = 2 * z + 2 * A * np.pi * np.sin(2 * np.pi * z)
        return np.array([grad_x, grad_y, grad_z])

    # Ackley Function and Gradient (3D version)
    def ackley(self, x, y, z):
        x = np.clip(x, -500, 500)
        y = np.clip(y, -500, 500)
        z = np.clip(z, -500, 500)
        value = -20 * np.exp(-0.2 * np.sqrt(0.3333 * (x**2 + y**2 + z**2))) - np.exp(0.3333 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z))) + np.e + 20
        return value

    def ackley_gradient(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)  # The Euclidean norm (distance from origin)
    
        # Gradient with respect to x
        grad_x = (x / r) * np.exp(-0.2 * r) + 2 * np.pi * np.sin(2 * np.pi * x) * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)) / 3)
    
        # Gradient with respect to y
        grad_y = (y / r) * np.exp(-0.2 * r) + 2 * np.pi * np.sin(2 * np.pi * y) * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)) / 3)
    
        # Gradient with respect to z
        grad_z = (z / r) * np.exp(-0.2 * r) + 2 * np.pi * np.sin(2 * np.pi * z) * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)) / 3)
    
        return np.array([grad_x, grad_y, grad_z])

    # Griewank Function and Gradient (3D version)
    def griewank(self, x, y, z):
        value = 1 + (x**2 / 4000) + (y**2 / 4000) + (z**2 / 4000) - np.cos(x) * np.cos(y / np.sqrt(2)) * np.cos(z / np.sqrt(3))
        return value

    def griewank_gradient(self, x, y, z):
        grad_x = (x / 2000) + np.sin(x) * np.cos(y / np.sqrt(2)) * np.cos(z / np.sqrt(3))
        grad_y = (y / 2000) - np.sin(y / np.sqrt(2)) * np.cos(x) * np.cos(z / np.sqrt(3))
        grad_z = (z / 2000) - np.sin(z / np.sqrt(3)) * np.cos(x) * np.cos(y / np.sqrt(2))
        return np.array([grad_x, grad_y, grad_z])

    # Sphere Function and Gradient (3D version)
    def sphere(self, x, y, z):
        value = x**2 + y**2 + z**2
        return value

    def sphere_gradient(self, x, y, z):
        grad_x = 2 * x
        grad_y = 2 * y
        grad_z = 2 * z
        return np.array([grad_x, grad_y, grad_z])

    # Rosenbrock (Banana) Function and Gradient (3D version)
    def rosenbrock(self, x, y, z, a=1, b=100):
        x = np.clip(x, -500, 500)
        y = np.clip(y, -500, 500)
        z = np.clip(z, -500, 500)
        value = (a - x)**2 + b * (y - x**2)**2 + (z - y**2)**2
        return value

    def rosenbrock_gradient(self, x, y, z, a=1, b=100):
        x = np.clip(x, -500, 500)
        y = np.clip(y, -500, 500)
        z = np.clip(z, -500, 500)

        grad_x = -2 * (a - x) - 4 * b * x * (y - x**2)
        grad_y = 2 * b * (y - x**2) - 2 * (z - y**2)
        grad_z = 2 * (z - y**2)
        return np.array([grad_x, grad_y, grad_z])

    # Michalewicz Function and Gradient (3D version) sba not working here
    def michalewicz(self, x, y, z, m=10):
        value = - (np.sin(x) * np.sin(x**2 / np.pi))**(2 * m) - (np.sin(y) * np.sin(y**2 / np.pi))**(2 * m) - (np.sin(z) * np.sin(z**2 / np.pi))**(2 * m)
        return value

    def michalewicz_gradient(self, x, y, z, m=10):
        grad_x = -2 * m * (np.sin(x) * np.sin(x**2 / np.pi))**(2 * m - 1) * np.cos(x) * np.cos(x**2 / np.pi) * 2 * x / np.pi
        grad_y = -2 * m * (np.sin(y) * np.sin(y**2 / np.pi))**(2 * m - 1) * np.cos(y) * np.cos(y**2 / np.pi) * 2 * y / np.pi
        grad_z = -2 * m * (np.sin(z) * np.sin(z**2 / np.pi))**(2 * m - 1) * np.cos(z) * np.cos(z**2 / np.pi) * 2 * z / np.pi
        return np.array([grad_x, grad_y, grad_z])

    # Booth Function and Gradient (3D version)
    def booth(self, x, y, z):
        x = np.clip(x, -500, 500)
        y = np.clip(y, -500, 500)
        z = np.clip(z, -500, 500)

        value = (x + 2 * y - 7)**2 + (2 * x + y - 5)**2 + (z - 1)**2
        return value

    def booth_gradient(self, x, y, z):
        grad_x = 2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5)
        grad_y = 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)
        grad_z = 2 * (z - 1)
        return np.array([grad_x, grad_y, grad_z])
