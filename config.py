import torch

# Physics parameters
VISCOSITY = 0.01 / torch.pi # Burgers equation

# Domain
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Number of train points
N_U = 100 # Initial and boundary conditions
N_F = 10000 # Collocation points

# Neural network architecture
LAYERS = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# Optimization hyperparameters
LEARNING_RATE = 1e-3
ADAM_ITERATIONS = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")