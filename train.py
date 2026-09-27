import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.model import PINNNet
from src.physics import compute_pde_residual
from src.dataset import generate_domain_samples
from src.utils import plot_results, plot_loss_history

def train():
    torch.manual_seed(1)
    
    print(f"Train initialization in: {config.DEVICE}")
    
    # Generate data
    (t_u, x_u, u_u), (t_f, x_f) = generate_domain_samples(
        config.N_U, config.N_F, config.X_MIN, config.X_MAX,
        config.T_MIN, config.T_MAX, config.DEVICE
    )
    
    # Model
    model = PINNNet(config.LAYERS).to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    mse_loss = nn.MSELoss()
    
    loss_history = []
    
    # Train loop
    model.train()
    
    for _ in tqdm(range(1, config.ADAM_ITERATIONS + 1), desc="Train progress", unit=" Epoch"):
        optimizer.zero_grad()
        
        u_pred = model(t_u, x_u)
        loss_u = mse_loss(u_pred, u_u)
        
        f_pred = compute_pde_residual(model, t_f, x_f, config.VISCOSITY)
        loss_f = mse_loss(f_pred, torch.zeros_like(f_pred))
        
        loss = loss_u + loss_f
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
    print("\nGenerating Results Charts")
    plot_results(model, config)
    plot_loss_history(loss_history)
    print("\nProcess completed successfully")
    
if __name__ == "__main__":
    train()