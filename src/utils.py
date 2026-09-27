from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_results(model, config, save_dir=OUTPUT_DIR):
    model.eval()
    
    t = np.linspace(config.T_MIN, config.T_MAX, 200)
    x = np.linspace(config.X_MIN, config.X_MAX, 200)
    T, X = np.meshgrid(t, x)
    
    t_tensor = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(config.DEVICE)
    x_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(config.DEVICE)
    
    with torch.no_grad():
        u_pred = model(t_tensor, x_tensor).cpu().numpy()
        
    U = u_pred.reshape(T.shape)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    contour = plt.pcolormesh(T, X, U, cmap="rainbow", shading="auto")
    
    plt.colorbar(contour, label='$u(t, x)$')
    
    plt.xlabel('Time ($t$)')
    plt.ylabel('Space ($x$)')
    plt.tight_layout()
    
    file_path = save_dir / "burgers_pinn_solution.png"
    plt.savefig(file_path, dpi=300)
    
    plt.close()
    
def plot_loss_history(loss_history, save_dir=OUTPUT_DIR):
    plt.figure(figsize=(10, 6))
    
    plt.plot(loss_history, label='Total Loss')
    plt.yscale('log')
    
    plt.xlabel('Iterations')
    plt.ylabel('MSE (Log Scale)')
    
    plt.grid(True, which="both", ls="--")
    
    plt.legend()
    plt.tight_layout()
    
    file_path = save_dir / "loss_history.png"
    plt.savefig(file_path, dpi=300)
    
    plt.close()
    