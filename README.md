# Physics-Informed Neural Networks from Scratch

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.5%2B%20%7C%20CUDA%2012.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A implementation from scratch of **Physics-Informed Neural Networks (PINNs)** to solve the non-linear 1D viscous **Burgers' Equation**, based on the paper by *Raissi, Perdikaris, and Karniadakis (2019)*.

---

## Problem Formulation

The 1D viscous Burgers' equation is a fundamental partial differential equation occurring in various areas of applied mathematics, including fluid mechanics, acoustics, and traffic flow.

### Mathematical Definition

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [-1, 1], \quad t \in [0, 1]$$

where:
* $\nu = \frac{0.01}{\pi}$ is the kinematic viscosity parameter.
* $u(t, x)$ represents the fluid velocity field.

### Boundary and Initial Conditions

* **Initial Condition:**
  $$u(0, x) = -\sin(\pi x), \quad x \in [-1, 1]$$

* **Dirichlet Boundary Conditions:**
  $$u(t, -1) = u(t, 1) = 0, \quad t \in [0, 1]$$

---

## Physics-Informed Loss Function

The neural network $\hat{u}(t, x; \theta)$ approximates the latent solution $u(t, x)$. The automatic differentiation engine of PyTorch is used to compute the PDE residual operator $f(t, x)$. This is defined as follows:

$$f(t, x) := \frac{\partial \hat{u}}{\partial t} + \hat{u} \frac{\partial \hat{u}}{\partial x} - \nu \frac{\partial^2 \hat{u}}{\partial x^2}$$

The overall loss function $\mathcal{L}(\theta)$ is minimized by optimizing the network parameters $\theta = \{W, b\}$ through a combined Mean Squared Error (MSE):

$$\mathcal{L}(\theta) = \text{MSE}_u + \text{MSE}_f$$

$$\text{MSE}_u = \frac{1}{N_u} \sum_{i=1}^{N_u} \left| \hat{u}(t_u^i, x_u^i) - u^i \right|^2$$

$$\text{MSE}_f = \frac{1}{N_f} \sum_{j=1}^{N_f} \left| f(t_f^j, x_f^j) \right|^2$$

Where:
* $\{t_u^i, x_u^i, u^i\}_{i=1}^{N_u}$ are sampled boundary and initial points.
* $\{t_f^j, x_f^j\}_{j=1}^{N_f}$ are collocation points randomly distributed within the space-time domain $(t, x) \in [0, 1] \times [-1, 1]$.

---

## Repository Structure

```text
pinn-from-scratch/
├── results/                        # Generated plots and output figures
│   ├── burgers_pinn_solution.png   # Comparison of predicted vs. exact PINN solution
│   └── loss_history.png            # Loss function convergence history
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # Domain sampling
│   ├── model.py                    # Fully-connected MLP neural network architecture
│   ├── physics.py                  # Automatic differentiation & PDE residual f(t, x)
│   └── utils.py                    # Plotting routines and visualization utilities
├── .gitignore                      # Excluded files and directories (__pycache__, .venv)
├── LICENSE                         # Project license
├── README.md                       # Main project documentation
├── config.py                       # Hyperparameters, physical constants, and device setup
├── train.py                        # Main training script
└── requirements.txt                # Project dependencies
```

---

## Installation & Setup

### Prerequisites
* **Python**: `3.12` (64-bit)
* **GPU**: NVIDIA GPU with CUDA drivers installed (optional, CPU fallback supported)

### 1. Clone the Repository
```bash
git clone https://github.com/jakomycat/pinn-from-scratch.git
cd pinn-from-scratch
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on CUDA Support**: The `requirements.txt` file uses `--extra-index-url https://download.pytorch.org/whl/cu121` to automatically pull PyTorch compiled with **CUDA 12.1** acceleration.

---

## Running the Model

To start training the PINN model, simply run:

```bash
python train.py
```

### Execution Pipeline
1. Samples $N_u = 100$ boundary/initial points and $N_f = 10,000$ interior collocation points.
2. Initializes an 8-layer deep neural network with $20$ neurons per layer and $\tanh$ activation functions.
3. Optimizes parameters using the **Adam** optimizer for $2,000$ epochs.
4. Exports evaluation plots into the results directory.

---

## Results and Output

After training, two visualization artifacts will be automatically generated in the results directory:

1. **`burgers_pinn_solution.png`**: Heatmap displaying the spatio-temporal dynamics of the approximated solution $u(t, x)$.
2. **`loss_history.png`**: Convergence curve plotting logarithmic MSE loss over training epochs.

---

## References

* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. **Journal of Computational Physics**, 378, 686-707.

---

## License

This project is open-source and available under the [MIT License](LICENSE).