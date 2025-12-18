# %% [markdown]
# L30: Neural Network–based Solvers for Convection–Diffusion Systems
#
# This notebook demonstrates how Physics-Informed Neural Networks (PINNs)
# can be used to solve a 1D steady convection–diffusion equation.
#
# PDE:
#   -ε u''(x) + b u'(x) = f(x),  x in (0,1)
# Boundary conditions:
#   u(0) = 0,  u(1) = 0
#
# We use a neural network to approximate u(x) and enforce the PDE
# and boundary conditions through the loss function.
#
# This notebook is designed for hands-on use in Google Colab.

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# %% [markdown]
# ## Problem setup
# We consider a convection-dominated problem to illustrate
# boundary-layer behavior.

# %%
# Parameters
epsilon = 0.01   # diffusion coefficient
b = 1.0          # convection coefficient

# Source term f(x)
def f(x):
    return torch.ones_like(x)

# %% [markdown]
# ## Neural Network Architecture

# %%
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        z = x
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        return self.layers[-1](z)

# Initialize model
layers = [1, 32, 32, 32, 1]
model = PINN(layers).to(DEVICE)

# %% [markdown]
# ## Physics-Informed Loss Function

# %%
def pde_residual(x, model):
    x.requires_grad_(True)
    u = model(x)

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

    residual = -epsilon * d2u_dx2 + b * du_dx - f(x)
    return residual

# %% [markdown]
# ## Training Data

# %%
# Collocation points inside the domain
N_f = 100
x_f = torch.rand((N_f, 1), device=DEVICE)

# Boundary points
x_bc = torch.tensor([[0.0], [1.0]], device=DEVICE)
u_bc = torch.zeros_like(x_bc)

# %% [markdown]
# ## Training Loop

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # PDE loss
    res = pde_residual(x_f, model)
    loss_pde = torch.mean(res**2)

    # Boundary condition loss
    u_pred_bc = model(x_bc)
    loss_bc = torch.mean((u_pred_bc - u_bc)**2)

    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.3e}")

# %% [markdown]
# ## Results and Visualization

# %%
x_plot = torch.linspace(0, 1, 200).view(-1, 1).to(DEVICE)
u_pred = model(x_plot).detach().cpu().numpy()

plt.figure(figsize=(6,4))
plt.plot(x_plot.cpu().numpy(), u_pred, label="PINN solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Neural Network Solution of Convection–Diffusion Equation")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Key Takeaways
# - Neural networks can approximate solutions of PDEs without meshes.
# - Physics constraints are enforced via the loss function.
# - PINNs are particularly attractive for convection–diffusion problems
#   with sharp boundary layers.
