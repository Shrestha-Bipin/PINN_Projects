import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sys

torch.cuda.init()
torch.cuda.empty_cache()

# =========================
# DEVICE SETUP (CUDA ONLY)
# =========================
if not torch.cuda.is_available():
    print("❌ CUDA is not available. This code requires a GPU.")
    sys.exit(1)

DEVICE = torch.device("cuda")
print(f"✅ Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU Memory: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total")
except:
    pass


# =========================
# PHYSICAL CONSTANTS
# =========================
T_env = -5.0       # Ambient temperature
T_initial = 100.0  # Initial temperature
R = 0.005          # Cooling rate


# =========================
# DATA GENERATION
# =========================
time_s = torch.linspace(0, 900, 10000, dtype=torch.float32).view(-1, 1)

def get_exact_solution(t):
    return T_env + (T_initial - T_env) * torch.exp(-R * t)

T_data = get_exact_solution(time_s).detach()


# Move to GPU + enable grad correctly
time_s = time_s.to(DEVICE).requires_grad_(True)
T_data = T_data.to(DEVICE)


# =========================
# PINN MODEL
# =========================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1)
        )

    def forward(self, t):
        return self.net(t)


model = PINN().to(DEVICE)


# =========================
# OPTIMIZER + SCHEDULER
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)


# =========================
# TRAINING SETUP
# =========================
num_epochs = 200000

training_losses = []
physics_losses = []
data_losses = []
weight_physics_losses = []


# =========================
# TRAINING LOOP
# =========================
try:
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        model.train()
        optimizer.zero_grad()

        # Forward pass
        T_pred = model(time_s)

        # Physics loss (ODE residual)
        dT_dt = torch.autograd.grad(
            T_pred,
            time_s,
            grad_outputs=torch.ones_like(T_pred),
            create_graph=True
        )[0]

        physics_residual = dT_dt + R * (T_pred - T_env)
        physics_loss = torch.mean(physics_residual ** 2)

        # Data loss
        data_loss = torch.mean((T_pred - T_data) ** 2)

        # Weighted loss
        weight_physics_loss = 1000 * physics_loss
        loss = data_loss + weight_physics_loss

        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Store losses
        training_losses.append(loss.item())
        physics_losses.append(physics_loss.item())
        data_losses.append(data_loss.item())
        weight_physics_losses.append(weight_physics_loss.item())

        # Logging
        if epoch == 1 or epoch % 1000 == 0 or epoch == num_epochs:
            print(f"\nEpoch {epoch}")
            print(f"Total Loss: {loss.item():.6f}")
            print(f"Data Loss: {data_loss.item():.6f}")
            print(f"Physics Loss: {physics_loss.item():.6f}")
            print(f"Weighted Physics Loss: {weight_physics_loss.item():.6f}")
            print(f"Time per epoch: {time.time() - start_time:.4f}s")


except torch.cuda.OutOfMemoryError:
    print("❌ CUDA out of memory! Try reducing model size or epochs.")
    torch.cuda.empty_cache()
    sys.exit(1)


# =========================
# FINAL OUTPUT PLOT
# =========================
model.eval()
with torch.no_grad():
    T_pred = model(time_s).cpu()

time_cpu = time_s.detach().cpu()
T_data_cpu = T_data.detach().cpu()

plt.figure()
plt.plot(time_cpu.numpy(), T_data_cpu.numpy(), label="Exact Solution")
plt.plot(time_cpu.numpy(), T_pred.numpy(), label="PINN Prediction")
plt.legend()
plt.title("Cooling Curve (PINN vs Exact)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.show()