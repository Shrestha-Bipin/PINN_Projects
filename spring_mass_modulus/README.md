# Spring-Mass PINN with NVIDIA Modulus Sym

This project solves a coupled three-mass spring system with a physics-informed neural network (PINN) using NVIDIA Modulus Sym.

## Project Structure

```text
.
|-- conf/
|   `-- config.yaml          # Modulus Sym training configuration
|-- mass_spring_ode.ipynb    # Notebook draft for the ODE definition
|-- spring_mass_solver.ipynb # Notebook draft for the full solver workflow
|-- spring_mass_solver.py    # Clean runnable training script
|-- requirements.txt
`-- .gitignore
```

## System

The modeled system is:

```text
m1*x1_tt + (k1 + k2)*x1 - k2*x2 = 0
m2*x2_tt + (k2 + k3)*x2 - k2*x1 - k3*x3 = 0
m3*x3_tt + (k3 + k4)*x3 - k3*x2 = 0
```

Default parameters:

```text
k = (2, 1, 1, 2)
m = (1, 1, 1)
initial state = x1(0)=1, x2(0)=0, x3(0)=0, x1_t(0)=x2_t(0)=x3_t(0)=0
time range = [0, 20]
```

## Setup

Use a Python environment compatible with NVIDIA Modulus Sym. A GPU-enabled Linux environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python spring_mass_solver.py
```

Training settings are in `conf/config.yaml`.

