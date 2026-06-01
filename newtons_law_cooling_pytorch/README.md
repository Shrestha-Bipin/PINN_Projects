# Newton's Law of Cooling PINN

A Physics-Informed Neural Network (PINN) implementation for modeling Newton's Law of Cooling using PyTorch.

## Overview

This project trains a neural network to approximate the temperature curve of an object cooling over time. The model is guided by both generated temperature data and the physical differential equation:

```text
dT/dt = -R(T - T_env)
```

where:

- `T` is the object temperature at time `t`
- `T_env` is the ambient temperature
- `R` is the cooling rate constant
- `dT/dt` is the rate of temperature change

The exact analytical solution is used as reference data:

```text
T(t) = T_env + (T_initial - T_env) * exp(-R * t)
```

## Project Files

```text
.
├── newtons_law_of_cooling.ipynb   # Main Jupyter notebook
├── netwons_law_of_cooling.py      # Python script version
└── README.md                      # Project documentation
```

Note: `netwons_law_of_cooling.py` appears to have a filename typo. A clearer name would be `newtons_law_of_cooling.py`.

## What The Notebook Does

- Defines the Newton's Law of Cooling equation.
- Generates synthetic temperature data from the exact analytical solution.
- Builds a simple fully connected neural network using PyTorch.
- Trains the model with a combined data loss and physics loss.
- Compares the PINN prediction against the exact solution.
- Plots training losses and cooling curves.

## Requirements

Install the main dependencies:

```bash
pip install torch numpy matplotlib jupyter
```

The Python script version currently expects CUDA/GPU support. The notebook also targets CUDA.

## How To Run

### Run the notebook

```bash
jupyter notebook newtons_law_of_cooling.ipynb
```

Then run the cells from top to bottom.

### Run the Python script

```bash
python netwons_law_of_cooling.py
```

If CUDA is not available, the script will exit because it is written for GPU execution.

## Model Details

The PINN learns a mapping:

```text
t -> T(t)
```

The training objective combines:

- Data loss: mean squared error between predicted and exact temperatures.
- Physics loss: mean squared residual of the cooling differential equation.

The total loss is:

```text
loss = data_loss + 1000 * physics_loss
```

