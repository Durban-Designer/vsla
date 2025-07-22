# VSLA Python Bindings

This package provides Python bindings for the VSLA (Variable-Shape Linear Algebra) library.

## Installation

To install the Python bindings, you can use `pip`:

```bash
pip install vsla
```

To install in editable mode for development, clone the repository and run the following command from the `python/` directory:

```bash
pip install -e .
```

## Basic Usage

```python
import numpy as np
import vsla

# Create a VSLA tensor from a NumPy array
a = vsla.Tensor(np.array([1.0, 2.0, 3.0]))
b = vsla.Tensor(np.array([4.0, 5.0, 6.0, 7.0]))

# Perform variable-shape addition
result = a + b

# Convert the result back to a NumPy array
result_np = result.to_numpy()

print(f"Result shape: {result.shape()}")
print(f"Result data: {result_np}")
```
