"""
VSLA: Variable-Shape Linear Algebra

Mathematical foundations and high-performance implementation of variable-shape linear algebra
with automatic dimension promotion and semiring structures.

Backend Selection:
The computation backend (CPU/CUDA) is automatically selected at import time based on hardware
availability. Runtime backend switching is not supported for performance reasons.

Key Features:
- Automatic shape promotion in tensor operations
- Matrix multiplication, convolution, and reduction operations  
- Multi-dimensional tensor support (up to 8D)
- Memory-efficient tensor representations
- Both semiring models A (addition/convolution) and B (addition/Kronecker)
"""

__version__ = "0.1.0"
__author__ = "Royce Birnbaum"
__email__ = "royce.birnbaum@gmail.com"

# Import core functionality when C extension is available
try:
    from ._core import *
    _has_core = True
except ImportError:
    _has_core = False
    import warnings
    warnings.warn(
        "VSLA C extension not available. "
        "Install with: pip install vsla[dev] and rebuild.",
        ImportWarning
    )

# Pure Python fallbacks (minimal implementation)
if not _has_core:
    import numpy as np
    
    class VslaTensor:
        """Pure Python fallback for VSLA tensor operations"""
        
        def __init__(self, data, model='A'):
            self.data = np.asarray(data)
            self.model = model
            self.shape = self.data.shape
        
        def __add__(self, other):
            if isinstance(other, VslaTensor):
                # Simple zero-padding to common shape
                max_shape = tuple(max(a, b) for a, b in zip(self.shape, other.shape))
                self_padded = np.zeros(max_shape)
                other_padded = np.zeros(max_shape)
                
                self_padded[:self.shape[0]] = self.data
                other_padded[:other.shape[0]] = other.data
                
                return VslaTensor(self_padded + other_padded, self.model)
            return NotImplemented
        
        def convolve(self, other):
            """Convolution operation (Model A)"""
            if self.model != 'A':
                raise ValueError("Convolution only available in Model A")
            return VslaTensor(np.convolve(self.data, other.data), self.model)
        
        def kronecker(self, other):
            """Kronecker product operation (Model B)"""
            if self.model != 'B':
                raise ValueError("Kronecker product only available in Model B")
            return VslaTensor(np.kron(self.data, other.data), self.model)

    def tensor(data, model='A'):
        """Create a VSLA tensor"""
        return VslaTensor(data, model)

# Export public API
__all__ = [
    '__version__',
    '__author__',
    '__email__',
]

if _has_core:
    __all__.extend([
        # Core tensor operations (from C extension)
        'Tensor',
        'Model',
        'get_backend_info',
        'zeros',
        'ones',
    ])
else:
    __all__.extend([
        # Pure Python fallbacks
        'VslaTensor',
        'tensor',
    ])