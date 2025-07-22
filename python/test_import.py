#!/usr/bin/env python3
"""Test VSLA import and functionality"""

try:
    import vsla._core as core
    print('Core loaded:', core)
except Exception as e:
    print('Error loading core:', e)
    
import vsla
print('Has C extension:', vsla._has_core)

if vsla._has_core:
    import numpy as np
    t = vsla.Tensor(np.array([1.0, 2.0, 3.0]))
    print('Created tensor:', t)
    
    t2 = vsla.Tensor(np.array([4.0, 5.0]))
    print('Created tensor 2:', t2)
    
    # Test addition
    t3 = t.add(t2)
    print('Sum:', t3)
else:
    print('Using pure Python fallback')
    t = vsla.tensor([1, 2, 3])
    print('Created fallback tensor:', t)