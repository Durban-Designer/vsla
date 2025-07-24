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
    
    # Test with variable shapes (VSLA's core strength)
    t2 = vsla.Tensor(np.array([4.0, 5.0]))  # shorter tensor (length 2)
    print('Created tensor 2 (short):', t2)
    
    # This should work with VSLA's ambient promotion: [1,2,3] + [4,5] -> [5,7,3]
    print(f"Attempting VSLA ambient promotion: {t.to_numpy()} + {t2.to_numpy()}")
    try:
        t3 = t.add(t2)
        result = t3.to_numpy()
        print('✅ Variable-shape sum successful:', result)
        print('Expected: [5.0, 7.0, 3.0] (ambient promotion with trailing zeros)')
    except Exception as e:
        print('❌ Variable-shape addition failed:', e)
        print('This indicates the C library needs ambient promotion implementation')
    
    # Test same-shape addition (should work)
    t4 = vsla.Tensor(np.array([4.0, 5.0, 6.0]))
    print('\nTesting same-shape addition:', t.to_numpy(), '+', t4.to_numpy())
    try:
        t5 = t.add(t4)
        result = t5.to_numpy()
        print('✅ Same-shape sum:', result)
    except Exception as e:
        print('❌ Even same-shape addition failed:', e)
else:
    print('Using pure Python fallback')
    t = vsla.tensor([1, 2, 3])
    print('Created fallback tensor:', t)