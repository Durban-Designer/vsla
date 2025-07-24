#!/usr/bin/env python3
"""Debug test for VSLA Python interface"""

import numpy as np
import vsla

print("=== Testing VSLA Variable-Shape Operations ===")

# Test 1: Create tensors of different shapes
print("\n1. Creating variable-shape tensors:")
a = vsla.Tensor(np.array([1.0, 2.0, 3.0]))  # shape (3,)
b = vsla.Tensor(np.array([4.0, 5.0]))       # shape (2,)

print(f"Tensor A: {a.to_numpy()} (shape: {a.shape()})")
print(f"Tensor B: {b.to_numpy()} (shape: {b.shape()})")

# Test 2: Try variable-shape addition (should work with VSLA!)
print(f"\n2. Attempting VSLA variable-shape addition:")
print(f"   A + B = {a.to_numpy()} + {b.to_numpy()}")
print(f"   Expected: [5.0, 7.0, 3.0] (ambient promotion)")

try:
    c = a.add(b)
    result = c.to_numpy()
    print(f"   ✅ SUCCESS: {result}")
    if np.allclose(result, [5.0, 7.0, 3.0]):
        print("   ✅ CORRECT: Result matches expected ambient promotion!")
    else:
        print(f"   ⚠️  UNEXPECTED: Got {result}, expected [5.0, 7.0, 3.0]")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    print(f"   This indicates the Python bindings aren't working with the C library properly")

# Test 3: Same-shape addition (should definitely work)
print(f"\n3. Testing same-shape addition:")
d = vsla.Tensor(np.array([4.0, 5.0, 6.0]))  # shape (3,)
print(f"   A + D = {a.to_numpy()} + {d.to_numpy()}")

try:
    e = a.add(d)
    result = e.to_numpy()
    print(f"   ✅ SUCCESS: {result}")
    if np.allclose(result, [5.0, 7.0, 9.0]):
        print("   ✅ CORRECT: Same-shape addition works!")
    else:
        print(f"   ⚠️  UNEXPECTED: Got {result}, expected [5.0, 7.0, 9.0]")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n=== Debug Test Complete ===")