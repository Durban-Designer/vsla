#!/usr/bin/env python3
"""
VSLA Python Basic Usage Examples

Demonstrates core VSLA concepts for Python developers:
- Variable-shape tensor operations
- Ambient promotion semantics  
- Dual semiring models
- Memory efficiency advantages
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import vsla

def demo_ambient_promotion():
    """Demonstrate VSLA's core advantage: automatic ambient promotion"""
    print("🚀 VSLA Ambient Promotion Demo")
    print("=" * 40)
    
    # Create tensors of different shapes
    a = vsla.Tensor(np.array([1.0, 2.0, 3.0]))  # shape (3,)  
    b = vsla.Tensor(np.array([4.0, 5.0]))       # shape (2,)
    
    print(f"Tensor A: {a.to_numpy()} (shape: {a.shape()})")
    print(f"Tensor B: {b.to_numpy()} (shape: {b.shape()})")
    
    # VSLA's ambient promotion: [1,2,3] + [4,5] → [5,7,3]
    try:
        c = a.add(b)  # Should work with ambient promotion
        result = c.to_numpy()
        print(f"A + B = {result}")
        print("✅ SUCCESS: Variable-shape addition with ambient promotion!")
        
        # Verify correctness
        expected = np.array([5.0, 7.0, 3.0])  # [4,5,0] + [1,2,3]
        if np.allclose(result, expected):
            print("✅ CORRECT: Result matches mathematical expectation")
        else:
            print(f"⚠️  Unexpected result: got {result}, expected {expected}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("Note: This indicates the Python interface needs debugging")
    
    print()

def demo_model_differences():
    """Show the difference between Model A (convolution) and Model B (Kronecker)"""
    print("🔄 Dual Semiring Models Demo")  
    print("=" * 30)
    
    # Model A: Convolution semiring
    print("Model A (Convolution Semiring):")
    try:
        signal = vsla.Tensor(np.array([1.0, 2.0, 3.0]), model=vsla.Model.A)
        kernel = vsla.Tensor(np.array([0.5, 0.5]), model=vsla.Model.A)
        
        print(f"  Signal: {signal.to_numpy()}")
        print(f"  Kernel: {kernel.to_numpy()}")
        
        conv_result = signal.convolve(kernel)
        print(f"  Convolution: {conv_result.to_numpy()}")
        print("  ✅ FFT-accelerated convolution for signal processing")
        
    except Exception as e:
        print(f"  ❌ Model A failed: {e}")
    
    # Model B: Kronecker semiring  
    print("\nModel B (Kronecker Semiring):")
    try:
        a = vsla.Tensor(np.array([1.0, 2.0]), model=vsla.Model.B)
        b = vsla.Tensor(np.array([3.0, 4.0]), model=vsla.Model.B)
        
        print(f"  Vector A: {a.to_numpy()}")
        print(f"  Vector B: {b.to_numpy()}")
        
        kron_result = a.kronecker(b)  # Should be [3, 4, 6, 8]
        print(f"  Kronecker: {kron_result.to_numpy()}")
        print("  ✅ Tensor products for quantum computing and tensor networks")
        
    except Exception as e:
        print(f"  ❌ Model B failed: {e}")
    
    print()

def demo_memory_efficiency():
    """Compare VSLA vs traditional padding approaches"""
    print("💾 Memory Efficiency Comparison")
    print("=" * 35)
    
    # Simulate variable-length sequences (common in NLP/time series)
    sequences = [
        [1.0, 2.0],
        [3.0, 4.0, 5.0], 
        [6.0],
        [7.0, 8.0, 9.0, 10.0]
    ]
    
    # Traditional approach: pad to maximum length
    max_len = max(len(seq) for seq in sequences)
    padded_np = np.array([seq + [0.0] * (max_len - len(seq)) for seq in sequences])
    
    print("Traditional NumPy approach (with padding):")
    print(f"  Max length: {max_len}")  
    print(f"  Padded array shape: {padded_np.shape}")
    print(f"  Memory usage: {padded_np.nbytes} bytes")
    total_elements = sum(len(seq) for seq in sequences)
    wasted_elements = padded_np.size - total_elements
    waste_percentage = (wasted_elements / padded_np.size) * 100
    print(f"  Wasted memory: {waste_percentage:.1f}% ({wasted_elements}/{padded_np.size} elements)")
    
    # VSLA approach: native variable-shape storage
    print("\nVSLA approach (no padding):")
    try:
        vsla_tensors = [vsla.Tensor(np.array(seq)) for seq in sequences]
        total_memory = sum(len(seq) * 8 for seq in sequences)  # 8 bytes per double
        print(f"  Individual shapes: {[t.shape() for t in vsla_tensors]}")
        print(f"  Total memory: {total_memory} bytes")
        print(f"  Memory efficiency: {padded_np.nbytes / total_memory:.1f}x better than padding")
        print("  ✅ Zero waste: Only store actual data")
        
        # Stacking still works with ambient promotion
        if hasattr(vsla, 'stack'):
            stacked = vsla.stack(vsla_tensors)
            print(f"  Stacked shape: {stacked.shape()} (ambient promotion)")
        
    except Exception as e:
        print(f"  ❌ VSLA approach failed: {e}")
    
    print()

def demo_practical_applications():
    """Show practical use cases where VSLA excels"""
    print("🎯 Practical Applications")
    print("=" * 25)
    
    print("1. Variable-length sequence processing:")
    print("   - NLP: sentences of different lengths")
    print("   - Time series: irregular sampling rates") 
    print("   - Audio: variable duration signals")
    print("   ✅ No padding waste, native variable operations")
    
    print("\n2. Signal processing:")
    print("   - Convolution with different kernel sizes")
    print("   - Multi-rate filtering")  
    print("   - Adaptive filter lengths")
    print("   ✅ FFT acceleration, no zero-padding artifacts")
    
    print("\n3. Machine learning:")
    print("   - Transformer attention with variable context")
    print("   - RNN sequences without padding")
    print("   - Dynamic neural network architectures")
    print("   ✅ Mathematical rigor, proven efficiency gains")
    
    print("\n4. Scientific computing:")
    print("   - Tensor networks with variable bond dimensions")
    print("   - Quantum circuit simulation")
    print("   - Multi-physics coupling")
    print("   ✅ Principled variable-shape mathematics")
    
    print()

def run_basic_examples():
    """Run all basic examples"""
    print("🧪 VSLA Python Interface - Basic Examples")
    print("=" * 50)
    print("Demonstrating variable-shape linear algebra concepts")
    print()
    
    demo_ambient_promotion()
    demo_model_differences()  
    demo_memory_efficiency()
    demo_practical_applications()
    
    print("📚 For more examples, see:")
    print("   - benchmarks/python/ for performance comparisons")
    print("   - examples/python/ for advanced usage patterns") 
    print("   - docs/PYTHON_QUICKSTART.md for complete guide")

if __name__ == "__main__":
    run_basic_examples()