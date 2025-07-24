#!/usr/bin/env python3
"""
VSLA vs PyTorch Ragged Tensor Benchmark

Compares VSLA's native variable-shape operations against PyTorch's
nested tensors and manual padding approaches.

Single-threaded CPU-only for fair comparison.
"""

import sys
import os
import time
import statistics
from typing import List, Tuple, Dict, Any

# Add parent directory to path for VSLA import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

# Set to single-threaded for fair comparison
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)  # Single-threaded CPU
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "not available"

try:
    import vsla
    VSLA_AVAILABLE = vsla._has_core
except ImportError:
    VSLA_AVAILABLE = False

class BenchmarkSuite:
    """Comprehensive benchmark comparing VSLA to PyTorch ragged operations"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_func(self, func, *args, num_passes=10):
        """Run statistical benchmark with warmup"""
        # Warmup
        for _ in range(3):
            try:
                func(*args)
            except Exception:
                return {'mean': float('inf'), 'std': 0, 'failed': True}
        
        # Measure
        times = []
        for _ in range(num_passes):
            start = time.perf_counter()
            try:
                result = func(*args)
                end = time.perf_counter()
                times.append(end - start)
            except Exception:
                times.append(float('inf'))
        
        return {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'failed': any(t == float('inf') for t in times)
        }
    
    def test_variable_addition(self):
        """Compare variable-shape addition approaches"""
        print("🔢 Variable-Shape Addition: VSLA vs PyTorch")
        print("=" * 50)
        
        test_cases = [
            ([1.0, 2.0, 3.0], [4.0, 5.0]),
            ([1.0, 2.0], [3.0, 4.0, 5.0, 6.0]),
            (list(range(1, 11)), list(range(1, 6))),
        ]
        
        for i, (a_data, b_data) in enumerate(test_cases):
            print(f"\n--- Test {i+1}: [{len(a_data)}] + [{len(b_data)}] ---")
            
            # PyTorch approach 1: Manual padding
            def pytorch_padded():
                max_len = max(len(a_data), len(b_data))
                a_padded = torch.tensor(a_data + [0.0] * (max_len - len(a_data)))
                b_padded = torch.tensor(b_data + [0.0] * (max_len - len(b_data)))
                return a_padded + b_padded
            
            # PyTorch approach 2: Nested tensors (if available)
            def pytorch_nested():
                if hasattr(torch, 'nested') and hasattr(torch.nested, 'nested_tensor'):
                    a_tensor = torch.tensor(a_data)
                    b_tensor = torch.tensor(b_data)
                    # Note: nested tensor addition is limited
                    return torch.nested.nested_tensor([a_tensor, b_tensor])
                else:
                    raise NotImplementedError("Nested tensors not available")
            
            # VSLA approach: Native variable-shape
            def vsla_native():
                if not VSLA_AVAILABLE:
                    raise RuntimeError("VSLA not available")
                a = vsla.Tensor(np.array(a_data))
                b = vsla.Tensor(np.array(b_data))
                return a.add(b)
            
            # Benchmark all approaches
            pytorch_pad_result = self.benchmark_func(pytorch_padded) if TORCH_AVAILABLE else {'failed': True}
            pytorch_nest_result = self.benchmark_func(pytorch_nested) if TORCH_AVAILABLE else {'failed': True}
            vsla_result = self.benchmark_func(vsla_native)
            
            # Display results
            print(f"PyTorch (padded):  {pytorch_pad_result['mean']*1000:.3f} ms" if not pytorch_pad_result['failed'] else "PyTorch (padded):  FAILED")
            print(f"PyTorch (nested):  {pytorch_nest_result['mean']*1000:.3f} ms" if not pytorch_nest_result['failed'] else "PyTorch (nested):  FAILED/UNAVAILABLE")
            print(f"VSLA (native):     {vsla_result['mean']*1000:.3f} ms" if not vsla_result['failed'] else "VSLA (native):     FAILED")
            
            # Calculate speedups
            if not vsla_result['failed'] and not pytorch_pad_result['failed']:
                speedup = pytorch_pad_result['mean'] / vsla_result['mean']
                print(f"VSLA vs PyTorch:   {speedup:.2f}x {'(VSLA faster)' if speedup > 1 else '(PyTorch faster)'}")
    
    def test_memory_efficiency(self):
        """Compare memory usage patterns"""
        print("\n💾 Memory Efficiency Comparison")
        print("=" * 35)
        
        # Realistic variable-length sequence scenarios
        scenarios = [
            {
                'name': 'NLP Batch Processing',
                'sequences': [
                    [1, 2, 3],           # 3 tokens
                    [1, 2, 3, 4, 5, 6],  # 6 tokens  
                    [1, 2],              # 2 tokens
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
                ],
                'description': 'Variable-length sentences in NLP batch'
            },
            {
                'name': 'Time Series Data',
                'sequences': [
                    list(range(50)),   # 50 timesteps
                    list(range(120)),  # 120 timesteps
                    list(range(80)),   # 80 timesteps
                    list(range(200)),  # 200 timesteps
                ],
                'description': 'Irregular sampling rates in time series'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            print(f"Description: {scenario['description']}")
            sequences = scenario['sequences']
            
            # Calculate padding waste
            max_len = max(len(seq) for seq in sequences)
            total_actual = sum(len(seq) for seq in sequences)
            total_padded = len(sequences) * max_len
            waste_percent = ((total_padded - total_actual) / total_padded) * 100
            
            print(f"Sequence lengths: {[len(seq) for seq in sequences]}")
            print(f"Max length: {max_len}")
            
            # PyTorch padded approach
            if TORCH_AVAILABLE:
                padded_tensor = torch.tensor([seq + [0.0] * (max_len - len(seq)) for seq in sequences])
                pytorch_memory = padded_tensor.element_size() * padded_tensor.nelement()
                print(f"PyTorch (padded): {pytorch_memory / 1024:.1f} KB, {waste_percent:.1f}% waste")
            else:
                print("PyTorch: Not available")
            
            # VSLA native approach
            if VSLA_AVAILABLE:
                try:
                    vsla_tensors = [vsla.Tensor(np.array(seq, dtype=np.float64)) for seq in sequences]
                    vsla_memory = sum(len(seq) * 8 for seq in sequences)  # 8 bytes per double
                    efficiency = (pytorch_memory / vsla_memory) if TORCH_AVAILABLE else 0
                    print(f"VSLA (native):    {vsla_memory / 1024:.1f} KB, 0% waste ({efficiency:.1f}x efficient)")
                except Exception as e:
                    print(f"VSLA: Failed - {e}")
            else:
                print("VSLA: Not available")
    
    def test_sequence_processing(self):
        """Test realistic sequence processing scenarios"""
        print("\n🔄 Sequence Processing Scenarios")
        print("=" * 35)
        
        # Variable-length sequence data (simulating real NLP/time series)
        sequences = [
            np.random.randn(10),   # Short sequence
            np.random.randn(25),   # Medium sequence  
            np.random.randn(5),    # Very short sequence
            np.random.randn(40),   # Long sequence
        ]
        
        print(f"Processing {len(sequences)} variable-length sequences")
        print(f"Lengths: {[len(seq) for seq in sequences]}")
        
        # PyTorch approach: Pad and process
        def pytorch_batch_processing():
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            
            max_len = max(len(seq) for seq in sequences)
            padded = torch.stack([
                torch.cat([torch.tensor(seq), torch.zeros(max_len - len(seq))])
                for seq in sequences
            ])
            
            # Simulate processing: normalization across batch
            normalized = torch.nn.functional.normalize(padded, dim=1)
            return normalized.sum(dim=0)  # Aggregate result
        
        # VSLA approach: Native variable-shape processing
        def vsla_batch_processing():
            if not VSLA_AVAILABLE:
                raise RuntimeError("VSLA not available")
            
            tensors = [vsla.Tensor(seq) for seq in sequences]
            
            # Process each tensor individually (no padding needed)
            processed = []
            for tensor in tensors:
                # Simulate normalization
                norm = tensor.norm()
                if norm > 0:
                    normalized = tensor * (1.0 / norm)  # Element-wise scaling
                    processed.append(normalized)
            
            # Aggregate with ambient promotion
            if processed:
                result = processed[0]
                for tensor in processed[1:]:
                    result = result.add(tensor)
                return result
            return None
        
        # Benchmark both approaches
        pytorch_result = self.benchmark_func(pytorch_batch_processing)
        vsla_result = self.benchmark_func(vsla_batch_processing)
        
        print(f"PyTorch (padded processing): {pytorch_result['mean']*1000:.3f} ms" if not pytorch_result['failed'] else "PyTorch: FAILED")
        print(f"VSLA (native processing):    {vsla_result['mean']*1000:.3f} ms" if not vsla_result['failed'] else "VSLA: FAILED")
        
        if not pytorch_result['failed'] and not vsla_result['failed']:
            speedup = pytorch_result['mean'] / vsla_result['mean']
            print(f"Speedup: {speedup:.2f}x {'(VSLA faster)' if speedup > 1 else '(PyTorch faster)'}")
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks and generate report"""
        print("🚀 VSLA vs PyTorch Comprehensive Benchmark")
        print("=" * 50)
        print("Single-threaded CPU comparison")
        print(f"PyTorch available: {TORCH_AVAILABLE} (version: {TORCH_VERSION})")
        print(f"VSLA available: {VSLA_AVAILABLE}")
        
        if TORCH_AVAILABLE:
            print(f"PyTorch device: {torch.get_default_dtype()}")
            print(f"PyTorch threads: {torch.get_num_threads()}")
        
        # Run benchmark categories
        self.test_variable_addition()
        self.test_memory_efficiency()
        self.test_sequence_processing()
        
        # Summary
        print("\n📊 Benchmark Summary")
        print("=" * 20)
        
        if VSLA_AVAILABLE and TORCH_AVAILABLE:
            print("✅ Both frameworks available for comparison")
            print("Key findings:")
            print("• VSLA provides native variable-shape operations")
            print("• PyTorch requires manual padding or nested tensors")
            print("• Memory efficiency varies by use case")
            print("• Performance depends on sequence length distribution")
        elif VSLA_AVAILABLE:
            print("⚠️  Only VSLA available - install PyTorch for full comparison")
        elif TORCH_AVAILABLE:
            print("⚠️  Only PyTorch available - VSLA Python interface needs debugging")
        else:
            print("❌ Neither framework available for testing")
        
        print(f"\n📚 Additional Resources:")
        print(f"• VSLA Documentation: PYTHON_QUICKSTART.md")
        print(f"• PyTorch Nested Tensors: https://pytorch.org/docs/stable/nested.html")
        print(f"• Benchmark Source: python/benchmarks/")

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_comprehensive_benchmark()