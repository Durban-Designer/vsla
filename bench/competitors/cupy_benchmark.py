#!/usr/bin/env python3
"""
CuPy benchmark implementation for VSLA comparison.
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class CupyBenchmark:
    """CuPy benchmark implementation."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        if CUPY_AVAILABLE:
            cp.cuda.Device(device_id).use()
    
    def is_available(self) -> bool:
        """Check if CuPy is available."""
        return CUPY_AVAILABLE
    
    def benchmark_vector_addition(self, size1: int, size2: int, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark variable-shape vector addition."""
        if not CUPY_AVAILABLE:
            return {'error': 'CuPy not available'}
        
        # Create vectors of different sizes (simulating variable shapes)
        max_size = max(size1, size2)
        a = cp.random.random(size1).astype(cp.float64)
        b = cp.random.random(size2).astype(cp.float64)
        
        # Manual padding to common size (what users have to do)
        a_padded = cp.zeros(max_size, dtype=cp.float64)
        b_padded = cp.zeros(max_size, dtype=cp.float64)
        a_padded[:size1] = a
        b_padded[:size2] = b
        
        # Warmup
        for _ in range(5):
            result = a_padded + b_padded
        
        # Benchmark
        cp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = a_padded + b_padded
        
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        
        mean_time_us = (end_time - start_time) * 1e6 / iterations
        
        return {
            'method': 'cupy_manual_padding',
            'operation': 'vector_addition',
            'size1': size1,
            'size2': size2,
            'result_size': max_size,
            'iterations': iterations,
            'mean_time_us': mean_time_us,
            'memory_mb': self._get_memory_usage()
        }
    
    def benchmark_matrix_multiplication(self, m: int, n: int, k: int, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark matrix multiplication."""
        if not CUPY_AVAILABLE:
            return {'error': 'CuPy not available'}
        
        a = cp.random.random((m, k)).astype(cp.float64)
        b = cp.random.random((k, n)).astype(cp.float64)
        
        # Warmup
        for _ in range(5):
            result = cp.matmul(a, b)
        
        # Benchmark
        cp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = cp.matmul(a, b)
        
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        
        mean_time_us = (end_time - start_time) * 1e6 / iterations
        
        return {
            'method': 'cupy_matmul',
            'operation': 'matrix_multiplication',
            'matrix_size': f'{m}x{k}x{n}',
            'iterations': iterations,
            'mean_time_us': mean_time_us,
            'memory_mb': self._get_memory_usage()
        }
    
    def benchmark_convolution(self, signal_size: int, kernel_size: int, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark convolution operation."""
        if not CUPY_AVAILABLE:
            return {'error': 'CuPy not available'}
        
        signal = cp.random.random(signal_size).astype(cp.float64)
        kernel = cp.random.random(kernel_size).astype(cp.float64)
        
        # Warmup
        for _ in range(5):
            result = cp.convolve(signal, kernel, mode='full')
        
        # Benchmark
        cp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = cp.convolve(signal, kernel, mode='full')
        
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        
        mean_time_us = (end_time - start_time) * 1e6 / iterations
        
        return {
            'method': 'cupy_convolve',
            'operation': 'convolution',
            'signal_size': signal_size,
            'kernel_size': kernel_size,
            'result_size': signal_size + kernel_size - 1,
            'iterations': iterations,
            'mean_time_us': mean_time_us,
            'memory_mb': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if not CUPY_AVAILABLE:
            return 0.0
        
        try:
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes() / (1024 * 1024)
        except:
            return 0.0

def main():
    """Run CuPy benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CuPy benchmark runner')
    parser.add_argument('--operation', type=str, required=True,
                       choices=['vector_add', 'matrix_multiply', 'convolution'],
                       help='Operation to benchmark')
    parser.add_argument('--size1', type=int, default=1024,
                       help='First dimension size')
    parser.add_argument('--size2', type=int, default=1024,
                       help='Second dimension size')
    parser.add_argument('--size3', type=int, default=1024,
                       help='Third dimension size (for matrix multiply)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations')
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device ID')
    
    args = parser.parse_args()
    
    benchmark = CupyBenchmark(args.device)
    
    if not benchmark.is_available():
        print(json.dumps({'error': 'CuPy not available'}))
        return
    
    if args.operation == 'vector_add':
        result = benchmark.benchmark_vector_addition(args.size1, args.size2, args.iterations)
    elif args.operation == 'matrix_multiply':
        result = benchmark.benchmark_matrix_multiplication(args.size1, args.size2, args.size3, args.iterations)
    elif args.operation == 'convolution':
        result = benchmark.benchmark_convolution(args.size1, args.size2, args.iterations)
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()