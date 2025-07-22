#!/usr/bin/env python3
"""
VSLA Benchmark Suite (New API)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    sizes: List[int]
    iterations: int

class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.bench_root = Path(__file__).parent
        self.new_benchmark_c = self.bench_root / 'src' / 'new_benchmark.c'
        self.new_benchmark_exe = self.bench_root / 'build' / 'new_benchmark'

    def ensure_built(self) -> bool:
        """Ensure the new benchmark is built."""
        if not self.new_benchmark_exe.exists():
            print("New benchmark not found. Building...")
            try:
                build_cmd = [
                    'gcc', '-I', '../include', str(self.new_benchmark_c), '../build/libvsla.a',
                    '-lm', '-lpthread', '-o', str(self.new_benchmark_exe)
                ]
                result = subprocess.run(build_cmd, cwd=self.bench_root, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Build failed: {result.stderr}")
                    return False
                print("New benchmark built successfully")
            except Exception as e:
                print(f"Failed to build new benchmark: {e}")
                return False
        return True

    def run_benchmarks(self) -> None:
        """Run the benchmark suite."""
        if not self.ensure_built():
            return

        print("\n--- Running New Benchmarks ---")
        for size in self.config.sizes:
            cmd = [str(self.new_benchmark_exe), str(size), str(self.config.iterations)]
            subprocess.run(cmd, cwd=self.bench_root)

def main():
    parser = argparse.ArgumentParser(description='VSLA Benchmark Suite')
    parser.add_argument('--sizes', type=str, default='128,256,512,1024',
                       help='Comma-separated list of test sizes')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per test')
    args = parser.parse_args()

    config = BenchmarkConfig(
        sizes=[int(x) for x in args.sizes.split(',')],
        iterations=args.iterations,
    )

    runner = BenchmarkRunner(config)
    runner.run_benchmarks()

if __name__ == '__main__':
    main()
