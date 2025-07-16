#!/usr/bin/env python3
"""
VSLA Complete Benchmark Suite
Single command to run all benchmarks and generate comprehensive reports.
Usage: python3 run_benchmark.py [--quick] [--sizes SIZE1,SIZE2,...]
"""

import os
import sys
import json
import time
import argparse
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    sizes: List[int]
    iterations: int
    warmup: int
    output_dir: str
    enable_gpu: bool
    enable_competitors: bool
    enable_cpu: bool
    precision: str
    reproducible: bool

class SystemInfo:
    """Gather system information for reproducible benchmarks."""
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            gpu_info = result.stdout.strip().split(', ')
            return {
                'name': gpu_info[0],
                'memory_mb': int(gpu_info[1]),
                'driver_version': gpu_info[2],
                'compute_capability': gpu_info[3],
                'available': True
            }
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
            return {'available': False}
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
            
            cpu_info = {}
            for line in lines:
                if line.startswith('model name'):
                    cpu_info['name'] = line.split(':')[1].strip()
                elif line.startswith('cpu cores'):
                    cpu_info['cores'] = int(line.split(':')[1].strip())
                elif line.startswith('siblings'):
                    cpu_info['threads'] = int(line.split(':')[1].strip())
            
            return cpu_info
        except Exception:
            return {'name': 'Unknown', 'cores': 0, 'threads': 0}
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get system memory information."""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    return {'total_gb': mem_kb // 1024 // 1024}
            
            return {'total_gb': 0}
        except Exception:
            return {'total_gb': 0}

    @staticmethod
    def get_system_fingerprint() -> str:
        """Generate system fingerprint for report naming."""
        cpu_info = SystemInfo.get_cpu_info()
        gpu_info = SystemInfo.get_gpu_info()
        mem_info = SystemInfo.get_memory_info()
        
        # Create compact system identifier
        cpu_name = cpu_info.get('name', 'Unknown').replace('Intel(R) Core(TM) ', '').replace(' CPU', '').replace(' ', '')
        gpu_name = gpu_info.get('name', 'NoGPU').replace('NVIDIA GeForce ', '').replace(' ', '').replace('Laptop', '')
        memory = f"{mem_info.get('total_gb', 0)}GB"
        
        return f"{cpu_name}_{gpu_name}_{memory}"

class CPUBenchmark:
    """CPU benchmark runner using VSLA native operations."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.vsla_root = Path(__file__).parent.parent  # bench folder is inside vsla
        self.bench_root = Path(__file__).parent
        self.cpu_benchmark = self.bench_root / 'build' / 'cpu_benchmark'
        
    def ensure_built(self) -> bool:
        """Ensure CPU benchmark is built."""
        if not self.cpu_benchmark.exists():
            print("CPU benchmark not found. Building...")
            try:
                build_cmd = [
                    'gcc', '-I', '../include', 'src/cpu_benchmark.c', '../build/libvsla.a',
                    '-lm', '-lpthread', '-o', 'build/cpu_benchmark'
                ]
                
                result = subprocess.run(build_cmd, cwd=self.bench_root, 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"CPU build failed: {result.stderr}")
                    return False
                    
                print("CPU benchmark built successfully")
            except Exception as e:
                print(f"Failed to build CPU benchmark: {e}")
                return False
        
        return True
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run CPU benchmark."""
        if not self.ensure_built():
            return {"error": "Failed to build CPU benchmark"}
        
        try:
            if operation == "vector_add":
                cmd = [str(self.cpu_benchmark), "vector_add", str(size), str(size), str(self.config.iterations)]
            elif operation == "convolution":
                # Use smaller second dimension for convolution
                size2 = max(8, size // 8)
                cmd = [str(self.cpu_benchmark), "convolution", str(size), str(size2), str(self.config.iterations)]
            elif operation == "kronecker":
                # Use much smaller sizes for Kronecker due to O(n*m) output
                size1 = min(size, 64)
                size2 = min(size // 4, 16)
                cmd = [str(self.cpu_benchmark), "kronecker", str(size1), str(size2), str(self.config.iterations)]
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=self.bench_root)
            data = json.loads(result.stdout)
            data['platform'] = 'cpu'
            return data
            
        except subprocess.CalledProcessError as e:
            return {"error": f"CPU benchmark failed: {e.stderr}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse CPU benchmark output: {e}"}

class GPUBenchmark:
    """GPU benchmark runner using VSLA GPU implementation."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.vsla_root = Path(__file__).parent.parent  # bench folder is inside vsla
        self.bench_root = Path(__file__).parent
        self.gpu_benchmark = self.bench_root / 'build' / 'gpu_head_to_head'
        
    def ensure_built(self) -> bool:
        """Ensure GPU benchmark is built."""
        if not self.gpu_benchmark.exists():
            print("GPU benchmark not found. Building...")
            try:
                build_cmd = [
                    'gcc', '-I', '../include', 'src/gpu_head_to_head.c', '../build/libvsla.a',
                    '-lm', '-lpthread', '-lcudart', 
                    '-L/usr/local/cuda-12.6/targets/x86_64-linux/lib',
                    '-o', 'build/gpu_head_to_head'
                ]
                
                env = os.environ.copy()
                env['PATH'] = '/usr/local/cuda-12.6/bin:' + env.get('PATH', '')
                
                result = subprocess.run(build_cmd, cwd=self.bench_root, 
                                      capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    print(f"GPU build failed: {result.stderr}")
                    return False
                    
                print("GPU benchmark built successfully")
            except Exception as e:
                print(f"Failed to build GPU benchmark: {e}")
                return False
        
        return True
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run GPU benchmark."""
        if not self.ensure_built():
            return {"error": "Failed to build GPU benchmark"}
        
        try:
            if operation == "matrix_multiply":
                cmd = [str(self.gpu_benchmark), "matrix_multiply", str(size), str(self.config.iterations)]
            elif operation == "vector_add":
                cmd = [str(self.gpu_benchmark), "vector_add", str(size), str(size), str(self.config.iterations)]
            elif operation == "convolution":
                # Use smaller kernel size for convolution
                kernel_size = max(8, size // 8)
                cmd = [str(self.gpu_benchmark), "convolution", str(size), str(kernel_size), str(self.config.iterations)]
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=self.bench_root)
            data = json.loads(result.stdout)
            data['platform'] = 'gpu'
            return data
            
        except subprocess.CalledProcessError as e:
            return {"error": f"GPU benchmark failed: {e.stderr}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse GPU benchmark output: {e}"}

class CompetitorBenchmark:
    """Competitor benchmark runner."""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        self.name = name
        self.config = config
        self.bench_root = Path(__file__).parent
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run competitor benchmark."""
        if self.name == "cublas":
            return self._run_cublas_benchmark(operation, size)
        elif self.name == "cupy":
            return self._run_cupy_benchmark(operation, size)
        elif self.name == "cufft":
            return self._run_cufft_benchmark(operation, size)
        else:
            return {"error": f"Unknown competitor: {self.name}"}
    
    def _run_cublas_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run cuBLAS benchmark."""
        cublas_binary = self.bench_root / 'build' / 'cublas_benchmark'
        
        if not cublas_binary.exists():
            return {"error": "cuBLAS benchmark not built"}
        
        try:
            if operation == "matrix_multiply":
                op = "matrix_multiply"
            elif operation == "vector_add":
                op = "vector_add"
            else:
                return {"error": f"Operation {operation} not supported by cuBLAS benchmark"}
            
            cmd = [
                str(cublas_binary),
                '--operation', op,
                '--size1', str(size),
                '--size2', str(size),
                '--size3', str(size),
                '--iterations', str(self.config.iterations)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            data['competitor'] = 'cublas'
            return data
            
        except Exception as e:
            return {"error": f"cuBLAS benchmark failed: {e}"}
    
    def _run_cupy_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run CuPy benchmark (if available)."""
        try:
            import cupy
            # CuPy is available, run benchmark
            return {"note": "CuPy benchmark would run here", "competitor": "cupy"}
        except ImportError:
            return {"error": "CuPy not available", "competitor": "cupy"}
    
    def _run_cufft_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run cuFFT benchmark."""
        cufft_binary = self.bench_root / 'build' / 'cufft_benchmark'
        
        if not cufft_binary.exists():
            return {"error": "cuFFT benchmark not built"}
        
        try:
            if operation == "convolution":
                op = "fft_convolution"
                # Use smaller kernel size for convolution
                size2 = max(8, size // 8)
                cmd = [
                    str(cufft_binary),
                    '--operation', op,
                    '--size1', str(size),
                    '--size2', str(size2),
                    '--iterations', str(self.config.iterations)
                ]
            elif operation == "fft_1d":
                op = "fft_1d"
                cmd = [
                    str(cufft_binary),
                    '--operation', op,
                    '--size1', str(size),
                    '--size2', str(1),
                    '--iterations', str(self.config.iterations)
                ]
            else:
                return {"error": f"Operation {operation} not supported by cuFFT benchmark"}
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            data['competitor'] = 'cufft'
            return data
            
        except Exception as e:
            return {"error": f"cuFFT benchmark failed: {e}"}

class ComprehensiveBenchmarkRunner:
    """Main benchmark orchestrator."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.cpu_benchmark = CPUBenchmark(config) if config.enable_cpu else None
        self.gpu_benchmark = GPUBenchmark(config) if config.enable_gpu else None
        self.competitors = []
        
        if config.enable_competitors:
            self.competitors = [
                CompetitorBenchmark('cublas', config),
                CompetitorBenchmark('cupy', config),
                CompetitorBenchmark('cufft', config),
            ]
        
        self.results = {
            'metadata': self._get_metadata(),
            'config': config.__dict__,
            'cpu_results': [],
            'gpu_results': [],
            'competitor_results': []
        }
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu': SystemInfo.get_cpu_info(),
                'memory': SystemInfo.get_memory_info(),
                'gpu': SystemInfo.get_gpu_info()
            },
            'software': {
                'python_version': sys.version,
                'vsla_version': '1.0.0'
            }
        }
    
    def run_comprehensive_benchmarks(self) -> None:
        """Run complete benchmark suite."""
        print("🚀 Starting VSLA Comprehensive Benchmark Suite")
        print("=" * 60)
        print(f"System: {SystemInfo.get_cpu_info().get('name', 'Unknown')}")
        gpu_info = SystemInfo.get_gpu_info()
        if gpu_info.get('available'):
            print(f"GPU: {gpu_info['name']}")
        print(f"Iterations: {self.config.iterations}")
        print(f"Sizes: {self.config.sizes}")
        print("=" * 60)
        
        # Run CPU benchmarks
        if self.config.enable_cpu and self.cpu_benchmark:
            print("\n📊 CPU BENCHMARKS")
            print("-" * 30)
            
            cpu_operations = ['vector_add', 'convolution', 'kronecker']
            for operation in cpu_operations:
                print(f"\n  {operation}:")
                for size in self.config.sizes:
                    print(f"    Size {size}...", end=" ")
                    result = self.cpu_benchmark.run_benchmark(operation, size)
                    result['size'] = size
                    result['operation'] = operation
                    self.results['cpu_results'].append(result)
                    
                    if 'error' in result:
                        print(f"ERROR: {result['error']}")
                    else:
                        print(f"{result.get('mean_time_us', 0):.1f}μs")
        
        # Run GPU benchmarks
        if self.config.enable_gpu and self.gpu_benchmark:
            print("\n🔥 GPU BENCHMARKS")
            print("-" * 30)
            
            gpu_operations = ['vector_add', 'matrix_multiply', 'convolution']
            for operation in gpu_operations:
                print(f"\n  {operation}:")
                for size in self.config.sizes:
                    print(f"    Size {size}...", end=" ")
                    result = self.gpu_benchmark.run_benchmark(operation, size)
                    result['size'] = size
                    result['operation'] = operation
                    self.results['gpu_results'].append(result)
                    
                    if 'error' in result:
                        print(f"ERROR: {result['error']}")
                    else:
                        print(f"{result.get('mean_time_us', 0):.1f}μs")
        
        # Run competitor benchmarks
        if self.config.enable_competitors:
            print("\n⚔️  COMPETITOR BENCHMARKS")
            print("-" * 30)
            
            for competitor in self.competitors:
                print(f"\n  {competitor.name}:")
                if competitor.name == 'cufft':
                    competitor_operations = ['convolution']
                else:
                    competitor_operations = ['vector_add', 'matrix_multiply']
                for operation in competitor_operations:
                    print(f"    {operation}:")
                    for size in self.config.sizes:
                        print(f"      Size {size}...", end=" ")
                        result = competitor.run_benchmark(operation, size)
                        result['size'] = size
                        result['operation'] = operation
                        self.results['competitor_results'].append(result)
                        
                        if 'error' in result:
                            print(f"ERROR: {result['error']}")
                        else:
                            print(f"{result.get('mean_time_us', 0):.1f}μs")
        
        print("\n" + "=" * 60)
        print("✅ All benchmarks completed!")
    
    def generate_report_filename(self) -> str:
        """Generate report filename with system info and date."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        system_fp = SystemInfo.get_system_fingerprint()
        return f"vsla_benchmark_{system_fp}_{timestamp}"
    
    def save_results(self) -> str:
        """Save results to JSON file."""
        filename = self.generate_report_filename()
        output_path = Path(self.config.output_dir) / f'{filename}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Raw results saved to: {output_path}")
        return str(output_path)
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        report = []
        
        # Header
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cpu_name = self.results['metadata']['system']['cpu']['name']
        gpu_info = self.results['metadata']['system']['gpu']
        gpu_name = gpu_info.get('name', 'N/A') if gpu_info.get('available') else 'N/A'
        
        report.append("# VSLA Comprehensive Benchmark Report")
        report.append("=" * 50)
        report.append("")
        report.append(f"**Generated**: {timestamp}")
        report.append(f"**System**: {cpu_name}")
        report.append(f"**GPU**: {gpu_name}")
        report.append(f"**Memory**: {self.results['metadata']['system']['memory']['total_gb']} GB")
        report.append("")
        
        # Performance Summary Tables
        if self.results['cpu_results']:
            report.append("## CPU Performance Results")
            report.append("")
            report.append("| Operation | Size | Mean Time (μs) | Std Dev (μs) |")
            report.append("|-----------|------|----------------|--------------|")
            
            for result in self.results['cpu_results']:
                if 'error' not in result:
                    op = result.get('operation', 'unknown')
                    size = result.get('size', 0)
                    mean_time = result.get('mean_time_us', 0)
                    std_time = result.get('std_time_us', 0)
                    report.append(f"| {op} | {size} | {mean_time:.2f} | {std_time:.2f} |")
            report.append("")
        
        if self.results['gpu_results']:
            report.append("## GPU Performance Results")
            report.append("")
            report.append("| Operation | Size | Mean Time (μs) | Std Dev (μs) | GFLOPS |")
            report.append("|-----------|------|----------------|--------------|--------|")
            
            for result in self.results['gpu_results']:
                if 'error' not in result:
                    op = result.get('operation', 'unknown')
                    size = result.get('size', 0)
                    mean_time = result.get('mean_time_us', 0)
                    std_time = result.get('std_time_us', 0)
                    
                    # Calculate GFLOPS for matrix operations
                    gflops = 0
                    if op == 'matrix_multiply' and mean_time > 0:
                        flops = 2.0 * size * size * size
                        gflops = flops / (mean_time * 1000)
                    
                    gflops_str = f"{gflops:.0f}" if gflops > 0 else "N/A"
                    report.append(f"| {op} | {size} | {mean_time:.2f} | {std_time:.2f} | {gflops_str} |")
            report.append("")
        
        # Performance Comparison
        if self.results['gpu_results'] and self.results['competitor_results']:
            report.append("## GPU vs Competition Comparison")
            report.append("")
            report.append("| Operation | Size | VSLA GPU (μs) | cuBLAS (μs) | Speedup |")
            report.append("|-----------|------|---------------|-------------|---------|")
            
            for gpu_result in self.results['gpu_results']:
                if 'error' not in gpu_result:
                    size = gpu_result.get('size')
                    operation = gpu_result.get('operation')
                    gpu_time = gpu_result.get('mean_time_us', 0)
                    
                    # Find matching competitor result
                    cublas_result = next((r for r in self.results['competitor_results'] 
                                        if r.get('competitor') == 'cublas' 
                                        and r.get('size') == size 
                                        and r.get('operation') == operation
                                        and 'error' not in r), None)
                    
                    if cublas_result:
                        cublas_time = cublas_result.get('mean_time_us', 0)
                        speedup = cublas_time / gpu_time if gpu_time > 0 else 0
                        speedup_str = f"{speedup:.2f}×" if speedup > 0 else "N/A"
                        
                        report.append(f"| {operation} | {size} | {gpu_time:.2f} | {cublas_time:.2f} | {speedup_str} |")
            report.append("")
        
        # System Configuration
        report.append("## System Configuration")
        report.append("")
        report.append(f"- **CPU**: {cpu_name}")
        report.append(f"- **Cores**: {self.results['metadata']['system']['cpu'].get('cores', 'Unknown')}")
        report.append(f"- **Threads**: {self.results['metadata']['system']['cpu'].get('threads', 'Unknown')}")
        report.append(f"- **Memory**: {self.results['metadata']['system']['memory']['total_gb']} GB")
        if gpu_info.get('available'):
            report.append(f"- **GPU**: {gpu_name}")
            report.append(f"- **GPU Memory**: {gpu_info.get('memory_mb', 0)} MB")
            report.append(f"- **CUDA Compute**: {gpu_info.get('compute_capability', 'Unknown')}")
        report.append("")
        
        # Reproduction Instructions
        report.append("## Reproduction")
        report.append("```bash")
        report.append("python3 run_benchmark.py")
        report.append("```")
        
        return "\\n".join(report)
    
    def save_report(self) -> str:
        """Save performance report."""
        filename = self.generate_report_filename()
        report_content = self.generate_report()
        
        output_path = Path(self.config.output_dir) / f'{filename}.md'
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"📊 Performance report saved to: {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(
        description='VSLA Comprehensive Benchmark Suite - Single Command'
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with smaller sizes and fewer iterations')
    parser.add_argument('--sizes', type=str, default='128,256,512,1024',
                       help='Comma-separated list of test sizes')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations per test')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU benchmarks')
    parser.add_argument('--no-cpu', action='store_true',
                       help='Disable CPU benchmarks')
    parser.add_argument('--no-competitors', action='store_true',
                       help='Disable competitor benchmarks')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.quick:
        sizes = [64, 128, 256]
        iterations = 10
    else:
        sizes = [int(x) for x in args.sizes.split(',')]
        iterations = args.iterations or 50
    
    config = BenchmarkConfig(
        sizes=sizes,
        iterations=iterations,
        warmup=5,
        output_dir='./reports',
        enable_gpu=not args.no_gpu,
        enable_cpu=not args.no_cpu,
        enable_competitors=not args.no_competitors,
        precision='float32',
        reproducible=True
    )
    
    # Run benchmarks
    runner = ComprehensiveBenchmarkRunner(config)
    runner.run_comprehensive_benchmarks()
    
    # Save results and generate report
    json_path = runner.save_results()
    report_path = runner.save_report()
    
    print(f"\\n🎉 Benchmark complete!")
    print(f"   JSON: {json_path}")
    print(f"   Report: {report_path}")

if __name__ == '__main__':
    main()