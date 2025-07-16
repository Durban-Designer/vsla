#!/usr/bin/env python3
"""
VSLA Comprehensive Benchmark Suite
Run complete benchmarks against top 3 competitors and generate final report.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
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
    competitors: List[str]
    enable_gpu: bool
    precision: str  # 'float32' or 'float64'
    reproducible: bool

class SystemInfo:
    """Gather system information for reproducibility."""
    
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

class CompetitorBenchmark:
    """Base class for competitor benchmarks."""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        self.name = name
        self.config = config
        self.results = []
    
    def check_availability(self) -> bool:
        """Check if competitor is available on system."""
        raise NotImplementedError
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run specific benchmark operation."""
        raise NotImplementedError
    
    def setup_environment(self) -> None:
        """Setup environment for reproducible benchmarks."""
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        if not self.config.enable_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

class CupyBenchmark(CompetitorBenchmark):
    """CuPy benchmark implementation."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__('cupy', config)
    
    def check_availability(self) -> bool:
        """Check if CuPy is available."""
        try:
            import cupy
            return True
        except ImportError:
            return False
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run CuPy benchmark."""
        if not self.check_availability():
            return {'error': 'CuPy not available'}
        
        script_path = Path(__file__).parent / 'competitors' / 'cupy_benchmark.py'
        if not script_path.exists():
            return {'error': 'CuPy benchmark script not found'}
        
        # Map operation names to CuPy benchmark operations
        op_map = {
            'vector_add': 'vector_add',
            'matrix_multiply': 'matrix_multiply',
            'convolution': 'convolution'
        }
        
        if operation not in op_map:
            return {'error': f'Unknown operation: {operation}'}
        
        cmd = [
            'python3', str(script_path),
            '--operation', op_map[operation],
            '--size1', str(size),
            '--size2', str(size // 4),  # Smaller second dimension
            '--iterations', str(self.config.iterations)
        ]
        
        if not self.config.enable_gpu:
            cmd.extend(['--device', '-1'])  # Force CPU
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            data['competitor'] = 'cupy'
            return data
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            return {'error': f'CuPy benchmark failed: {e}'}

class CublasBenchmark(CompetitorBenchmark):
    """cuBLAS benchmark implementation."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__('cublas', config)
    
    def check_availability(self) -> bool:
        """Check if cuBLAS is available."""
        try:
            # Check for CUDA toolkit
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run cuBLAS benchmark."""
        if not self.check_availability():
            return {'error': 'cuBLAS not available'}
        
        binary_path = Path(__file__).parent / 'competitors' / 'cublas_benchmark'
        if not binary_path.exists():
            return {'error': 'cuBLAS benchmark binary not found'}
        
        # Map operation names to cuBLAS benchmark operations
        op_map = {
            'vector_add': 'vector_add',
            'matrix_multiply': 'matrix_multiply'
        }
        
        if operation not in op_map:
            return {'error': f'Unknown operation: {operation}'}
        
        cmd = [
            str(binary_path),
            '--operation', op_map[operation],
            '--size1', str(size),
            '--size2', str(size),
            '--size3', str(size),
            '--iterations', str(self.config.iterations)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            data['competitor'] = 'cublas'
            return data
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            return {'error': f'cuBLAS benchmark failed: {e}'}

class CufftBenchmark(CompetitorBenchmark):
    """cuFFT benchmark implementation."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__('cufft', config)
    
    def check_availability(self) -> bool:
        """Check if cuFFT is available."""
        # Same as cuBLAS - part of CUDA toolkit
        return CublasBenchmark(self.config).check_availability()
    
    def run_benchmark(self, operation: str, size: int) -> Dict[str, Any]:
        """Run cuFFT benchmark."""
        if not self.check_availability():
            return {'error': 'cuFFT not available'}
        
        binary_path = Path(__file__).parent / 'competitors' / 'cufft_benchmark'
        if not binary_path.exists():
            return {'error': 'cuFFT benchmark binary not found'}
        
        # Map operation names to cuFFT benchmark operations
        op_map = {
            'convolution': 'fft_convolution',
            'fft_1d': 'fft_1d'
        }
        
        if operation not in op_map:
            return {'error': f'Unknown operation: {operation}'}
        
        cmd = [
            str(binary_path),
            '--operation', op_map[operation],
            '--size1', str(size),
            '--size2', str(size // 4),  # Smaller kernel
            '--iterations', str(self.config.iterations)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            data['competitor'] = 'cufft'
            return data
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            return {'error': f'cuFFT benchmark failed: {e}'}

class VSLABenchmark:
    """VSLA benchmark runner."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.build_dir = Path(__file__).parent / 'build'
    
    def ensure_built(self) -> bool:
        """Ensure VSLA benchmarks are built."""
        if not self.build_dir.exists():
            print("Build directory not found. Please run: cd bench && mkdir build && cd build && cmake .. && make")
            return False
        
        required_binaries = ['bench_comparison', 'bench_convolution']
        for binary in required_binaries:
            if not (self.build_dir / binary).exists():
                print(f"Binary {binary} not found. Please rebuild benchmarks.")
                return False
        
        return True
    
    def run_comparison_benchmark(self, sizes: List[int]) -> List[Dict[str, Any]]:
        """Run VSLA comparison benchmark."""
        if not self.ensure_built():
            return []
        
        cmd = [
            str(self.build_dir / 'bench_comparison'),
            '--sizes', ','.join(map(str, sizes)),
            '--iterations', str(self.config.iterations),
            '--warmup', str(self.config.warmup)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse JSON output
            return self._parse_json_output(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"VSLA comparison benchmark failed: {e}")
            return []
    
    def run_convolution_benchmark(self, sizes: List[int]) -> List[Dict[str, Any]]:
        """Run VSLA convolution benchmark."""
        if not self.ensure_built():
            return []
        
        cmd = [
            str(self.build_dir / 'bench_convolution'),
            '--sizes', ','.join(map(str, sizes)),
            '--iterations', str(self.config.iterations),
            '--warmup', str(self.config.warmup)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return self._parse_json_output(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"VSLA convolution benchmark failed: {e}")
            return []
    
    def _parse_json_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse JSON output from benchmark binaries."""
        try:
            # Remove comments and parse JSON
            lines = output.split('\n')
            json_lines = [line for line in lines if not line.strip().startswith('//')]
            json_str = '\n'.join(json_lines)
            
            # Handle multiple JSON objects
            results = []
            objects = json_str.strip().split('},{')
            for i, obj in enumerate(objects):
                if i > 0:
                    obj = '{' + obj
                if i < len(objects) - 1:
                    obj = obj + '}'
                
                try:
                    result = json.loads(obj)
                    results.append(result)
                except json.JSONDecodeError:
                    continue
            
            return results
        except Exception as e:
            print(f"Failed to parse JSON output: {e}")
            return []

class BenchmarkRunner:
    """Main benchmark runner orchestrating all competitors."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.vsla = VSLABenchmark(config)
        self.competitors = self._init_competitors()
        self.results = {
            'metadata': self._get_metadata(),
            'config': config.__dict__,
            'vsla': [],
            'competitors': []
        }
    
    def _init_competitors(self) -> List[CompetitorBenchmark]:
        """Initialize competitor benchmarks."""
        competitors = []
        
        if 'cupy' in self.config.competitors:
            competitors.append(CupyBenchmark(self.config))
        if 'cublas' in self.config.competitors:
            competitors.append(CublasBenchmark(self.config))
        if 'cufft' in self.config.competitors:
            competitors.append(CufftBenchmark(self.config))
        
        return competitors
    
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
                'vsla_version': '1.0.0'  # TODO: Get from library
            }
        }
    
    def setup_environment(self) -> None:
        """Setup reproducible environment."""
        print("Setting up reproducible environment...")
        
        # Set environment variables
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        if not self.config.enable_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("GPU acceleration disabled")
        else:
            print("GPU acceleration enabled")
        
        # Set CPU governor for consistent performance
        if self.config.reproducible:
            try:
                subprocess.run(['sudo', 'cpupower', 'frequency-set', '--governor', 'performance'], 
                             check=True, capture_output=True)
                print("CPU governor set to performance mode")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: Could not set CPU governor to performance mode")
    
    def run_vsla_benchmarks(self) -> None:
        """Run all VSLA benchmarks."""
        print("Running VSLA benchmarks...")
        
        # Run comparison benchmarks
        comparison_results = self.vsla.run_comparison_benchmark(self.config.sizes)
        self.results['vsla'].extend(comparison_results)
        
        # Run convolution benchmarks
        convolution_results = self.vsla.run_convolution_benchmark(self.config.sizes)
        self.results['vsla'].extend(convolution_results)
        
        print(f"Completed {len(self.results['vsla'])} VSLA benchmarks")
    
    def run_competitor_benchmarks(self) -> None:
        """Run all competitor benchmarks."""
        print("Running competitor benchmarks...")
        
        for competitor in self.competitors:
            print(f"Running {competitor.name} benchmarks...")
            
            if not competitor.check_availability():
                print(f"Warning: {competitor.name} not available, skipping...")
                continue
            
            competitor.setup_environment()
            
            # Run benchmarks for each operation and size
            operations = ['vector_add', 'matrix_multiply', 'convolution']
            for operation in operations:
                for size in self.config.sizes:
                    try:
                        result = competitor.run_benchmark(operation, size)
                        self.results['competitors'].append(result)
                    except Exception as e:
                        print(f"Error running {competitor.name} {operation} benchmark: {e}")
        
        print(f"Completed {len(self.results['competitors'])} competitor benchmarks")
    
    def save_results(self) -> None:
        """Save benchmark results to file."""
        output_path = Path(self.config.output_dir) / 'benchmark_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def run_all(self) -> None:
        """Run complete benchmark suite."""
        print("Starting VSLA Comprehensive Benchmark Suite")
        print("=" * 50)
        
        self.setup_environment()
        self.run_vsla_benchmarks()
        self.run_competitor_benchmarks()
        self.save_results()
        
        print("=" * 50)
        print("Benchmark suite completed successfully!")

class ReportGenerator:
    """Generate comprehensive benchmark report."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        
        # Header
        report.append("# VSLA Comprehensive Benchmark Report")
        report.append("")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**System**: {self.results['metadata']['system']['cpu']['name']}")
        report.append(f"**GPU**: {self.results['metadata']['system']['gpu'].get('name', 'N/A')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(self._generate_executive_summary())
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        report.append(self._generate_detailed_results())
        report.append("")
        
        # Competitor Analysis
        report.append("## Competitor Analysis")
        report.append("")
        report.append(self._generate_competitor_analysis())
        report.append("")
        
        # Reproducibility Information
        report.append("## Reproducibility Information")
        report.append("")
        report.append(self._generate_reproducibility_info())
        report.append("")
        
        return "\n".join(report)
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary."""
        vsla_results = self.results.get('vsla', [])
        competitor_results = self.results.get('competitors', [])
        
        if not vsla_results:
            return "No VSLA benchmark results available."
        
        # Analyze VSLA performance
        total_vsla_tests = len(vsla_results)
        avg_vsla_time = sum(r.get('mean_time_us', 0) for r in vsla_results) / max(1, total_vsla_tests)
        
        # Analyze competitor performance
        total_competitor_tests = len(competitor_results)
        competitors_available = len(set(r.get('competitor', 'unknown') for r in competitor_results))
        
        summary = []
        summary.append(f"**VSLA Performance**: {total_vsla_tests} tests completed with average execution time of {avg_vsla_time:.1f}μs")
        summary.append(f"**Competitor Analysis**: {total_competitor_tests} tests across {competitors_available} competitors")
        
        # Calculate speedup analysis if we have both VSLA and competitor data
        if vsla_results and competitor_results:
            # Find matching operations for comparison
            vsla_ops = {(r.get('operation'), r.get('size')): r for r in vsla_results}
            competitor_ops = {(r.get('operation'), r.get('size')): r for r in competitor_results}
            
            speedups = []
            for key in vsla_ops:
                if key in competitor_ops:
                    vsla_time = vsla_ops[key].get('mean_time_us', 0)
                    comp_time = competitor_ops[key].get('mean_time_us', 0)
                    if comp_time > 0:
                        speedup = comp_time / vsla_time
                        speedups.append(speedup)
            
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                max_speedup = max(speedups)
                min_speedup = min(speedups)
                summary.append(f"**Performance Range**: {min_speedup:.1f}× to {max_speedup:.1f}× (avg: {avg_speedup:.1f}×)")
        
        return "\n".join(summary)
    
    def _generate_detailed_results(self) -> str:
        """Generate detailed results section."""
        results = []
        
        # VSLA Results
        vsla_results = self.results.get('vsla', [])
        if vsla_results:
            results.append("### VSLA Performance Results")
            results.append("")
            results.append("| Operation | Size | Mean Time (μs) | Std Dev (μs) | Memory (MB) |")
            results.append("|-----------|------|----------------|--------------|-------------|")
            
            for result in vsla_results:
                operation = result.get('operation', 'unknown')
                size = result.get('size', 0)
                mean_time = result.get('mean_time_us', 0)
                std_time = result.get('std_time_us', 0)
                memory = result.get('memory_mb', 0)
                
                results.append(f"| {operation} | {size} | {mean_time:.2f} | {std_time:.2f} | {memory:.1f} |")
            
            results.append("")
        
        # Competitor Results
        competitor_results = self.results.get('competitors', [])
        if competitor_results:
            results.append("### Competitor Performance Results")
            results.append("")
            results.append("| Competitor | Operation | Size | Mean Time (μs) | Std Dev (μs) | Memory (MB) |")
            results.append("|------------|-----------|------|----------------|--------------|-------------|")
            
            for result in competitor_results:
                competitor = result.get('competitor', 'unknown')
                operation = result.get('operation', 'unknown')
                size = result.get('size', 0)
                mean_time = result.get('mean_time_us', 0)
                std_time = result.get('std_time_us', 0)
                memory = result.get('memory_mb', 0)
                
                if 'error' in result:
                    results.append(f"| {competitor} | {operation} | {size} | ERROR | - | - |")
                else:
                    results.append(f"| {competitor} | {operation} | {size} | {mean_time:.2f} | {std_time:.2f} | {memory:.1f} |")
            
            results.append("")
        
        return "\n".join(results)
    
    def _generate_competitor_analysis(self) -> str:
        """Generate competitor analysis."""
        vsla_results = self.results.get('vsla', [])
        competitor_results = self.results.get('competitors', [])
        
        if not vsla_results or not competitor_results:
            return "Insufficient data for competitive analysis."
        
        analysis = []
        analysis.append("### Performance Comparison")
        analysis.append("")
        
        # Group results by operation and size
        vsla_by_op = {}
        competitor_by_op = {}
        
        for result in vsla_results:
            key = (result.get('operation'), result.get('size'))
            vsla_by_op[key] = result
        
        for result in competitor_results:
            if 'error' not in result:
                key = (result.get('operation'), result.get('size'))
                competitor = result.get('competitor')
                if key not in competitor_by_op:
                    competitor_by_op[key] = {}
                competitor_by_op[key][competitor] = result
        
        # Generate comparison table
        analysis.append("| Operation | Size | VSLA (μs) | CuPy (μs) | cuBLAS (μs) | cuFFT (μs) | Best Speedup |")
        analysis.append("|-----------|------|-----------|-----------|-------------|------------|--------------|")
        
        for key in sorted(vsla_by_op.keys()):
            operation, size = key
            vsla_time = vsla_by_op[key].get('mean_time_us', 0)
            
            competitors = competitor_by_op.get(key, {})
            cupy_time = competitors.get('cupy', {}).get('mean_time_us', 0)
            cublas_time = competitors.get('cublas', {}).get('mean_time_us', 0)
            cufft_time = competitors.get('cufft', {}).get('mean_time_us', 0)
            
            # Calculate best speedup
            competitor_times = [t for t in [cupy_time, cublas_time, cufft_time] if t > 0]
            if competitor_times and vsla_time > 0:
                best_speedup = max(competitor_times) / vsla_time
                speedup_str = f"{best_speedup:.1f}×"
            else:
                speedup_str = "N/A"
            
            analysis.append(f"| {operation} | {size} | {vsla_time:.2f} | "
                          f"{cupy_time:.2f} | {cublas_time:.2f} | {cufft_time:.2f} | {speedup_str} |")
        
        analysis.append("")
        
        # Summary analysis
        analysis.append("### Key Insights")
        analysis.append("")
        
        # Calculate overall statistics
        speedups = []
        for key in vsla_by_op:
            vsla_time = vsla_by_op[key].get('mean_time_us', 0)
            competitors = competitor_by_op.get(key, {})
            
            for comp_name, comp_result in competitors.items():
                comp_time = comp_result.get('mean_time_us', 0)
                if comp_time > 0 and vsla_time > 0:
                    speedup = comp_time / vsla_time
                    speedups.append((comp_name, speedup))
        
        if speedups:
            avg_speedup = sum(s[1] for s in speedups) / len(speedups)
            max_speedup = max(speedups, key=lambda x: x[1])
            min_speedup = min(speedups, key=lambda x: x[1])
            
            analysis.append(f"- **Average Performance**: {avg_speedup:.1f}× faster than competitors")
            analysis.append(f"- **Best Performance**: {max_speedup[1]:.1f}× faster than {max_speedup[0]}")
            analysis.append(f"- **Worst Performance**: {min_speedup[1]:.1f}× vs {min_speedup[0]}")
        
        return "\n".join(analysis)
    
    def _generate_reproducibility_info(self) -> str:
        """Generate reproducibility information."""
        config = self.results['config']
        system = self.results['metadata']['system']
        
        info = []
        info.append("### System Configuration")
        info.append(f"- **CPU**: {system['cpu']['name']}")
        info.append(f"- **Memory**: {system['memory']['total_gb']} GB")
        info.append(f"- **GPU**: {system['gpu'].get('name', 'N/A')}")
        info.append("")
        
        info.append("### Benchmark Configuration")
        info.append(f"- **Test Sizes**: {config['sizes']}")
        info.append(f"- **Iterations**: {config['iterations']}")
        info.append(f"- **Warmup**: {config['warmup']}")
        info.append(f"- **Precision**: {config['precision']}")
        info.append(f"- **GPU Enabled**: {config['enable_gpu']}")
        info.append("")
        
        info.append("### Reproduction Instructions")
        info.append("```bash")
        info.append("# Install dependencies")
        info.append("pip install cupy-cuda12x  # or appropriate CUDA version")
        info.append("")
        info.append("# Run benchmark")
        info.append("cd bench")
        info.append("python run_full_benchmark.py --reproducible")
        info.append("```")
        
        return "\n".join(info)
    
    def save_report(self, output_path: str) -> None:
        """Save report to file."""
        report_content = self.generate_report()
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive VSLA benchmarks against top competitors'
    )
    
    parser.add_argument('--sizes', type=str, default='256,512,1024,2048',
                       help='Comma-separated list of test sizes')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per test')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--competitors', type=str, default='cupy,cublas,cufft',
                       help='Comma-separated list of competitors to benchmark')
    parser.add_argument('--enable-gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--precision', type=str, default='float64',
                       choices=['float32', 'float64'],
                       help='Floating point precision')
    parser.add_argument('--reproducible', action='store_true',
                       help='Enable reproducible benchmarking mode')
    parser.add_argument('--report-only', type=str,
                       help='Generate report from existing results file')
    
    args = parser.parse_args()
    
    if args.report_only:
        # Generate report from existing results
        generator = ReportGenerator(args.report_only)
        output_path = Path(args.report_only).parent / 'benchmark_report.md'
        generator.save_report(str(output_path))
        return
    
    # Configure benchmark
    config = BenchmarkConfig(
        sizes=[int(x) for x in args.sizes.split(',')],
        iterations=args.iterations,
        warmup=args.warmup,
        output_dir=args.output_dir,
        competitors=args.competitors.split(','),
        enable_gpu=args.enable_gpu,
        precision=args.precision,
        reproducible=args.reproducible
    )
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    runner.run_all()
    
    # Generate report
    results_path = Path(config.output_dir) / 'benchmark_results.json'
    generator = ReportGenerator(str(results_path))
    report_path = Path(config.output_dir) / 'benchmark_report.md'
    generator.save_report(str(report_path))

if __name__ == '__main__':
    main()