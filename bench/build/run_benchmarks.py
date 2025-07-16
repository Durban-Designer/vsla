#!/usr/bin/env python3
"""
Master benchmark runner for VSLA performance evaluation
"""

import subprocess
import json
import argparse
import sys
import os
from pathlib import Path
import time
from datetime import datetime

def run_benchmark(executable, args, output_dir):
    """Run a single benchmark executable and capture results"""
    cmd = [executable] + args
    result_file = output_dir / f"{Path(executable).stem}_{int(time.time())}.json"
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run benchmark and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Warning: {executable} returned non-zero exit code: {result.returncode}")
            print(f"Stderr: {result.stderr}")
            return False
        
        # Save results to file
        with open(result_file, 'w') as f:
            f.write(result.stdout)
        
        print(f"Results saved to: {result_file}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Error: {executable} timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"Error running {executable}: {e}")
        return False

def check_build_status(bench_dir):
    """Check if benchmarks are built"""
    build_dir = bench_dir / "build"
    if not build_dir.exists():
        return False
    
    required_executables = [
        "bench_convolution",
        "bench_vector_add", 
        "bench_matvec",
        "bench_kronecker"
    ]
    
    for exe in required_executables:
        if not (build_dir / exe).exists():
            return False
    
    return True

def build_benchmarks(bench_dir):
    """Build the benchmark suite"""
    print("Building benchmark suite...")
    
    build_dir = bench_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Run CMake
    cmake_cmd = ["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."]
    result = subprocess.run(cmake_cmd, cwd=build_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"CMake failed: {result.stderr}")
        return False
    
    # Run make
    make_cmd = ["make", "-j", str(os.cpu_count() or 4)]
    result = subprocess.run(make_cmd, cwd=build_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Make failed: {result.stderr}")
        return False
    
    print("Benchmark suite built successfully")
    return True

def setup_output_directory(base_dir, timestamp=None):
    """Setup output directory for results"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    output_dir = Path(base_dir) / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to latest
    latest_link = Path(base_dir) / "results" / "latest"
    if latest_link.exists() and latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        latest_link.rename(latest_link.with_suffix('.backup'))
    
    try:
        latest_link.symlink_to(timestamp, target_is_directory=True)
    except OSError:
        # Windows might not support symlinks
        pass
    
    return output_dir

def run_convolution_benchmarks(build_dir, output_dir, args):
    """Run convolution benchmarks"""
    print("\n=== Convolution Benchmarks ===")
    
    bench_exe = build_dir / "bench_convolution"
    if not bench_exe.exists():
        print(f"Warning: {bench_exe} not found, skipping convolution benchmarks")
        return
    
    # Default convolution benchmark
    conv_args = [
        "--sizes", "64,128,256,512,1024,2048",
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup)
    ]
    
    return run_benchmark(str(bench_exe), conv_args, output_dir)

def run_vector_add_benchmarks(build_dir, output_dir, args):
    """Run vector addition benchmarks"""
    print("\n=== Vector Addition Benchmarks ===")
    
    bench_exe = build_dir / "bench_vector_add"
    if not bench_exe.exists():
        print(f"Warning: {bench_exe} not found, skipping vector addition benchmarks")
        return False
    
    # Vector addition with different dimension mismatches
    add_args = [
        "--sizes", "64,128,256,512,1024,2048,4096",
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup)
    ]
    
    return run_benchmark(str(bench_exe), add_args, output_dir)

def run_matrix_vector_benchmarks(build_dir, output_dir, args):
    """Run matrix-vector multiplication benchmarks"""
    print("\n=== Matrix-Vector Benchmarks ===")
    
    bench_exe = build_dir / "bench_matvec"
    if not bench_exe.exists():
        print(f"Warning: {bench_exe} not found, skipping matrix-vector benchmarks")
        return False
    
    matvec_args = [
        "--matrices", "64x64,128x128,256x256,512x512",
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup)
    ]
    
    return run_benchmark(str(bench_exe), matvec_args, output_dir)

def run_kronecker_benchmarks(build_dir, output_dir, args):
    """Run Kronecker product benchmarks"""
    print("\n=== Kronecker Product Benchmarks ===")
    
    bench_exe = build_dir / "bench_kronecker"
    if not bench_exe.exists():
        print(f"Warning: {bench_exe} not found, skipping Kronecker benchmarks")
        return False
    
    kron_args = [
        "--dimensions", "32,64,128,256,512",
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup)
    ]
    
    return run_benchmark(str(bench_exe), kron_args, output_dir)

def generate_summary_report(output_dir):
    """Generate a summary report of all benchmarks"""
    print("\n=== Generating Summary Report ===")
    
    # Find all JSON result files
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print("No benchmark results found")
        return
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_benchmarks": len(json_files),
        "benchmarks": []
    }
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'benchmark' in item:
                            summary["benchmarks"].append({
                                "file": json_file.name,
                                "benchmark": item.get('benchmark'),
                                "method": item.get('method'),
                                "mean_time_us": item.get('results', {}).get('mean_time_us')
                            })
                elif isinstance(data, dict) and 'benchmark' in data:
                    summary["benchmarks"].append({
                        "file": json_file.name,
                        "benchmark": data.get('benchmark'),
                        "method": data.get('method'),
                        "mean_time_us": data.get('results', {}).get('mean_time_us')
                    })
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")
    
    # Write summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary report written to: {summary_file}")
    print(f"Total benchmarks run: {len(summary['benchmarks'])}")

def main():
    parser = argparse.ArgumentParser(description='Run VSLA benchmark suite')
    parser.add_argument('--output', '-o', 
                       help='Output directory (default: results/TIMESTAMP)')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Number of iterations per benchmark (default: 100)')
    parser.add_argument('--warmup', '-w', type=int, default=5,
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('--build', '-b', action='store_true',
                       help='Force rebuild of benchmarks')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip build check (assume benchmarks are built)')
    parser.add_argument('--benchmarks', nargs='+', 
                       choices=['convolution', 'vector_add', 'matvec', 'kronecker', 'all'],
                       default=['all'],
                       help='Which benchmarks to run (default: all)')
    parser.add_argument('--reproducible', action='store_true',
                       help='Set environment for reproducible results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    
    args = parser.parse_args()
    
    # Find benchmark directory
    bench_dir = Path(__file__).parent
    build_dir = bench_dir / "build"
    
    # Setup reproducible environment
    if args.reproducible:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        print("Set environment for reproducible results")
    
    # Check/build benchmarks
    if args.build or (not args.skip_build and not check_build_status(bench_dir)):
        if not build_benchmarks(bench_dir):
            print("Failed to build benchmarks")
            return 1
    
    # Setup output directory
    output_dir = setup_output_directory(bench_dir, args.output)
    print(f"Results will be saved to: {output_dir}")
    
    # Write benchmark configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "benchmarks": args.benchmarks,
        "reproducible": args.reproducible,
        "seed": args.seed if args.reproducible else None
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run selected benchmarks
    success_count = 0
    total_count = 0
    
    if 'all' in args.benchmarks or 'convolution' in args.benchmarks:
        total_count += 1
        if run_convolution_benchmarks(build_dir, output_dir, args):
            success_count += 1
    
    if 'all' in args.benchmarks or 'vector_add' in args.benchmarks:
        total_count += 1  
        if run_vector_add_benchmarks(build_dir, output_dir, args):
            success_count += 1
    
    if 'all' in args.benchmarks or 'matvec' in args.benchmarks:
        total_count += 1
        if run_matrix_vector_benchmarks(build_dir, output_dir, args):
            success_count += 1
    
    if 'all' in args.benchmarks or 'kronecker' in args.benchmarks:
        total_count += 1
        if run_kronecker_benchmarks(build_dir, output_dir, args):
            success_count += 1
    
    # Generate summary
    generate_summary_report(output_dir)
    
    print(f"\n=== Benchmark Summary ===")
    print(f"Successful: {success_count}/{total_count}")
    print(f"Results directory: {output_dir}")
    
    # Generate Table 2 if we have results
    table2_script = bench_dir / "scripts" / "generate_table2.py"
    if table2_script.exists():
        print("\nGenerating Table 2...")
        try:
            subprocess.run([sys.executable, str(table2_script), 
                          "--input", str(output_dir),
                          "--output", str(output_dir / "table2.tex")],
                          check=True)
            print(f"Table 2 saved to: {output_dir / 'table2.tex'}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate Table 2: {e}")
    
    return 0 if success_count == total_count else 1

if __name__ == '__main__':
    sys.exit(main())