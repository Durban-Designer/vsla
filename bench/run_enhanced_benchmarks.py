#!/usr/bin/env python3
"""
Enhanced VSLA Benchmark Runner
Comprehensive real-world performance validation for VSLA library
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
import platform
import argparse

def get_system_info():
    """Collect system information for reproducibility"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
    }
    
    # Try to get more detailed CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_lines = f.readlines()
            for line in cpu_lines:
                if line.startswith('model name'):
                    info['cpu_model'] = line.split(':')[1].strip()
                    break
    except:
        pass
    
    # Get memory info
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_lines = f.readlines()
            for line in mem_lines:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    info['memory_gb'] = round(mem_kb / (1024 * 1024), 1)
                    break
    except:
        pass
    
    return info

def build_benchmarks(build_dir, source_dir):
    """Build the benchmark executables"""
    print("Building VSLA library and benchmarks...")
    
    # First build VSLA library
    vsla_build_dir = os.path.join(source_dir, '..', 'build')
    if not os.path.exists(vsla_build_dir):
        os.makedirs(vsla_build_dir)
    
    # Build VSLA
    os.chdir(vsla_build_dir)
    subprocess.run(['cmake', '-DCMAKE_BUILD_TYPE=Release', '..'], check=True)
    subprocess.run(['make', '-j', str(os.cpu_count())], check=True)
    
    # Build benchmarks
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    os.chdir(build_dir)
    subprocess.run(['cmake', '-DCMAKE_BUILD_TYPE=Release', source_dir], check=True)
    subprocess.run(['make', '-j', str(os.cpu_count())], check=True)
    
    print("Build completed successfully!")

def run_enhanced_benchmark(build_dir, vsla_lib_dir, output_dir):
    """Run the enhanced benchmark with proper library paths"""
    print("Running enhanced benchmarks...")
    
    # Set up environment
    env = os.environ.copy()
    
    # Add VSLA library to LD_LIBRARY_PATH
    ld_path = env.get('LD_LIBRARY_PATH', '')
    if ld_path:
        env['LD_LIBRARY_PATH'] = f"{vsla_lib_dir}:{ld_path}"
    else:
        env['LD_LIBRARY_PATH'] = vsla_lib_dir
    
    # Change to build directory
    os.chdir(build_dir)
    
    # Run the enhanced benchmark
    benchmark_exe = os.path.join(build_dir, 'enhanced_benchmark')
    if not os.path.exists(benchmark_exe):
        raise FileNotFoundError(f"Benchmark executable not found: {benchmark_exe}")
    
    try:
        result = subprocess.run([benchmark_exe], 
                              env=env, 
                              cwd=build_dir,
                              capture_output=True, 
                              text=True, 
                              timeout=600)  # 10 minute timeout
        
        print("Benchmark output:")
        print(result.stdout)
        
        if result.stderr:
            print("Benchmark errors:")
            print(result.stderr)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, benchmark_exe)
        
    except subprocess.TimeoutExpired:
        print("Benchmark timed out after 10 minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with return code {e.returncode}")
        return False
    
    return True

def analyze_results(results_dir):
    """Analyze and summarize benchmark results"""
    print(f"Analyzing results in {results_dir}...")
    
    # Find all JSON result files
    json_files = []
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            json_files.append(os.path.join(results_dir, file))
    
    if not json_files:
        print("No JSON result files found")
        return
    
    # Create summary
    summary = {
        'total_benchmarks': len(json_files),
        'benchmark_files': json_files,
        'system_info': get_system_info()
    }
    
    # Process each result file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            filename = os.path.basename(json_file)
            print(f"\nSummary for {filename}:")
            
            if 'benchmark_type' in data:
                print(f"  Type: {data['benchmark_type']}")
            
            if 'results' in data:
                results = data['results']
                if isinstance(results, list) and results:
                    # Calculate aggregate statistics
                    times = [r.get('mean_time_us', 0) for r in results if isinstance(r, dict)]
                    if times:
                        avg_time = sum(times) / len(times)
                        print(f"  Average execution time: {avg_time:.3f} μs")
                        print(f"  Total test cases: {len(times)}")
                elif isinstance(results, dict):
                    if 'mean_time_us' in results:
                        print(f"  Mean time: {results['mean_time_us']:.3f} μs")
                    if 'throughput_mops' in results:
                        print(f"  Throughput: {results['throughput_mops']:.3f} MOPS")
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Save summary
    summary_file = os.path.join(results_dir, 'benchmark_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBenchmark summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Run VSLA enhanced benchmarks')
    parser.add_argument('--build', action='store_true', 
                       help='Build benchmarks before running')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for results')
    parser.add_argument('--analyze-only', type=str,
                       help='Only analyze results from specified directory')
    
    args = parser.parse_args()
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = script_dir  # bench directory
    build_dir = os.path.join(source_dir, 'build')
    vsla_lib_dir = os.path.join(source_dir, '..', 'build')  # where libvsla.so is built
    
    if args.analyze_only:
        analyze_results(args.analyze_only)
        return
    
    print("VSLA Enhanced Benchmark Runner")
    print("=" * 40)
    print(f"Source directory: {source_dir}")
    print(f"Build directory: {build_dir}")
    print(f"VSLA library directory: {vsla_lib_dir}")
    
    try:
        # Build if requested
        if args.build or not os.path.exists(build_dir):
            build_benchmarks(build_dir, source_dir)
        
        # Create output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"enhanced_benchmark_results_{timestamp}"
        
        # Run benchmarks
        success = run_enhanced_benchmark(build_dir, vsla_lib_dir, output_dir)
        
        if success:
            print("\nBenchmark completed successfully!")
            
            # Find the actual results directory (created by the C program)
            # Look for directories matching the pattern
            potential_dirs = []
            for item in os.listdir(build_dir):
                if item.startswith('enhanced_benchmark_results_') and os.path.isdir(os.path.join(build_dir, item)):
                    potential_dirs.append(os.path.join(build_dir, item))
            
            if potential_dirs:
                # Use the most recent directory
                results_dir = max(potential_dirs, key=os.path.getctime)
                analyze_results(results_dir)
            else:
                print("Warning: Could not find results directory for analysis")
        else:
            print("Benchmark failed!")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())