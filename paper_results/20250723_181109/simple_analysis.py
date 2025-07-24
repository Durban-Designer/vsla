#!/usr/bin/env python3

import os
import sys
import glob
import csv
from collections import defaultdict
import statistics

def analyze_csv_files(session_dir):
    """Simple analysis of benchmark CSV files without pandas."""
    
    results = {}
    parsed_dir = os.path.join(session_dir, "parsed_data")
    
    # Vector addition analysis
    vector_files = glob.glob(os.path.join(parsed_dir, "vector_add_run_*.csv"))
    if vector_files:
        vector_data = defaultdict(list)
        for file in vector_files:
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config = row['Configuration']
                    try:
                        vsla_time = float(row['VSLA_ms'])
                        manual_time = float(row['Manual_ms'])
                        ratio = float(row['Ratio']) if row['Ratio'] != '0.00' else (vsla_time / manual_time if manual_time > 0 else float('inf'))
                        
                        vector_data[config].append({
                            'vsla_time': vsla_time,
                            'manual_time': manual_time,
                            'ratio': ratio
                        })
                    except (ValueError, ZeroDivisionError):
                        continue
        
        results['vector_addition'] = vector_data
    
    # FFT analysis
    fft_files = glob.glob(os.path.join(parsed_dir, "fft_run_*.csv"))
    if fft_files:
        fft_data = defaultdict(list)
        for file in fft_files:
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Size' in row and 'Time_ms' in row:
                        try:
                            size = int(row['Size'])
                            time_ms = float(row['Time_ms'])
                            ops_per_ms = float(row['Ops_per_ms']) if 'Ops_per_ms' in row else 0
                            
                            fft_data[size].append({
                                'time_ms': time_ms,
                                'ops_per_ms': ops_per_ms
                            })
                        except (ValueError, KeyError):
                            continue
        
        results['fft_convolution'] = fft_data
    
    # Kronecker analysis
    kron_files = glob.glob(os.path.join(parsed_dir, "kronecker_run_*.csv"))
    if kron_files:
        kron_data = defaultdict(list)
        for file in kron_files:
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        config = f"{row['Size_A']}x{row['Size_B']}"
                        vsla_time = float(row['VSLA_ms'])
                        ratio_manual = float(row['Ratio_Manual'])
                        ratio_simple = float(row['Ratio_Simple'])
                        
                        kron_data[config].append({
                            'vsla_time': vsla_time,
                            'ratio_manual': ratio_manual,
                            'ratio_simple': ratio_simple
                        })
                    except (ValueError, KeyError):
                        continue
        
        results['kronecker_product'] = kron_data
    
    return results

def compute_statistics(data_list, key):
    """Compute basic statistics for a list of data."""
    values = [item[key] for item in data_list if key in item]
    if not values:
        return None
    
    return {
        'mean': statistics.mean(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

def generate_summary_report(results, session_dir):
    """Generate comprehensive summary report."""
    
    report_path = os.path.join(session_dir, "PAPER_RESULTS_SUMMARY.txt")
    
    with open(report_path, 'w') as f:
        f.write("VSLA PERFORMANCE BENCHMARKS - PAPER RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Statistical Summary (10 runs per benchmark)\n")
        f.write("Format: mean ± std (min, max)\n\n")
        
        # Vector Addition Results
        if 'vector_addition' in results:
            f.write("VECTOR ADDITION PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            for config, data_list in results['vector_addition'].items():
                if not data_list:
                    continue
                    
                vsla_stats = compute_statistics(data_list, 'vsla_time')
                ratio_stats = compute_statistics(data_list, 'ratio')
                
                if vsla_stats and ratio_stats:
                    f.write(f"{config:15s}: {vsla_stats['mean']:.3f}±{vsla_stats['stdev']:.3f} ms ")
                    f.write(f"({vsla_stats['min']:.3f}, {vsla_stats['max']:.3f}) ")
                    f.write(f"Ratio: {ratio_stats['mean']:.2f}±{ratio_stats['stdev']:.2f}\n")
            f.write("\n")
        
        # FFT Results
        if 'fft_convolution' in results:
            f.write("FFT CONVOLUTION PERFORMANCE
")
            f.write("-" * 30 + "
")
            
            for size in sorted(results['fft_convolution'].keys()):
                data_list = results['fft_convolution'][size]
                if not data_list:
                    continue
                    
                time_stats = compute_statistics(data_list, 'time_ms')
                ops_stats = compute_statistics(data_list, 'ops_per_ms')
                
                if time_stats:
                    f.write(f"Size {size:4d}: {time_stats['mean']:.3f}±{time_stats['stdev']:.3f} ms")
                    if ops_stats:
                        f.write(f", {ops_stats['mean']:.0f}±{ops_stats['stdev']:.0f} ops/ms")
                    f.write("
")
            f.write("
")
        
        # Kronecker Results
        if 'kronecker_product' in results:
            f.write("KRONECKER PRODUCT PERFORMANCE
")
            f.write("-" * 30 + "
")
            
            for config in sorted(results['kronecker_product'].keys()):
                data_list = results['kronecker_product'][config]
                if not data_list:
                    continue
                    
                manual_stats = compute_statistics(data_list, 'ratio_manual')
                simple_stats = compute_statistics(data_list, 'ratio_simple')
                
                if manual_stats and simple_stats:
                    f.write(f"{config:8s}: Manual {manual_stats['mean']:.2f}±{manual_stats['stdev']:.2f}, ")
                    f.write(f"Simple {simple_stats['mean']:.2f}±{simple_stats['stdev']:.2f}
")
            f.write("
")
        
        # Performance Analysis Summary
        f.write("PERFORMANCE ANALYSIS SUMMARY
")
        f.write("-" * 30 + "
")
        
        if 'vector_addition' in results:
            excellent = []
            competitive = []
            needs_work = []
            
            for config, data_list in results['vector_addition'].items():
                if not data_list:
                    continue
                    
                ratio_stats = compute_statistics(data_list, 'ratio')
                if ratio_stats:
                    ratio_mean = ratio_stats['mean']
                    if ratio_mean < 1.1:
                        excellent.append(f"{config} ({ratio_mean:.2f}x)")
                    elif ratio_mean < 2.0:
                        competitive.append(f"{config} ({ratio_mean:.2f}x)")
                    else:
                        needs_work.append(f"{config} ({ratio_mean:.2f}x)")
            
            if excellent:
                f.write("VSLA Excels (≤1.1x):
")
                for item in excellent:
                    f.write(f"  ✓ {item}
")
                f.write("
")
            
            if competitive:
                f.write("VSLA Competitive (1.1-2.0x):
")
                for item in competitive:
                    f.write(f"  ~ {item}
")
                f.write("
")
            
            if needs_work:
                f.write("VSLA Limitations (>2.0x):
")
                for item in needs_work:
                    f.write(f"  ⚠ {item}
")
                f.write("
")
        
        f.write("Raw data files available in parsed_data/ directory
")
        f.write("Individual run logs available in raw_logs/ directory
")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_analysis.py <session_dir>")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    try:
        results = analyze_csv_files(session_dir)
        generate_summary_report(results, session_dir)
        print(f"✅ Analysis complete. Results in {session_dir}/PAPER_RESULTS_SUMMARY.txt")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)