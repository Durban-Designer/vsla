#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def analyze_benchmark_data(session_dir):
    """Analyze benchmark results and generate summary statistics."""
    
    parsed_dir = Path(session_dir) / "parsed_data"
    results = {}
    
    # Analyze vector addition results
    vector_files = list(parsed_dir.glob("vector_add_run_*.csv"))
    if vector_files:
        all_vector_data = []
        for f in vector_files:
            df = pd.read_csv(f)
            df['run'] = int(f.stem.split('_')[-1])
            all_vector_data.append(df)
        
        vector_df = pd.concat(all_vector_data, ignore_index=True)
        vector_stats = vector_df.groupby('Configuration').agg({
            'VSLA_ms': ['mean', 'std', 'min', 'max'],
            'Manual_ms': ['mean', 'std', 'min', 'max'], 
            'Ratio': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        results['vector_addition'] = vector_stats
    
    # Analyze FFT results
    fft_files = list(parsed_dir.glob("fft_run_*.csv"))
    if fft_files:
        all_fft_data = []
        for f in fft_files:
            df = pd.read_csv(f)
            df['run'] = int(f.stem.split('_')[-1])
            all_fft_data.append(df)
        
        fft_df = pd.concat(all_fft_data, ignore_index=True)
        fft_stats = fft_df.groupby('Size').agg({
            'Time_ms': ['mean', 'std', 'min', 'max'],
            'Ops_per_ms': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        results['fft_convolution'] = fft_stats
    
    # Analyze Kronecker results
    kron_files = list(parsed_dir.glob("kronecker_run_*.csv"))
    if kron_files:
        all_kron_data = []
        for f in kron_files:
            df = pd.read_csv(f)
            df['run'] = int(f.stem.split('_')[-1])
            all_kron_data.append(df)
        
        kron_df = pd.concat(all_kron_data, ignore_index=True)
        kron_df['Config'] = kron_df['Size_A'].astype(str) + 'x' + kron_df['Size_B'].astype(str)
        kron_stats = kron_df.groupby('Config').agg({
            'VSLA_ms': ['mean', 'std', 'min', 'max'],
            'Ratio_Manual': ['mean', 'std', 'min', 'max'],
            'Ratio_Simple': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        results['kronecker_product'] = kron_stats
    
    # Analyze stacking results
    for stacking_type in ['window_stacking', 'pyramid_stacking']:
        stacking_files = list(parsed_dir.glob(f"bench_{stacking_type}_run_*.csv"))
        if stacking_files:
            all_stacking_data = []
            for f in stacking_files:
                df = pd.read_csv(f)
                df['run'] = int(f.stem.split('_')[-1])
                all_stacking_data.append(df)
            
            stacking_df = pd.concat(all_stacking_data, ignore_index=True)
            stacking_stats = stacking_df.groupby('Configuration').agg({
                'VSLA_ms': ['mean', 'std', 'min', 'max'],
                'Competitor_ms': ['mean', 'std', 'min', 'max'],
                'Ratio': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            results[stacking_type] = stacking_stats
    
    return results

def generate_summary_report(results, session_dir):
    """Generate a comprehensive summary report."""
    
    report_path = Path(session_dir) / "PAPER_RESULTS_SUMMARY.txt"
    
    with open(report_path, 'w') as f:
        f.write("VSLA PERFORMANCE BENCHMARKS - PAPER RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Statistical Summary (10 runs per benchmark)\n")
        f.write("Format: mean ± std (min, max)\n\n")
        
        # Vector Addition Results
        if 'vector_addition' in results:
            f.write("VECTOR ADDITION PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            va_stats = results['vector_addition']
            for config in va_stats.index:
                vsla_mean = va_stats.loc[config, ('VSLA_ms', 'mean')]
                vsla_std = va_stats.loc[config, ('VSLA_ms', 'std')]
                vsla_min = va_stats.loc[config, ('VSLA_ms', 'min')]
                vsla_max = va_stats.loc[config, ('VSLA_ms', 'max')]
                
                ratio_mean = va_stats.loc[config, ('Ratio', 'mean')]
                ratio_std = va_stats.loc[config, ('Ratio', 'std')]
                
                f.write(f"{config:15s}: {vsla_mean:.3f}±{vsla_std:.3f} ms ")
                f.write(f"({vsla_min:.3f}, {vsla_max:.3f}) ")
                f.write(f"Ratio: {ratio_mean:.2f}±{ratio_std:.2f}\n")
            f.write("\n")
        
        # FFT Results
        if 'fft_convolution' in results:
            f.write("FFT CONVOLUTION PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            fft_stats = results['fft_convolution']
            for size in sorted(fft_stats.index):
                time_mean = fft_stats.loc[size, ('Time_ms', 'mean')]
                time_std = fft_stats.loc[size, ('Time_ms', 'std')]
                
                ops_mean = fft_stats.loc[size, ('Ops_per_ms', 'mean')]
                ops_std = fft_stats.loc[size, ('Ops_per_ms', 'std')]
                
                f.write(f"Size {size:4d}: {time_mean:.3f}±{time_std:.3f} ms, ")
                f.write(f"{ops_mean:.0f}±{ops_std:.0f} ops/ms\n")
            f.write("\n")
        
        # Kronecker Results
        if 'kronecker_product' in results:
            f.write("KRONECKER PRODUCT PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            kron_stats = results['kronecker_product']
            for config in kron_stats.index:
                ratio_manual_mean = kron_stats.loc[config, ('Ratio_Manual', 'mean')]
                ratio_manual_std = kron_stats.loc[config, ('Ratio_Manual', 'std')]
                
                ratio_simple_mean = kron_stats.loc[config, ('Ratio_Simple', 'mean')]
                ratio_simple_std = kron_stats.loc[config, ('Ratio_Simple', 'std')]
                
                f.write(f"{config:8s}: Manual {ratio_manual_mean:.2f}±{ratio_manual_std:.2f}, ")
                f.write(f"Simple {ratio_simple_mean:.2f}±{ratio_simple_std:.2f}\n")
            f.write("\n")
        
        # Performance Summary
        f.write("PERFORMANCE ANALYSIS SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        # Identify where VSLA excels vs struggles
        if 'vector_addition' in results:
            va_stats = results['vector_addition']
            excellent = []
            competitive = []
            needs_work = []
            
            for config in va_stats.index:
                ratio_mean = va_stats.loc[config, ('Ratio', 'mean')]
                if ratio_mean < 1.1:
                    excellent.append(f"{config} ({ratio_mean:.2f}x)")
                elif ratio_mean < 2.0:
                    competitive.append(f"{config} ({ratio_mean:.2f}x)")
                else:
                    needs_work.append(f"{config} ({ratio_mean:.2f}x)")
            
            if excellent:
                f.write("VSLA Excels (≤1.1x):\n")
                for item in excellent:
                    f.write(f"  ✓ {item}\n")
                f.write("\n")
            
            if competitive:
                f.write("VSLA Competitive (1.1-2.0x):\n")
                for item in competitive:
                    f.write(f"  ~ {item}\n")
                f.write("\n")
            
            if needs_work:
                f.write("VSLA Limitations (>2.0x):\n")
                for item in needs_work:
                    f.write(f"  ⚠ {item}\n")
                f.write("\n")
        
        f.write("Raw data files available in parsed_data/ directory\n")
        f.write("Individual run logs available in raw_logs/ directory\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <session_dir>")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    try:
        results = analyze_benchmark_data(session_dir)
        generate_summary_report(results, session_dir)
        print(f"✅ Analysis complete. Results in {session_dir}/PAPER_RESULTS_SUMMARY.txt")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)
