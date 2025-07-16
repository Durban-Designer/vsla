#!/usr/bin/env python3
"""
Compare benchmark results for performance regression detection
"""

import json
import argparse
import sys
from pathlib import Path
import statistics

def load_benchmark_results(results_dir):
    """Load benchmark results from a directory"""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory {results_dir} does not exist", file=sys.stderr)
        return results
    
    for json_file in results_path.glob("*.json"):
        if json_file.name == "config.json" or json_file.name == "summary.json":
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'benchmark' in item:
                            key = f"{item['benchmark']}_{item.get('method', 'unknown')}"
                            if key not in results:
                                results[key] = []
                            results[key].append(item.get('results', {}).get('mean_time_us', 0))
                elif isinstance(data, dict) and 'benchmark' in data:
                    key = f"{data['benchmark']}_{data.get('method', 'unknown')}"
                    if key not in results:
                        results[key] = []
                    results[key].append(data.get('results', {}).get('mean_time_us', 0))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)
    
    return results

def compute_aggregate_time(times):
    """Compute aggregate timing from list of measurements"""
    if not times:
        return 0.0
    return statistics.mean(times)

def compare_performance(baseline_results, current_results, threshold):
    """Compare performance between baseline and current results"""
    regressions = []
    improvements = []
    stable = []
    
    all_benchmarks = set(baseline_results.keys()) | set(current_results.keys())
    
    for benchmark in sorted(all_benchmarks):
        baseline_time = compute_aggregate_time(baseline_results.get(benchmark, []))
        current_time = compute_aggregate_time(current_results.get(benchmark, []))
        
        if baseline_time == 0 and current_time == 0:
            continue
        elif baseline_time == 0:
            # New benchmark
            print(f"NEW: {benchmark} - {current_time:.3f} μs")
            continue
        elif current_time == 0:
            # Removed benchmark
            print(f"REMOVED: {benchmark} - was {baseline_time:.3f} μs")
            continue
        
        # Calculate relative change
        relative_change = (current_time - baseline_time) / baseline_time
        
        if relative_change > threshold:
            regressions.append((benchmark, baseline_time, current_time, relative_change))
        elif relative_change < -threshold:
            improvements.append((benchmark, baseline_time, current_time, relative_change))
        else:
            stable.append((benchmark, baseline_time, current_time, relative_change))
    
    return regressions, improvements, stable

def print_comparison_report(regressions, improvements, stable, threshold):
    """Print detailed comparison report"""
    
    print("=" * 80)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 80)
    
    if regressions:
        print(f"\n🚨 PERFORMANCE REGRESSIONS (>{threshold*100:.1f}% slower):")
        print("-" * 60)
        for benchmark, baseline, current, change in regressions:
            print(f"  {benchmark:40} {baseline:10.3f} → {current:10.3f} μs ({change:+7.2%})")
    
    if improvements:
        print(f"\n🚀 PERFORMANCE IMPROVEMENTS (>{threshold*100:.1f}% faster):")
        print("-" * 60)
        for benchmark, baseline, current, change in improvements:
            print(f"  {benchmark:40} {baseline:10.3f} → {current:10.3f} μs ({change:+7.2%})")
    
    if stable:
        print(f"\n✅ STABLE PERFORMANCE (±{threshold*100:.1f}%):")
        print("-" * 60)
        for benchmark, baseline, current, change in stable[:10]:  # Limit output
            print(f"  {benchmark:40} {baseline:10.3f} → {current:10.3f} μs ({change:+7.2%})")
        if len(stable) > 10:
            print(f"  ... and {len(stable) - 10} more stable benchmarks")
    
    print(f"\nSUMMARY:")
    print(f"  Regressions:  {len(regressions):3d}")
    print(f"  Improvements: {len(improvements):3d}")
    print(f"  Stable:       {len(stable):3d}")
    print(f"  Total:        {len(regressions) + len(improvements) + len(stable):3d}")

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('--baseline', required=True, 
                       help='Directory containing baseline benchmark results')
    parser.add_argument('--current', required=True,
                       help='Directory containing current benchmark results')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Threshold for detecting regressions (default: 0.05 = 5%%)')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with non-zero code if regressions detected')
    parser.add_argument('--output', help='Save comparison report to file')
    
    args = parser.parse_args()
    
    # Load results
    baseline_results = load_benchmark_results(args.baseline)
    current_results = load_benchmark_results(args.current)
    
    if not baseline_results and not current_results:
        print("Error: No benchmark results found in either directory", file=sys.stderr)
        return 1
    
    # Compare performance
    regressions, improvements, stable = compare_performance(
        baseline_results, current_results, args.threshold
    )
    
    # Print report
    print_comparison_report(regressions, improvements, stable, args.threshold)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Redirect stdout to file temporarily
            original_stdout = sys.stdout
            sys.stdout = f
            print_comparison_report(regressions, improvements, stable, args.threshold)
            sys.stdout = original_stdout
        print(f"\nReport saved to: {args.output}")
    
    # Exit with error if regressions found and requested
    if args.fail_on_regression and regressions:
        print(f"\n❌ FAILURE: {len(regressions)} performance regression(s) detected!")
        return 1
    
    print(f"\n✅ SUCCESS: No significant performance regressions detected.")
    return 0

if __name__ == '__main__':
    sys.exit(main())