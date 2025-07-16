#!/usr/bin/env python3
"""
Analyze benchmark results for VSLA performance evaluation
"""

import json
import sys
from pathlib import Path

def analyze_results(results_dir):
    """Analyze benchmark results from directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist")
        return
    
    json_files = list(results_path.glob("*.json"))
    if not json_files:
        print("No benchmark results found")
        return
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'benchmark' in item:
                            print(f"  {item['benchmark']}: {item.get('results', {}).get('mean_time_us', 'N/A')} μs")
                elif isinstance(data, dict) and 'benchmark' in data:
                    print(f"  {data['benchmark']}: {data.get('results', {}).get('mean_time_us', 'N/A')} μs")
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        analyze_results('results/latest/')