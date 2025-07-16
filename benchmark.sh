#!/bin/bash
# VSLA Benchmark Runner
# Simple wrapper script to run comprehensive benchmarks

cd "$(dirname "$0")/bench"
python3 run_benchmark.py "$@"