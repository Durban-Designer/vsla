#!/usr/bin/env python3
"""
VSLA Python Package Setup - Simple Direct Build
Uses pybind11 directly without CMake integration
"""

import os
from pathlib import Path
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

__version__ = "0.1.0"

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "vsla._core",
        [
            "src/bindings.cpp",
        ],
        include_dirs=[
            # Path to VSLA headers
            "../include",
            # Path to pybind11 headers (automatically handled by Pybind11Extension)
        ],
        library_dirs=[
            # Path to pre-built VSLA library with PIC
            "../build_pic",
        ],
        libraries=[
            # Link against the pre-built static library
            "vsla_static",
        ],
        define_macros=[
            ("VERSION_INFO", '"dev"'),
            ("VSLA_BUILD_CPU", "1"),
        ],
        cxx_std=14,
        language='c++',
    ),
]

setup(
    name="vsla",
    version=__version__,
    author="Royce Birnbaum",
    author_email="royce.birnbaum@gmail.com",
    description="Variable-Shape Linear Algebra: Mathematical foundations and high-performance implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/royce-birnbaum/vsla",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark", 
            "black",
            "flake8",
            "mypy",
        ],
        "benchmarks": [
            "scipy>=1.7.0",
            "matplotlib>=3.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)