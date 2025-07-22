#!/usr/bin/env python3
"""
VSLA Python Package Setup
"""

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension, find_packages

# The main interface to the package
__version__ = "0.1.0"

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Get the absolute path to the current directory
current_dir = Path(__file__).parent.absolute()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        pybind11_cmake_dir = os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11')
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            "-DVSLA_BUILD_PYTHON=ON",
            "-DVSLA_ENABLE_TESTS=OFF",
            "-DVSLA_ENABLE_BENCHMARKS=OFF",
        ]

        build_args = []
        
        # Platform-specific configuration
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{os.cpu_count() or 4}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir + "/../"] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="vsla",
    version=__version__,
    author="Royce Birnbaum",
    author_email="royce.birnbaum@gmail.com",
    description="Variable-Shape Linear Algebra: Mathematical foundations and high-performance implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/royce-birnbaum/vsla",
    project_urls={
        "Bug Tracker": "https://github.com/royce-birnbaum/vsla/issues",
        "Documentation": "https://github.com/royce-birnbaum/vsla/blob/main/README.md",
        "Source Code": "https://github.com/royce-birnbaum/vsla",
    },
    packages=find_packages(),
    package_dir={"": "."},
    ext_modules=[CMakeExtension("vsla._core")],
    cmdclass={"build_ext": CMakeBuild},
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
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            "myst-parser",
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
    keywords="linear-algebra, tensors, semiring, automatic-differentiation, high-performance-computing",
)