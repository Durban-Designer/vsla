# VSLA Documentation

This directory contains the theoretical paper and documentation for the Variable-Shape Linear Algebra (VSLA) library.

## Building the Paper

The VSLA theoretical paper is written in LaTeX. To build the PDF:

### Using Make
```bash
cd docs
make paper
```

### Using pdflatex directly
```bash
cd docs
pdflatex vsla_paper.tex
pdflatex vsla_paper.tex  # Run twice to resolve references
```

### Using latexmk (if installed)
```bash
cd docs
latexmk -pdf vsla_paper.tex
```

## Viewing the Paper

After building:
```bash
make view  # Opens the PDF in your default viewer
```

Or open `vsla_paper.pdf` manually.

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- pdflatex command
- Optional: latexmk for automatic compilation

## Paper Contents

The paper covers:
- Mathematical foundations of VSLA
- Model A: Convolution semiring
- Model B: Kronecker semiring
- Variable-shape matrices and operations
- Computational complexity analysis
- Applications in AI, signal processing, and scientific computing