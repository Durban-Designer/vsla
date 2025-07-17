# VSLA Research Papers

This directory contains the research papers for the Variable-Shape Linear Algebra (VSLA) library.

## Available Papers

### 1. Regular Version (`vsla_paper.pdf`)
- **Format**: Standard academic article format  
- **Pages**: 12 pages
- **Template**: Standard LaTeX article class
- **Content**: Complete mathematical foundations, implementation details, and empirical evaluation

### 2. ACM Conference Version (`vsla_paper_acm.pdf`)
- **Format**: ACM conference format (two-column)
- **Pages**: 2 pages (condensed version)
- **Template**: ACM-compatible formatting
- **Content**: Condensed overview focusing on key contributions and results

## Build System

### Quick Start
```bash
# Build both papers
make all

# Build individual papers
make regular    # Standard version
make acm        # ACM version

# Check build status
make status

# Clean all generated files
make clean
```

### Build Requirements

**Essential Dependencies:**
- `pdflatex` - LaTeX compiler
- `amsmath`, `tikz`, `algorithm` packages
- Standard LaTeX distribution (TeX Live recommended)

**Optional Dependencies:**
- `acmart.cls` - Official ACM template (auto-fallback available)
- `bibtex` - For bibliography (if references added)

**Installation on Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-latex-extra
sudo apt-get install texlive-publishers texlive-science
sudo apt-get install texlive-fonts-recommended
```

### Build Features

✅ **Automatic Template Detection**: Uses official ACM template if available, falls back to compatibility version  
✅ **Clean Artifact Management**: All `.aux`, `.log`, `.out` files are isolated in `src/`  
✅ **Dependency Checking**: `make check-deps` verifies LaTeX installation  
✅ **Error Handling**: Graceful fallbacks and clear error messages  
✅ **Development Workflow**: `make draft` for quick single-pass builds  

### Advanced Usage

```bash
# Check LaTeX dependencies
make check-deps

# Build with verbose output
make regular PDFLATEX="pdflatex -interaction=errorstopmode"

# Quick draft build (single pass)
make draft

# Force rebuild ignoring timestamps
make force

# Test compilation without keeping outputs
make test

# Get build system help
make help
```

## Directory Structure

```
papers/
├── README.md                    # This file
├── Makefile                     # Build system
├── vsla_paper.pdf              # Standard version (output)
├── vsla_paper_acm.pdf          # ACM version (output)
└── src/                        # Source files and build artifacts
    ├── vsla_paper.tex          # Main paper source
    ├── vsla_paper_acm.tex      # Original ACM version
    ├── vsla_paper_acm_compat.tex # ACM compatibility version
    ├── acmart-compat.sty       # ACM fallback template
    └── *.aux, *.log, *.out     # Build artifacts (auto-generated)
```

## Content Overview

### Mathematical Foundations
- **Equivalence Classes**: Formalization of dimension-aware vectors through zero-padding equivalence
- **Semiring Structures**: Two models for variable-shape computation:
  - **Model A (Convolution)**: FFT-based operations, commutative semiring
  - **Model B (Kronecker)**: Tiled operations, non-commutative semiring
- **Complexity Analysis**: Asymptotic bounds for FFT-accelerated operations

### Implementation Highlights  
- **C99 Library**: Production-ready implementation with comprehensive error handling
- **Python Bindings**: Full Python integration with NumPy compatibility
- **Hardware Abstraction**: Automatic CPU/GPU selection and optimization
- **Performance Validation**: 3-16× speedups demonstrated empirically

### Key Results
- **Memory Efficiency**: 20-50% reduction through variable shapes vs. fixed padding
- **Development Productivity**: 10-50× reduction in code complexity
- **Performance**: Comparable to manual optimization with automatic hardware selection

## Paper Versions Explained

### Why Two Versions?

**Regular Version (12 pages)**:
- Complete mathematical exposition with full proofs
- Detailed implementation architecture
- Comprehensive experimental evaluation
- Suitable for journal submission or technical report

**ACM Version (2 pages)**:
- Condensed overview highlighting key contributions
- Focus on novel mathematical structures and performance results  
- Conference-appropriate format and length
- Suitable for workshop or poster presentations

### Content Mapping

| Section | Regular Paper | ACM Paper |
|---------|---------------|-----------|
| Abstract | Full (4 contributions) | Condensed |
| Introduction | Complete motivation | Key problem only |
| Math Foundations | Full formal development | Core definitions |
| Implementation | Detailed architecture | Performance highlights |
| Evaluation | Complete experimental study | Key results |
| Related Work | Comprehensive survey | Brief comparison |

## Compilation Status

### Current Build Health
- ✅ **Regular paper**: Compiles successfully (12 pages, ~294KB)
- ✅ **ACM paper**: Compiles with compatibility template (2 pages, ~152KB)
- ✅ **Build system**: Fully automated with dependency checking
- ✅ **Clean artifacts**: No build files pollute main directory

### Known Issues
- Some undefined references in regular paper (incomplete sections)
- Minor overfull hboxes (formatting warnings, not errors)
- ACM template fallback uses two-column article instead of true ACM format

### Quality Assurance
- **Compilation**: Both papers build without fatal errors
- **Content**: Mathematical notation consistent across versions
- **Formatting**: Professional appearance with proper spacing
- **Dependencies**: Minimal requirements, broad compatibility

## Development Workflow

### For Authors
1. **Edit sources** in `src/` directory
2. **Build drafts** with `make draft` for quick iteration  
3. **Check status** with `make status` to verify builds
4. **Clean workspace** with `make clean` before commits

### For Reviewers
1. **Download PDFs** directly from main directory
2. **Check latest build** with `make status`
3. **Rebuild if needed** with `make all`

### For Deployment
1. **Final build** with `make force` to ensure clean compilation
2. **Verify outputs** are present in main directory
3. **Archive sources** including `src/` for reproducibility

## Citation

```bibtex
@article{birnbaum2025vsla,
  title={Variable-Shape Linear Algebra: Mathematical Foundations and High-Performance Implementation},
  author={Birnbaum, Royce},
  year={2025},
  month={July},
  note={Technical Report}
}
```

## Support

For build issues:
1. Check `make check-deps` for missing LaTeX packages
2. Review `src/*.log` files for compilation errors
3. Try `make clean && make test` for diagnostic information
4. See `make help` for additional options

---

**Last Updated**: July 17, 2025  
**Build System Version**: 1.0  
**Tested Platforms**: Ubuntu 20.04+, TeX Live 2023+