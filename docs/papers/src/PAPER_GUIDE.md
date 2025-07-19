# VSLA Paper Guide

This directory contains the source files for two versions of the Variable-Shape Linear Algebra (VSLA) paper.

## Paper Versions

### 1. Main Paper (`vsla_paper.tex`)
- **Purpose**: Full academic paper with complete mathematical details, proofs, and comprehensive evaluation
- **Length**: ~19 pages
- **Format**: Standard article class
- **Use for**: ArXiv submissions, journal submissions, detailed technical reference

### 2. ACM Extended Abstract (`vsla_paper_acm.tex`)
- **Purpose**: Condensed 2-page version for ACM conferences and workshops
- **Length**: Exactly 2 pages
- **Format**: ACM `sigconf` template
- **Use for**: Conference submissions with strict page limits, quick overview of the work

## Version Management

### Current Version: v0.1
- Both papers are synchronized at version 0.1
- Compiled PDFs are stored in `/docs/papers/`:
  - `vsla_paper_v0.1.pdf` - Full paper
  - `vsla_paper_acm_v0.1.pdf` - ACM 2-page version

### Workflow
1. **Development**: Make all changes to the main paper (`vsla_paper.tex`)
2. **Versioning**: Only update the ACM version at major milestones (v0.2, v0.3, etc.)
3. **Compilation**: 
   ```bash
   # Compile main paper
   pdflatex vsla_paper.tex
   pdflatex vsla_paper.tex  # Run twice for references
   
   # Compile ACM version
   pdflatex vsla_paper_acm.tex
   pdflatex vsla_paper_acm.tex  # Run twice for references
   ```

### Key Differences

| Aspect | Main Paper | ACM Version |
|--------|------------|-------------|
| Mathematical proofs | Full proofs included | Only key theorems stated |
| Examples | Multiple detailed examples | One running example |
| Evaluation | Comprehensive benchmarks | Summary results only |
| Related work | Detailed comparison | Brief mentions |
| Appendix | Monoidal category proof | None |

## Maintenance Notes

- The ACM version requires careful editing to fit the 2-page limit
- Focus on core contributions: semiring models, implementation, and key results
- Remove verbose proofs, extended examples, and detailed evaluations
- Keep the abstract concise but comprehensive
- Tables and figures must be minimal in the ACM version

## Future Versions

When creating new versions:
1. Update both `.tex` files as needed
2. Compile both PDFs
3. Copy to `/docs/papers/` with new version number (e.g., `vsla_paper_v0.2.pdf`)
4. Update this guide with version notes