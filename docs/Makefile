# Makefile for building VSLA paper PDF

.PHONY: all clean paper

all: paper

paper: vsla_paper.pdf

vsla_paper.pdf: vsla_paper.tex
	@echo "Building PDF from LaTeX..."
	pdflatex vsla_paper.tex
	pdflatex vsla_paper.tex  # Run twice for references
	@echo "PDF generated: vsla_paper.pdf"

clean:
	rm -f *.aux *.log *.out *.toc *.pdf

view: vsla_paper.pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open vsla_paper.pdf; \
	elif command -v open > /dev/null; then \
		open vsla_paper.pdf; \
	else \
		echo "Please open vsla_paper.pdf manually"; \
	fi

# Alternative using latexmk if available
latexmk: vsla_paper.tex
	latexmk -pdf vsla_paper.tex

clean-all:
	latexmk -C
	rm -f *.bbl *.blg