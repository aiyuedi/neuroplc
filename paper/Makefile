# NeuroPLC Paper Build
# ====================
# xelatex + bibtex + xelatex × 2 for cross-references
# Requires: MiKTeX (xelatex) with IEEEtran class

MAIN = main
TEXFILES = main.tex section_svnn.tex section_svnn_chebykan.tex section_svnn_theorems.tex references.bib
FIGDIR = figures

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(TEXFILES)
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).synctex.gz
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls
	rm -f *.aux

view: $(MAIN).pdf
	start $(MAIN).pdf
