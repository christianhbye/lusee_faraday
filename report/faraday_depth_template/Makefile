# Build the Faraday-depth report.
#
# Figures come from report/figures/ and are tracked (the .gitignore
# carries an exception for step5_*.pdf), so this builds from a fresh
# checkout with no analysis re-run.  To regenerate the figures
# themselves:  uv run python scripts/step5_plots.py
#
# Every NUMBER lives in generated.tex, which scripts/step5_tables.py
# writes straight from the npz products -- nothing is typed into the
# .tex by hand.  It is tracked for the same reason the figures are.

GEN  := generated.tex
TEX  := faraday_depth_template.tex
PDF  := $(TEX:.tex=.pdf)
FIGS := $(wildcard ../figures/step5_*.pdf)

.PHONY: all clean

all: $(PDF)

# Two passes: the first writes the .aux/.toc, the second resolves the
# table of contents and every \ref.  -halt-on-error so a broken build
# fails loudly instead of dropping into interactive mode.
$(PDF): $(TEX) $(GEN) $(FIGS)
	pdflatex -halt-on-error -interaction=nonstopmode $(TEX)
	pdflatex -halt-on-error -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz $(PDF)
