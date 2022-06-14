.PHONY: main.pdf
main.pdf: main.tex
	pdflatex main
	pdflatex main
	bibtex main
	pdflatex main


clean:
	rm -f *.aux *.log *.toc *.spl *.out *.bbl *.blg *~ \#*\#
