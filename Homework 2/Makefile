TEX = rapport.tex
PDF = rapport.pdf
.PHONY : pdf clean

default : pdf

pdf :
	pdflatex $(TEX)
	open $(PDF)
	rm *.log *.aux
clean :
	rm *.log *.aux
