TEX = rapport.tex
PDF = rapport.pdf
ZIP = ndt.py mysolve.py plot.py $(TEX) $(PDF) res Makefile README.md
.PHONY : pdf clean

default : pdf ndt

zip:
	zip -r 28681700_Romain_Graux.zip $(ZIP)

pdf :
	pdflatex $(TEX)
	open $(PDF)
	rm *.log *.aux
ndt :
	python ndt.py

clean :
	rm *.log *.aux *.out *.zip
