TEX = rapport.tex
PDF = rapport.pdf
PY  = ndt.py csrGMRES.py plot.py
RES = res/imgs res/plots res/py res/tex
ZIP = $(PY) $(TEX) $(PDF) $(RES) Makefile

.PHONY : pdf clean

default : pdf

zip:
	zip -r 28681700_Romain_Graux.zip $(ZIP)

ls :
	ls $(RES)

pdf :
	pdflatex $(TEX)
	open $(PDF)
	rm *.log *.aux
ndt :
	python ndt.py

clean :
	rm *.log *.aux *.out *.zip
