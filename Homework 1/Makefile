PY = mysolve.py


default : pdf open clean

py :
	python $(PY)

test :
	python test.py

ndt :
	python ndt.py

pdf :
	pdflatex hw1.tex

open :
	open hw1.pdf

zip : pdf
	zip myzip.zip hw1.pdf hw1.tex

clean :
	rm -rf *.log *.aux
