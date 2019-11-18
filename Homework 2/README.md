Homework 2 : LU & QR decomposition
=========
Makefile
--------

Compiler le pdf
```shell
make
```

Lancer la simulation
```shell
make ndt
```

Format
------

> Solvers & Factorisation

Tous les solvers et les factorisations sont repris dans le fichier ***mysolve*** :

- *LU*
- *QR*
- *ILU*
- *Cholesky*

> Ploters

Tous les ploters sont repris dans le fichier ***plot*** :

- *Complexité dû au raffinement : **plot_ref***
- *Complexité dû au régime : **plot_regimes***
- *Conditionnement de A et de ILU<sup>-1</sup>A : **plot_ilu***
- *Précision de la solution : **plot_accuracy***
