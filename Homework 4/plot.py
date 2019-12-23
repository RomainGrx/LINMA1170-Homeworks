#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date   : Wednesday, November 20th 2019
"""
# ======================= Import ==============================
# In[0]
from csrGMRES import csrGMRES, CSRformat, csrILU0, invCSRformat
import ndt
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Purple, Orange, Blue
COLORS = [purple, orange, blue, green] = ['#9400D3', '#FFA500', '#0080FF', '#92D832']

def density(A):
    NZ = np.nonzero(A)[0]
    All= len(A)
    All *= All
    return float(len(NZ))/All

def regimes(refs, iterations=1000, save=False, name='regimes'):
    vels  = [0, 50]
    freqs = [0, 50]
    linestyle   = [[10,0], [10,10]]
    lwidth = [1.5, 3.0]
    names = ['statique', 'harmonique', 'stationnaire', 'dynamique']
    xranger = np.arange(iterations)
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    plt.title('Convergence de la norme du résidu en fonction du nombre d\'itérations et des différents régimes')
    plt.xlabel('Nombre d\'itérations')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Norme du résidu (||b-Ax$_n$||$_2$)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for (i, ndt.ref) in zip(range(len(refs)), refs):
        for (j, ndt.freq) in zip(range(len(freqs)), freqs):
            for (k, ndt.vel) in zip(range(len(vels)), vels):
                ndt.main()
                A = np.array(ndt.A)
                b = np.array(ndt.b)
                _, res = csrGMRES(*CSRformat(A), b, rtol=1e-30, prec=False, max_iterations=iterations, res_history=[])
                plt.loglog(xranger, res, dashes=linestyle[j], color=COLORS[len(freqs)*j+k], label='Régime %s (%d noeuds)'%(names[len(freqs)*j+k], ndt.Nodes), linewidth=lwidth[i])
    plt.legend()
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def precond(refs, iterations=1000, save=False, name='precond'):
    ndt.vel = 0
    ndt.freq = 0
    linestyle   = ['--', '-']
    precs = [True, False]
    names = ['Avec préconditionnement', 'Sans préconditionnement']
    xranger = np.arange(iterations)
    fig = plt.Figure(figsize=(15,22))
    ax = fig.add_subplot(111)
    plt.title('Convergence de la norme du résidu en fonction du nombre d\'itéartions et du préconditionnement')
    plt.xlabel('Nombre d\'itérations')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Norme du résidu')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for (i, ndt.ref) in zip(range(len(refs)), refs):
        for j in range(len(precs)):
            ndt.main()
            A = np.array(ndt.A)
            b = np.array(ndt.b)
            _, res = csrGMRES(*CSRformat(A), b, rtol=1e-30, prec=precs[j], max_iterations=iterations, res_history=[])
            print('res 0 : ',res[0])
            plt.loglog(xranger, res, linestyle[i], color=COLORS[i+j], label='%s (%d noeuds)'%(names[i+j], ndt.Nodes))
    plt.legend()
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def spectrum(save=False, name='spectrum'):
    ndt.vel = 0
    ndt.freq= 0
    ndt.ref = 1
    ndt.main(solver=False, show=False, test=False)
    ndt.A = np.array(ndt.A)
    ndt.b = np.array(ndt.b)
    marksize = 4

    fig = plt.Figure()
    plt.subplot(211)
    plt.title('Spectre de la matrice initiale et de la matrice préconditionnée')
    plt.xlabel('$\Re(z)$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('$\Im(z)$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    eigen, _ = lin.eig(ndt.A)
    plt.plot(np.real(eigen), np.imag(eigen), 'o', markersize=marksize, color=orange,label='Spectre de la matrice initiale ($A$)')

    sM, iM, jM = csrILU0(*CSRformat(ndt.A))
    ILU = invCSRformat(sM, iM, jM)
    L = np.tril(ILU, -1) + np.eye(len(ILU))
    U = np.triu(ILU)
    M_1 = lin.inv(np.dot(L, U))@ndt.A
    eigen, _ = lin.eig(M_1)
    plt.plot(np.real(eigen), np.imag(eigen), 'o', markersize=marksize+2, color=purple, label='Spectre de la matrice préconditionnée (M$^{-1}A$)')
    plt.legend()
    plt.subplot(212)
    plt.title('Spectre de la matrice préconditionnée')
    plt.xlabel('$\Re(z)$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('$\Im(z)$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.plot(np.real(eigen), np.imag(eigen), 'o', markersize=marksize+2, color=purple, label='Spectre de la matrice préconditionnée (M$^{-1}A$)')
    plt.legend()
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return



if __name__=='__main__':
    big_acc = np.linspace(1,5,10)
    lil_acc = np.linspace(1,2,3)
    two_ref = np.array([1, 3])

    # regimes(two_ref, save=False)
    # precond([1], save=False, iterations=300)
    # print(np.log10([1e-12]))
    spectrum()
