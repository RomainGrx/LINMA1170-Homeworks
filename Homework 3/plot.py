#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date   : Wednesday, November 20th 2019
"""
# ======================= Import ==============================
# In[0]
import mysolve as my
import ndt
from LU import LU, LUsolve, pivot
from LUcsr import LUcsrsolve, RCMK, CSRformat, bandwidth, invCSRformat, LUcsr
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Purple, Orange, Blue
COLORS = [purple, orange, blue] = ['#9400D3', '#FFA500', '#0080FF']

def density(A):
    NZ = np.nonzero(A)[0]
    All= len(A)
    All *= All
    return float(len(NZ))/All


def all_RCMK():
    ndt.main()
    A = ndt.A
    L = LU.LU(A)[0]
    sA, iA, jA = CSRformat(A)
    r = RCMK(iA, jA)
    a_rcmk = (A[:, r])[r, :]
    sA, iA, jA = CSRformat(A)
    sA, iA, jA = CSRformat(a_rcmk)
    LRMCK = LU.LU(a_rcmk)[0]
    sA, iA, jA = CSRformat(LRMCK)
    plt.subplot(221)
    plt.title('Initial A')
    plt.spy(A)
    plt.subplot(222)
    plt.title('LU with initial A')
    plt.spy(L)
    plt.subplot(223)
    plt.title('A reordered')
    plt.spy((a_rcmk))
    plt.subplot(224)
    plt.title('LU with A reordered')
    plt.spy(LRMCK)
    plt.show()

def denseLU(precision, save=False, name='DensityRCMK'):
    Standard_densities = []
    RCMK_densities     = []
    ARCMK_densiies     = []
    sizes = []
    file = 'res/numpy/%s/%.1f.npy'
    for ref in precision:
        ndt.ref = ref
        AR     = np.load(file%('A/LUcsrsolveRCMK',ndt.ref))
        LU     = np.load(file%('LU/LUcsrsolve',ndt.ref))
        LURCMK = np.load(file%('LU/LUcsrsolveRCMK',ndt.ref))
        Standard_densities.append(density(LU))
        RCMK_densities.append(density(LURCMK))
        ARCMK_densiies.append(density(AR))
        sizes.append(len(LU))

    plt.Figure()
    plt.title('Densité des différentes matrices')
    plt.xlabel('Nombre de noeuds')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Densité')
    pr = plt.plot(sizes, ARCMK_densiies, '--o', color=orange)
    ps = plt.plot(sizes, Standard_densities, '--o', color=purple)
    pr = plt.plot(sizes, RCMK_densities, '--x', color=purple)
    plt.legend(['A', 'LU', 'LU réordonnée'])
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def denseA(precision, save=False, name='DensityA'):
    Standard_densities = []
    sizes = []
    for ref in precision:
        ndt.ref = ref
        ndt.main()
        A = np.array(ndt.A)

        Standard_densities.append(density(A))
        sizes.append(len(A))

    plt.Figure()
    plt.title('Densité de la matrice initiale')
    plt.xlabel('Nombre de noeuds')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Densité')
    plt.scatter(sizes, Standard_densities, color=purple)
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def band(precision, save=False, name='bandRCMK'):
    Standard_width = []
    RCMK_width     = []
    sizes = []
    for ndt.ref in precision:
        ndt.main()
        A = np.array(ndt.A)
        size = len(A)
        sizes.append(size)

        sA, iA, jA = CSRformat(A)
        r = RCMK(iA, jA)
        ARCMK = (A[:, r])[r, :]
        sARCMK, iARCMK, jARCMK = CSRformat(ARCMK)

        Standard_width.append(bandwidth(iA,jA)[2])
        RCMK_width.append(bandwidth(iARCMK,jARCMK)[2])


    plt.Figure()
    plt.title('Largeur de bande des matrices selon RCMK')
    plt.xlabel('Nombre de noeuds')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Largeur de la bande')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ps = plt.plot(sizes, Standard_width, 'o', color=purple)
    pr = plt.plot(sizes, RCMK_width, 'o', color=orange)
    ranger = np.linspace(sizes[0], sizes[-1], 100)
    spoly = np.polyfit(sizes,Standard_width,1)
    rpoly = np.polyfit(sizes,RCMK_width,1)
    pps= plt.plot(ranger,np.poly1d(spoly)(ranger), '--', color=purple)
    ppr= plt.plot(ranger,np.poly1d(rpoly)(ranger), '--', color=orange)
    plt.legend(['Sans RCMK', 'Avec RCMK', 'Polyfit sans RCMK : %.2fx+...'%(spoly[0]), 'Polyfit avec RCMK : %.2fx+...'%(rpoly[0])])
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def CSR(precision, save=False, name='CSR'):
    times = []
    sizes = []
    for ndt.ref in precision:
        ndt.main()
        A = np.array(ndt.A)
        size = len(A)
        sizes.append(size)
        mean = []
        for m in range(4):
            tic = timer()
            sA, iA, jA = CSRformat(A)
            tac = timer()
            mean.append(tac-tic)
        times.append(sum(mean)/len(mean))


    plt.Figure()
    plt.title('Complexité temporelle de CSRformat')
    plt.xlabel('Nombre de noeuds [log$_{10}$]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Temps d\'exécution [log$_{10}$ sec]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    logx = np.log10(sizes)
    logy = np.log10(times)

    ps = plt.scatter(logx, logy, color=purple)
    fitab = np.polyfit(logx, logy, 1)
    fit   = np.poly1d(fitab)
    ranger = np.linspace(logx[0], logx[-1], 100)
    pfit = plt.plot(ranger, fit(ranger), '--', color=orange)
    plt.legend(['Polyfit : (%.2fx-%.2f)'%(fitab[0],abs(fitab[1])),'Complexités temporelles'])
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return


def complexity(precision, save=False, name='complexity'):
    ndt.vel = 0
    ndt.freq = 50
    selfColors = [orange, purple, purple]
    selfChar   = ['--o', '--o', '--x']
    names = ['LUsolve', 'LUcsrsolve', 'LUcsrsolve avec RCMK']
    selfSolver = [LUsolve, LUcsrsolve, LUcsrsolve]
    selfRMCK   = [False, False, True]
    plt.Figure()
    plt.title('Complexité temporelle selon matrice creuse et RCMK')
    plt.xlabel('Nombre de noeuds')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Temps d\'exécution [sec]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plotID = []
    for i in range(len(selfSolver)):
        times = []
        sizes = []
        for ndt.ref in precision:
            ndt.main()
            A = np.array(ndt.A)
            b = np.array(ndt.b)
            if(selfRMCK[i]):
                r = RCMK(*CSRformat(A)[1:3])
                A = (A[:,r])[r,:]
                b = b[r]
            sizes.append(len(A))
            tic = timer()
            selfSolver[i](A,b)
            tac = timer()
            times.append(tac-tic)
        id = plt.scatter(sizes, times, marker=selfChar[i], color=selfColors[i])
        plotID.append(id)

    plt.legend(plotID, names)
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def complexitylog(precision, save=False, name='complexitylog'):
    ndt.vel = 1
    ndt.freq = 50
    selfColors = [orange, purple, purple]
    selfChar   = ['o', 'o', 'x']
    plus = [0.5,0.25,0]
    names = ['LUsolve', 'LUcsrsolve', 'LUcsrsolve avec RCMK']
    selfSolver = [LUsolve, LUcsrsolve, LUcsrsolve]
    selfRMCK   = [False, False, True]
    plt.Figure()
    plt.tick_params(
    labelleft=False,        # ticks along the top edge are off
    labelbottom=False)
    plt.title('Logarithme des complexités temporelles des différents solvers')
    plt.xlabel('Nombre de noeuds [log$_{10}$]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Temps d\'exécution [log$_{10}$ sec]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plotID = []
    for i in range(len(selfSolver)):
        times = []
        sizes = []
        for ndt.ref in precision:
            ndt.main()
            A = np.array(ndt.A)
            b = np.array(ndt.b)
            if(selfRMCK[i]):
                r = RCMK(*CSRformat(A)[1:3])
                A = (A[:,r])[r,:]
                b = b[r]
            sizes.append(len(A))
            tic = timer()
            selfSolver[i](A,b)
            tac = timer()
            times.append(tac-tic)
        logx = np.log10(sizes)
        logy = np.log10(times)+plus[i]
        ranger = np.linspace(logx[0], logx[-1], 100)
        fit = np.polyfit(logx,logy,1)
        plt.plot(logx, logy, selfChar[i], color=selfColors[i])
        plt.plot(ranger, np.poly1d(fit)(ranger), '--',color=selfColors[i])
        plotID.append(fit)
    plt.legend([names[0],'Polyfit LUsolve : %.2fx+...'%plotID[0][0],names[1],'Polyfit LUcsrsolve sans RCMK: %.2fx+...'%plotID[1][0], names[2], 'Polyfit LUcsrsolve avec RCMK: %.2fx+...'%plotID[2][0]])
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def accuracy(precision, save=False, name='accuracy'):
    ndt.vel = 0
    ndt.freq = 50
    file = 'res/numpy/%s/%.1f.npy'
    selfColors = [purple, purple, orange]
    selfChar   = ['--x', '--o', '--o']
    names = ['LUcsrsolve','LUcsrsolve avec RCMK', 'Numpy linalg']
    selfSolver = ['LUcsrsolve', 'LUcsrsolveRCMK','numpy']
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.set_ylim([0,10e-13])
    plt.title('Précision des solvers')
    plt.xlabel('Nombre de noeuds')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Précision')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plotID = []
    for i in range(len(selfSolver)):
        acc   = []
        sizes = []
        for ndt.ref in precision:
            A = np.load(file%('A/%s'%selfSolver[i], ndt.ref))
            b = np.load(file%('b/%s'%selfSolver[i], ndt.ref))
            x = np.load(file%('x/%s'%selfSolver[i], ndt.ref))
            sizes.append(len(A))
            acc.append(np.linalg.norm(A@x - b,ord=2)/np.linalg.norm(b,ord=2))
        plt.plot(sizes, acc, selfChar[i], color=selfColors[i])

    plt.legend(names)
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

def backward(precision, save=False, name='backward'):
    ndt.vel = 0
    ndt.freq = 50
    selfColors = [purple, purple, orange]
    selfChar   = ['--x', '--o','--o']
    names = ['LUcsrsolve','LUcsrsolve avec RCMK', 'Numpy linalg', 'Epsilon machine']
    selfSolver = ['LUcsrsolve', 'LUcsrsolveRCMK', 'numpy']
    selfRMCK   = [False, True, False]
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.set_ylim([0,10e-13])
    plt.title('Stabilité des solvers')
    plt.xlabel('Nombre de noeuds')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Stabilité')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plotID = []
    for i in range(len(selfSolver)):
        acc   = []
        sizes = []
        for ndt.ref in precision:
            print(selfSolver[i])
            A = np.load(file%('A/%s'%selfSolver[i], ndt.ref))
            b = np.load(file%('b/%s'%selfSolver[i], ndt.ref))
            x = np.load(file%('x/%s'%selfSolver[i], ndt.ref))
            sizes.append(len(A))
            acc.append(np.linalg.norm(b - A@x,ord=2)/(np.linalg.norm(x,ord=2)*np.linalg.norm(A, ord=2)))
        plt.plot(sizes, acc, selfChar[i], color=selfColors[i])
    plt.plot(np.linspace(sizes[0],sizes[-1],100), [np.finfo(float).eps]*100, '-', color='Grey')

    plt.legend(names)
    if save : plt.savefig('res/plots/%s.png'%(name))
    else : plt.show()
    return

if __name__=='__main__':
    big_acc = np.linspace(1,5,10)
    lil_acc = np.linspace(1,2,3)

    denseLU(big_acc, save=False)
    # denseA(big_acc, save=False)
    # band(big_acc, save=False)
    # CSR(big_acc, save=False)
    # complexity(big_acc, save=False)
    # complexitylog(big_acc, save=False)
    # accuracy(big_acc, save=False)
    # backward(big_acc, save=False)
