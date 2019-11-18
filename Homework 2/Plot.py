#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date   : Monday, November 18th 2019
"""
# ======================= Import ==============================
# In[0]
import mysolve as my
import ndt
import time
import numpy as np
import matplotlib.pyplot as plt

# Purple, Orange, Blue
COLORS = ['#9400D3', '#FFA500', '#0080FF']

# ======================= SETTERS ==============================
# In[1]

def setVariables(ref=1, gap=0.2*ndt.cm, freq=0, vel=0, mur=100):
    ndt.ref  = ref
    ndt.gap  = gap
    ndt.freq = freq
    ndt.vel  = vel
    ndt.mur  = mur
    return

def setSolver(SOLVER=None):
    my.SOLVER = SOLVER

def ILU(A):
    return np.linalg.inv(my.ILU0(A)[0])@A

def getA(A):
    return A

def timer(SOLVER, **kwargs):
    start = time.time()
    setSolver(SOLVER)
    ndt.main(**kwargs)
    return time.time()-start

# ======================= PLOTERS ==============================

def plot_ref(solver='ALL', mean=1, save=False):
    refs   = (1, 2, 3, 4, 5)
    if solver=='ALL' :
        solvers = ['Cholesky','LU', 'QR']
        name = 'complexiteLQC'
    elif solver=='LU':
        solvers = ['LU']
        name = 'complexiteLU'
    elif solver=='QR':
        solvers = ['QR']
    elif solver=='Cholesky':
        solvers = ['Cholesky']
        name = 'complexiteQR'
    else:
        exit()
    plots_id      = []
    plots_solvers = []
    plt.Figure(figsize=(20,20))
    plt.xlabel('log(Nombre de noeuds)')
    plt.ylabel('Temps [log(sec)]')
    plt.yscale('log')
    plt.xscale('log')
    for s in range(len(solvers)):
        setSolver(solvers[s])
        mean_times = []
        x  = []
        for ref in refs:
            print('\n Solver %s : ref %d'%(solvers[s], ref))
            times = []
            setVariables(ref=ref, vel=0, freq=0)
            for m in range(mean):
                times.append(timer(SOLVER=solvers[s], show=False, test=False))
            mean_times.append(sum(times)/len(times))
            x.append(ndt.Nodes)
        p = plt.scatter(x, mean_times, color=COLORS[s])
        plots_id.append(p)
        plots_solvers.append(solvers[s])
    plt.legend(plots_id, plots_solvers)
    if save : plt.savefig('res/plots/%s.png'%(name))
    plt.show()
    return

def plot_regimes(solver='ALL', mean=1, ref=1, save=False):
    params   = ((0,0), (0,50), (50,0), (50,50))
    regimes  = ['Statique', 'Stationnaire', 'Harmonique', 'Dynamique']
    if solver=='ALL' :
        solvers = ['LU', 'QR']
        name = 'regimes'
    elif solver=='LU':
        solvers = ['LU']
        name = 'regimesLU'
    elif solver=='QR':
        solvers = ['QR']
        name = 'regimesQR'
    else:
        exit()
    plots_id      = []
    plots_solvers = []
    plt.Figure(figsize=(20,20))
    plt.xlabel('Régime')
    plt.ylabel('Temps [sec]')
    for s in range(len(solvers)):
        setSolver(solvers[s])
        mean_times = []
        for param in params:
            times = []
            setVariables(ref=ref, vel=param[0], freq=param[1])
            for m in range(mean):
                times.append(timer(SOLVER=solvers[s], show=False, test=False))
            mean_times.append(sum(times)/len(times))
        p = plt.scatter(regimes, mean_times, color=COLORS[s])
        plots_id.append(p)
        plots_solvers.append(solvers[s])
    plt.legend(plots_id, plots_solvers)
    if save : plt.savefig('res/plots/%s.png'%(name))
    plt.show()
    return

def plot_ilu(ref=1, save=False):
    name = 'conditionnementILU'
    params   = ((0,0), (0,50), (50,0), (50,50))
    regimes  = ['Statique', 'Stationnaire', 'Harmonique', 'Dynamique']
    func     = [ILU,getA]
    plots_names   = ['M$^{{{}}}$A'.format(-1),'A']
    plots_id      = []
    plt.Figure(figsize=(20,20))
    plt.xlabel('Régime')
    plt.yscale('log')
    plt.ylabel('Nombre de conditionnement')
    for s in range(len(func)):
        k = []
        for param in params:
            setVariables(ref=ref, vel=param[0], freq=param[1])
            ndt.main(show=False, test=False, solver=False)
            A = func[s](ndt.A)
            [min,max] = my.get_min_max_singular_values(A)
            k.append(max/min)
        p = plt.scatter(regimes, k, color=COLORS[s])
        plots_id.append(p)
    plt.legend(plots_id, plots_names)
    if save : plt.savefig('res/plots/%s.png'%(name))
    plt.show()
    return

def plot_accuracy(ref=1, save=False):
    name = 'accuracy'
    params   = ((0,0), (0,50), (50,0), (50,50))
    regimes  = ['Statique', 'Stationnaire', 'Harmonique', 'Dynamique']
    solvers  = ['LU', 'QR', 'Numpy']
    plots_id      = []
    plt.Figure(figsize=(20,20))
    plt.title('Précision')
    plt.xlabel('Régime')
    # plt.yscale('log')
    plt.ylabel('Précision')
    plt.ylim(bottom=10e-16, top=2.8*10e-15)
    for s in range(len(solvers)):
        acc = []
        for param in params:
            setVariables(ref=ref, vel=param[0], freq=param[1])
            setSolver(solvers[s])
            ndt.main(show=False, test=False, solver=True)
            b = ndt.b
            A = np.array(ndt.A)
            num = np.linalg.norm(A@ndt.x-b)
            den = np.linalg.norm(b)
            acc.append((num/den))
            print("\n\nWOUHOU : %s : %.15f\n\n"%(solvers[s], (num/den)*10e13))
        p = plt.scatter(regimes, acc, color=COLORS[s])
        plots_id.append(p)
    plt.legend(plots_id, solvers)
    if save : plt.savefig('res/plots/%s.png'%(name))
    plt.show()
    return

if __name__=='__main__':
    Solvers = ['LU', 'QR', 'Cholesky']
    ref = 3
    setVariables(ref=ref)
    for s in range(len(Solvers)):
        print('%.2f secondes pour résoudre le système avec %s et un raffinement de %d'%(timer(Solvers[s], show=False),Solvers[s],ref))
        
    # plot_regimes(mean=2, ref=2)
    # plot_ref()
    # plot_ilu()
    # plot_ilu()
    # plot_accuracy()
    # timer("LU")
