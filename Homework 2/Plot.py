import mysolve as my
import matplotlib.pyplot as plt
import ndt
import time
import copy

# Purple, Orange, Blue
COLORS = ['#9400D3', '#FFA500', '#0080FF']

def setVariables(ref=1, gap=0.2*ndt.cm, freq=0, vel=0, mur=100):
    ndt.ref  = ref
    ndt.gap  = gap
    ndt.freq = freq
    ndt.vel  = vel
    ndt.mur  = mur
    return

def setSolver(SOLVER):
    my.SOLVER = SOLVER

def timer(SOLVER, **kwargs):
    start = time.time()
    setSolver(SOLVER)
    print("COUCOU",SOLVER)
    ndt.main(**kwargs)
    return time.time()-start

def plot_ref(solver='ALL', mean=1, save=False):
    # refs   = (1, 2, 3, 4, 5)
    refs = (1,2)
    if solver=='ALL' :
        solvers = ['LU', 'QR']
        name = 'complexite'
    elif solver=='LU':
        solvers = ['LU']
        name = 'complexiteLU'
    elif solver=='QR':
        solvers = ['QR']
        name = 'complexiteQR'
    else:
        exit()
    plots_id      = []
    plots_solvers = []
    plt.Figure(figsize=(20,20))
    for s in range(len(solvers)):
        setSolver(solvers[s])
        mean_times = []
        x  = []
        for ref in refs:
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

# plot_regimes(mean=2, ref=2, save=True)
# plot_ref(save=True)
timer("LU")






# e, t = plot_ref("QR")
# print("Times : ", t)
# print("Elems : ", e)

# timer(SOLVER='LU', show=True, test=True)
# regimes = ['Statique', 'Stationnaire', 'Harmonique', 'Dynamique']
# markers = ['o', 's', 'x', 'D']
# params  = ((0,0), (0,50), (50,0), (50,50))
# params = (((0,0), (0,50), (50,0), (50,50)),((10,10), (10,60), (60,10), (60,60)))
# solvers = ['LU','QR']
# plot_solvers = []
# fig=plt.Figure()
# for s in range(2):
#     # setSolver(SOLVER)
#     plot_regimes = []
#     for r in range(4):
#         regime = regimes[r]
#         p = plt.scatter([params[s][r][0]], [params[s][r][1]], marker=markers[r], color=COLORS[s])
#         plot_regimes.append(copy.copy(p))
#     plot_solvers.append(plot_regimes)
# for r in plot_solvers[0]:
#     r.set_color('black')
# legend_regimes = plt.legend(plot_solvers[0], regimes, loc=1)
# plt.legend([l[0] for l in plot_solvers], solvers, loc=4)
# plt.gca().add_artist(legend_regimes)
# plt.show()


# # pyplot.legend([l for l in plot_lines], parameters, loc=4)
# pyplot.gca().add_artist(legend1)
# plt.show()
