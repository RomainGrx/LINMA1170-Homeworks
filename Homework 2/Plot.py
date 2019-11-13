import mysolve as my
import ndt
import time

def setVariables(ref=1, gap=0.2*ndt.cm, freq=0, vel=0, mur=100):
    ndt.ref  = ref
    ndt.gap  = gap
    ndt.freq = freq
    ndt.vel  = vel
    ndt.mur  = mur
    return

def setSolver(SOLVER):
    my.SOLVER = SOLVER

def timer():
    start = time.time()
    ndt.main(show=False)
    return time.time()-start


setSolver("LU")
print(timer())
