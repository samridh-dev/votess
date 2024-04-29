# --------------------------------------------------------------------------- #

import pyvotess

import numpy as np
import time

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

N_arr = [1000, 10000, 100000]
iterations = 10

begin_slope_index = 1

k = 72
grid_resolution = 32

fname = "profile/nprof.dat"

# --------------------------------------------------------------------------- #
# Data Generation
# --------------------------------------------------------------------------- #

outfile = open(nprof,"w")

votess_times = []
qhull_times = []

for N in N_arr:
    xyzset = np.random.random((N,3))

    measured_times = []

    for iter in range(iterations + 1):
        t1 = time.time()
        vtargs = pyvotess.vtargs(k, grid_resolution)
        result = pyvotess.tesellate(xyzset, vtargs)
        t2 = time.time()

        if iter == 1: continue # throw away pre-jit optimized run
        measured_times.append(t2 - t1)

    average_time = sum(measured_times) / iterations
    votess_times.append(average_time)
    outfile.write(f"{N} : {average_time}\n")

# --------------------------------------------------------------------------- #
