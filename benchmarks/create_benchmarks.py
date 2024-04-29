import pyvotess
from scipy.spatial import Voronoi

import numpy as np
import time

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

N_arr = [1000, 10000, 100000]
iterations = 10

begin_slope_index = 1

k = 160
grid_resolution = 32

outdir = "plots/"
ext = "svg"

# --------------------------------------------------------------------------- #
# Data Generation
# --------------------------------------------------------------------------- #

outfile = open("profile.dat","w")

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
    outfile.write(f"{N} : {average_time})

    measured_times = []
    for iter in range(iterations + 1):
        t1 = time.time()
        Voronoi(xyzset)
        t2 = time.time()
        if iter == 1: continue # throw away pre-jit optimized run
        measured_times.append(t2 - t1)

    qhull_times.append(sum(measured_times) / iterations)

def calculate_slope(x, y, start_index):
    x_log = np.log(x[start_index:])
    y_log = np.log(y[start_index:])
    slope = (y_log[-1] - y_log[0]) / (x_log[-1] - x_log[0])
    return slope
  
slope_votess = calculate_slope(N_arr, votess_times, begin_slope_index)
slope_qhull  = calculate_slope(N_arr, qhull_times,  begin_slope_index)

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
from itertools import cycle

# --------------------------------------------------------------------------- #

plt.style.use('classic')
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(12, 8))
fig.set_facecolor('floralwhite') 

markers = cycle(['o', 's', '^', 'D', 'p', '*', 'X'])

ax.plot(N_arr, votess_times, label=f"votess",
        marker=next(markers), linestyle='-', linewidth=2, markersize=8,
        color='k')
ax.plot(N_arr, qhull_times, label=f"qhull",
        marker=next(markers), linestyle='-', linewidth=2, markersize=8,
        color='b')

ax.set_title('Performance Comparision', 
             fontsize=20, fontweight='bold')
ax.set_xlabel('dataset size', fontsize=16)
ax.set_ylabel('computation time (s)', fontsize=16)

ax.legend(title='Algorithm',
          title_fontsize='14', fontsize='12', loc='upper left', 
          bbox_to_anchor=(1, 1),
          fancybox=True)

ax.grid(True, which='both', linestyle='--', linewidth=1.0, color='k')
ax.tick_params(axis='both', which='major', labelsize=10) 

plt.tight_layout(rect=[0, 0, 0.98, 1])
plt.savefig(outdir + "plot" + '.' + ext, facecolor=fig.get_facecolor(),
            dpi=300)

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(12, 8))
fig.set_facecolor('floralwhite')

ax.plot(N_arr, votess_times, label="votess",
          marker=next(markers), linestyle='-', linewidth=2, markersize=8,
          color='k')
ax.plot(N_arr, qhull_times, label="qhull",
          marker=next(markers), linestyle='-', linewidth=2, markersize=8,
          color='b')

ax.set_xscale('log')
ax.set_yscale('log')

out_of_box=0.20
plt.xlim(N_arr[0] * ( 1 - out_of_box), N_arr[-1] * (1 + out_of_box))

ax.set_title('Performance comparision (Log-Log scale)', 
             fontsize=20, fontweight='bold')
ax.set_xlabel('Dataset Size', fontsize=16)
ax.set_ylabel('Computation Time (s)', fontsize=16)

ax.legend(title='Algorithm',
          title_fontsize='14', fontsize='12', loc='upper left', 
          bbox_to_anchor=(1, 1),
          fancybox=True)

ax.grid(True, which='major', linestyle='--', linewidth=1.0, color='k')
ax.tick_params(axis='both', which='major', labelsize=10)

x_offs = 1.01
y_offs = 1.2
ax.text(N_arr[0] * x_offs, votess_times[0] * y_offs, f"{slope_votess:.2f}",
        fontsize=12, ha='center', color = 'k')
ax.text(N_arr[0] * x_offs, qhull_times[0] * y_offs, f"{slope_qhull:.2f}", 
        fontsize=12, ha='center', color='k')

plt.tight_layout(rect=[0, 0, 0.98, 1])
plt.savefig(outdir + "logplot" + '.' + ext, facecolor=fig.get_facecolor(),
            dpi=300)

# --------------------------------------------------------------------------- #
