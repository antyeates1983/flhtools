"""
Read in 3d magnetic field datacube, and plot field line helicity of original field B and reference field Bp in DeVore-Coulomb gauge (integrating upward). Also plot the relative field-line helicity (i.e. their difference) for this gauge.


anthony.yeates@durham.ac.uk
"""
import numpy as np
from flhcart import BField
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import time

# Create 3d magnetic field object:
# ================================
# - creating this object automatically computes A in the DeVore-Coulomb gauge.
b = BField('bhesse.nc')
    
# Compute potential reference field and its vector potential:
# ===========================================================
b.computePotentialField()
b.computeADeVore(potential=True)

# Change the gauge of A so that n x A = n x Ap on the boundary:
# =============================================================
b.matchPotentialGauge()
    
# Choose grid of field lines at height z = z0:
# ============================================
z0 = 0
nx = 128
ny = 128
x1 = np.linspace(b.x1[0], b.x1[-1], nx)
y1 = np.linspace(b.y1[0], b.y1[-1], ny)
x1s, y1s = np.meshgrid(x1,y1)
z1s = x1s*0 + z0
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)

# Trace field lines and compute field-line helicity of B and Bp:
# ==============================================================
flh = b.flHelicity(x0)
flh = flh.reshape(nx, ny)
flhp = b.flHelicity(x0, potential=True)
flhp = flhp.reshape(nx, ny)

rflh = flh - flhp

# Plots:
# ======
fig = plt.figure(figsize=(12, 5))

# - FLH of B:
ax = fig.add_subplot(131)
ax.set_xlim(b.x1[0], b.x1[-1])
ax.set_ylim(b.y1[0], b.y1[-1])
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('FLH of B, z = %g' % z0)
pm = ax.pcolormesh(x1, y1, flh, cmap='bwr')
cmax = np.max(np.abs(flh))
pm.set_clim(vmin=-cmax, vmax=cmax)
plt.colorbar(pm)

# - FLH of Bp:
ax = fig.add_subplot(132)
ax.set_xlim(b.x1[0], b.x1[-1])
ax.set_ylim(b.y1[0], b.y1[-1])
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('FLH of Bp, z = %g' % z0)
pm = ax.pcolormesh(x1, y1, flhp, cmap='bwr')
cmax = np.max(np.abs(flh))
pm.set_clim(vmin=-cmax, vmax=cmax)
plt.colorbar(pm)

# - Relative FLH:
ax = fig.add_subplot(133)
ax.set_xlim(b.x1[0], b.x1[-1])
ax.set_ylim(b.y1[0], b.y1[-1])
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('relative FLH, z = %g' % z0)
pm = ax.pcolormesh(x1, y1, rflh, cmap='bwr')
cmax = np.max(np.abs(rflh))
pm.set_clim(vmin=-cmax, vmax=cmax)
plt.colorbar(pm)

plt.savefig('flhdevore.png', bbox_inches='tight')
plt.show()

