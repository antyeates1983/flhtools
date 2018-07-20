"""
Read in 3d magnetic field datacube, and compute relative field-line helicity in minimal gauge.

Use this to compute the total relative helicity Hr by integrating over all six boundaries, and compare this to the direct volume integration of Hr (in both DeVore-Coulomb and minimal gauges).

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

# Compute relative helicity directly by volume integration:
# =========================================================
Hr0 = b.relativeHelicity()
print('Hr from volume integration (DeVore-Coulomb upward gauge) = %g' % Hr0)

# Change both vector potentials to minimal gauge:
# ===============================================
# - alternatively, you could just change Ap, then match A to it with b.matchPotentialGauge().
b.matchUniversalGauge()
b.matchUniversalGauge(potential=True)

# Compute relative helicity directly by volume integration:
# =========================================================
Hr = b.relativeHelicity()
print('Hr from volume integration (minimal gauge) = %g' % Hr)

# Compute relative helicity from relative field-line helicity:
# ============================================================
# - map of FLH on lower boundary z = 0:
nx = b.nx
ny = b.ny
x1 = np.linspace(b.x1[0], b.x1[-1], nx)
y1 = np.linspace(b.y1[0], b.y1[-1], ny)
x1s, y1s = np.meshgrid(x1,y1, indexing='ij')
z1s = x1s*0
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)
bz0 = np.abs(b.bz(x1s, y1s, z1s))
flh_z0 = b.flHelicity(x0) - b.flHelicity(x0, potential=True)
flh_z0 = flh_z0.reshape(nx, ny)
# - map of FLH on top boundary z = 2:
z1s = x1s*0 + b.z1[-1]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)
bz1 = np.abs(b.bz(x1s, y1s, z1s)) 
flh_z1 = b.flHelicity(x0) - b.flHelicity(x0, potential=True)
flh_z1 = flh_z1.reshape(nx, ny)
# - map of FLH on right boundary x = 1:
ny = b.ny
nz = b.nz
y1 = np.linspace(b.y1[0], b.y1[-1], ny)
z1 = np.linspace(b.z1[0], b.z1[-1], nz)
y1s, z1s = np.meshgrid(y1,z1, indexing='ij')
x1s = y1s*0 + b.x1[-1]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)
bx1 = np.abs(b.bx(x1s, y1s, z1s)) 
flh_x1 = b.flHelicity(x0) - b.flHelicity(x0, potential=True)
flh_x1 = flh_x1.reshape(ny, nz)
# - map of FLH on left boundary x = -1:
x1s = y1s*0 + b.x1[0]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)
bx0 = np.abs(b.bx(x1s, y1s, z1s))
flh_x0 = b.flHelicity(x0) - b.flHelicity(x0, potential=True)
flh_x0 = flh_x0.reshape(ny, nz)    
# - map of FLH on top boundary y = 1:
nx = b.nx
nz = b.nz
x1 = np.linspace(b.x1[0], b.x1[-1], nx)
z1 = np.linspace(b.z1[0], b.z1[-1], nz)
x1s, z1s = np.meshgrid(x1,z1, indexing='ij')
y1s = x1s*0 + b.y1[-1]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)
by1 = np.abs(b.by(x1s, y1s, z1s))
flh_y1 = b.flHelicity(x0) - b.flHelicity(x0, potential=True)
flh_y1 = flh_y1.reshape(nx, nz)
# - map of FLH on bottom boundary y = -1:
y1s = x1s*0 + b.y1[0]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)
by0 = np.abs(b.by(x1s, y1s, z1s))
flh_y0 = b.flHelicity(x0) - b.flHelicity(x0, potential=True)
flh_y0 = flh_y0.reshape(nx, nz)
# - estimate total helicity by adding FLH over all field lines:
# - Add up all field lines and divide by two (each should be counted twice).
hf = np.trapz(np.trapz(flh_z0*bz0, x=x1, axis=0), x=y1, axis=0)
hf += np.trapz(np.trapz(flh_z1*bz1, x=x1, axis=0), x=y1, axis=0)
hf += np.trapz(np.trapz(flh_x0*bx0, x=y1, axis=0), x=z1, axis=0)  
hf += np.trapz(np.trapz(flh_x1*bx1, x=y1, axis=0), x=z1, axis=0)
hf += np.trapz(np.trapz(flh_y0*by0, x=x1, axis=0), x=z1, axis=0)   
hf += np.trapz(np.trapz(flh_y1*by1, x=x1, axis=0), x=z1, axis=0)
hf /= 2

print('Hr from relative FLH (minimal gauge) = %g' % hf)

