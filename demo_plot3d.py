"""
Read in 3d magnetic field datacube and plot some 3d field lines.
Also compute potential reference field and plot that alongside, for the same starting points.

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
b = BField('bhesse.nc')
    
# Set up 3d plot:
# ===============
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim(b.x1[0], b.x1[-1])
ax.set_ylim(b.y1[0], b.y1[-1])
ax.set_zlim(b.z1[0], b.z1[-1])

# Choose field line startpoints:
# ==============================
x1 = np.linspace(b.x1[0], b.x1[-1], 8)
y1 = np.linspace(b.y1[0], b.y1[-1], 8)
x1s, y1s = np.meshgrid(x1,y1)
z1s = x1s*0 + b.z1[0]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)

# Trace and plot field lines
# ==========================
print('Tracing %i field lines...' % np.size(x0,1))
tstart = time.time()
xls = b.trace(x0)
for xl in xls:
    ax.plot(xl[:,0], xl[:,1], xl[:,2])
print('...tracing time: %g sec' % (time.time() - tstart))

ax.set_title(r'${\bf B}$')

# Compute reference potential field:
# ==================================
b.computePotentialField()

# Plot the field lines of this reference field, from the same starting points:
# ============================================================================
ax = fig.add_subplot(122, projection='3d')
ax.set_xlim(b.x1[0], b.x1[-1])
ax.set_ylim(b.y1[0], b.y1[-1])
ax.set_zlim(b.z1[0], b.z1[-1])
print('Tracing %i field lines...' % np.size(x0,1))
tstart = time.time()
xls = b.trace(x0, potential=True)
for xl in xls:
    ax.plot(xl[:,0], xl[:,1], xl[:,2])
print('...tracing time: %g sec' % (time.time() - tstart))

ax.set_title(r'${\bf B}_{\rm p}$')

plt.savefig('plot3d.png', bbox_inches='tight')
plt.show()

