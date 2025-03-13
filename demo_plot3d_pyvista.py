"""
Read in 3d magnetic field datacube and plot some 3d field lines.

** This script uses the PYVISTA library instead of matplotlib. **

anthony.yeates@durham.ac.uk
"""
import numpy as np
from flhcart import BField
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import time
import pyvista as pv

# Create 3d magnetic field object:
# ================================
b = BField('bhesse.nc')

p = pv.Plotter()

# Create mesh for lower boundary, with contour plot of Bz:
# ========================================================
b0 = 1 # max of colour scale for Bz.
x, y = np.meshgrid(b.x1, b.y1)
bz0 = b.bz(x, y, x*0 + b.z1[0])
surf = pv.Plane(i_resolution=b.nx, j_resolution=b.ny, i_size=b.x1[-1]-b.x1[0], j_size=b.y1[-1]-b.y1[0], center=(0.5*(b.x1[0]+b.x1[-1]), 0.5*(b.y1[0]+b.y1[-1]), b.z1[0]))
p.add_mesh(surf, show_edges=False, scalars=bz0, cmap='gray', clim=[-b0, b0])
p.remove_scalar_bar()

# Choose field line startpoints:
# ==============================
x1 = np.linspace(b.x1[0], b.x1[-1], 8)
y1 = np.linspace(b.y1[0], b.y1[-1], 8)
x1s, y1s = np.meshgrid(x1,y1)
z1s = x1s*0 + b.z1[0]
x0 = np.stack((x1s.flatten(), y1s.flatten(), z1s.flatten()), axis=1)

# Trace field lines:
# ==================
xls = b.trace(x0)
for k, xl0 in enumerate(xls):
    p.add_lines(xl0, width=2, connected=True, color='tab:red')

p.add_axes()
p.show()

# Output to file:
plt.figure(figsize=(6,6))
plt.subplot(111)
plt.imshow(p.image)
p.close()
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('plot3d_pyvista.png', bbox_inches='tight', dpi=300)