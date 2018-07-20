"""
Python script to create a netcdf file for an analytical magnetic field.

anthony.yeates@durham.ac.uk
"""
import numpy as np
from scipy.io import netcdf

# "Hesse" field:

def bx(x, y, z):
    return x*0 - 2

def by(x, y, z, t=2):
    return -z - t*(1 - z**2)/(1 + z**2/25)**2/(1 + x**2/25)

def bz(x, y, z):
    return y

  
# Define the grid:
nx = 64
ny = 64
nz = 64
x1 = np.linspace(-20, 20, nx)
y1 = np.linspace(-20, 20, ny)
z1 = np.linspace(0, 40, nz)
z, y, x = np.meshgrid(z1, y1, x1, indexing='ij')

# Write to netcdf file:
fname = 'bhesse.nc'
f = netcdf.netcdf_file(fname, 'w')
f.createDimension('xdim', nx)
f.createDimension('ydim', ny)
f.createDimension('zdim', nz)
xv = f.createVariable('x', 'f', ('xdim',))
yv = f.createVariable('y', 'f', ('ydim',))
zv = f.createVariable('z', 'f', ('zdim',))
xv[:], yv[:], zv[:] = x1, y1, z1
bxv = f.createVariable('bx', 'f', ('zdim','ydim','xdim'))
byv = f.createVariable('by', 'f', ('zdim','ydim','xdim'))
bzv = f.createVariable('bz', 'f', ('zdim','ydim','xdim'))
bxv[:,:,:] = bx(x, y, z)
byv[:,:,:] = by(x, y, z)
bzv[:,:,:] = bz(x, y, z)
f.close()

print('Wrote file '+fname)
