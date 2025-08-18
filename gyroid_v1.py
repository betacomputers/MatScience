import numpy as np
from skimage import measure
from stl import mesh

# Parameters
cell_size_mm = 5.0  # TPMS unit cell length (mm)
num_cells = 4   # Number of cells along each axis
wall_thickness = 1.4    # mm
isovalue = 0.0  # Gyroid threshold surface
scale = 1.0 # Overall scaling factor

# Resolution: higher = smoother mesh but heavier file
points_per_cell = 50

# Create grid
L = cell_size_mm * num_cells
res = points_per_cell * num_cells
x = np.linspace(0, L, res)
y = np.linspace(0, L, res)
z = np.linspace(0, L, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Convert mm to radians for TPMS math
Xr = 2 * np.pi * X / cell_size_mm
Yr = 2 * np.pi * Y / cell_size_mm
Zr = 2 * np.pi * Z / cell_size_mm

# Gyroid implicit function
F = np.sin(Xr) * np.cos(Yr) + np.sin(Yr) * np.cos(Zr) + np.sin(Zr) * np.cos(Xr)

# Make shell by thickening surface
outer_surface = F - isovalue
inner_surface = F - isovalue - (wall_thickness / cell_size_mm) * (2 * np.pi)

# Boolean mask of shell region
mask = (outer_surface >= 0) & (inner_surface <= 0)

# Marching cubes mesh
verts, faces, normals, values = measure.marching_cubes(mask.astype(float), 0, spacing=(L/res, L/res, L/res))

# Create STL mesh
gyroid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        gyroid_mesh.vectors[i][j] = verts[f[j], :]

gyroid_mesh.save('gyroid_shell.stl')
print("Saved gyroid_shell.stl")
