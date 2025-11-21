import numpy as np
import trimesh
from gyroid_gen import (
    fn_plot_tpms_eq,
    fn_check_face_normals,
    fn_flip_face_normals,
    fn_generate_mesh,
    fn_export_stl_file
)

# TPMS type: 'Shell' or 'Skeletal'
tpms_type = 'Shell'

# TPMS design options:
# For Shell: 'Shell-TPMS Gyroid', 'Shell-TPMS Diamond', 'Shell-TPMS Lidinoid', 
#            'Shell-TPMS Split-P', 'Shell-TPMS Schwarz'
# For Skeletal: 'Skeletal-TPMS Schoen gyroid', 'Skeletal-TPMS Schwarz diamond',
#               'Skeletal-TPMS Schwarz primitive', 'Skeletal-TPMS Body diagonals with nodes'
tpms_design = 'Shell-TPMS Gyroid'

# Size of the structure in each dimension (mm)
sizes = [20.0, 20.0, 20.0]  # [x, y, z]

# Size of one unit cell in each dimension (mm)
cell_sizes = [4.0, 4.0, 4.0]  # [x, y, z]

# Origin offset (mm)
origin = [0.0, 0.0, 0.0]

# Mesh resolution (points per unit cell)
unit_cell_mesh_resolution = 50  # Higher = smoother but slower (30-100 recommended)

# Threshold parameter (controls volume fraction)
c = 0.0  # Typically -1.0 to 1.0, 0.0 is default

# Thickness for shell structures (0-1, controls wall thickness)
thickness = 0.3  # Only used for Shell type

# Whether to flip face normals
flip_face_normals = False

# Initial empty mesh (will be created by the functions)
mesh = trimesh.Trimesh()

# Export settings
export_stl = True
file_name = 'gyroid_output'
directory_path = '.'  # Current directory

# Visualization settings (set to False to skip plots and avoid segfaults)
show_plots = False  # Set to True if you want to see visualizations

if __name__ == "__main__":
    print(f"Type: {tpms_type}")
    print(f"Design: {tpms_design}")
    print(f"Size: {sizes} mm")
    print(f"Cell size: {cell_sizes} mm")
    print(f"Resolution: {unit_cell_mesh_resolution} points/unit cell")
    print(f"Show plots: {show_plots}")
    
    # Plot the TPMS equation (optional - for visualization)
    mesh, vertices = fn_plot_tpms_eq(
        tpms_type, tpms_design, sizes, cell_sizes, origin,
        unit_cell_mesh_resolution, c, thickness, mesh, show_plot=show_plots
    )
    mesh = fn_check_face_normals(mesh, silent=False, show_plot=show_plots)
    
    # Generate the mesh
    final_mesh = fn_generate_mesh(
        tpms_type, tpms_design, c, thickness, sizes, cell_sizes,
        origin, unit_cell_mesh_resolution, mesh, flip_face_normals, silent=False
    )
    
    # Export to STL file
    if export_stl:
        fn_export_stl_file(final_mesh, file_name, directory_path, silent=False)
