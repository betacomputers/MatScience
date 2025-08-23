# MatScience

This project generates 3D models of mathematically and biologically inspired structures for engineering applications. The codebase includes scripts for creating coral-like fractal geometries and gyroid periodic, suitable for 3D printing, simulation, or visualization.

## Features

- **Coral Structure Generation**: `coral_v1.py` and `coral_v2.py` create coral-like fractal structures using recursive branching algorithms. Outputs an STL file for simulation and 3D printing, and a png for visualization.
- **Gyroid Surface Generation**: `gyroid_v1.py` generates a gyroid shell using implicit surface equations and exports the result as an STL mesh.

## Requirements

- Python 3.7+
- NumPy
- scikit-image
- matplotlib
- numpy-stl

## File Overview

- `coral_v1.py`: Basic coral fractal generator with visualization.
- `coral_v2.py`: Advanced coral generator with improved branch distribution and shell enclosure.
- `gyroid_v1.py`: Gyroid TPMS shell generator.

## Output

- STL files: `_coral.stl`, `_coral_v2.stl`, `_gyroid.stl`
- PNG files: `_coral_visualization.png`, `_coral_v2_visualization.png`

## Customization

### gyroid_v1.py Parameters

- `cell_size_mm`: Size of one gyroid unit cell in mm. Larger values create bigger repeating patterns.
- `num_cells`: Number of unit cells along each axis. More cells increase the overall size and complexity.
- `wall_thickness`: Thickness (mm) of the gyroid shell. Larger values yield thicker walls.
- `isovalue`: Threshold for the gyroid surface. Adjusting this shifts the surface location.
- `points_per_cell`: Grid resolution per cell. Higher values produce smoother, more detailed surfaces but increase computation time and file size.
- `scale`: Overall scaling factor for the output geometry.

### coral_v1.py Parameters

- `container_size`: Size of the cubic container in mm. Larger values create bigger structures.
- `resolution`: Number of grid points per dimension. Higher values yield smoother, more detailed geometry but increase computation time and file size.
- `max_iterations`: Maximum recursion depth for branching. Higher values produce more complex, bushier coral.
- `branch_probability`: Probability of creating sub-branches at each recursion. Higher values result in more branches and denser coral.
- `branch_angle`: Angle (in radians) between branches. Larger angles create more spread-out branches; smaller angles yield denser packing.
- `branch_length_factor`: Fraction by which each sub-branch is shorter than its parent. Lower values make branches shrink faster, resulting in more compact coral.
- `branch_thickness_factor`: Fraction by which each sub-branch is thinner than its parent. Lower values make branches thin out more quickly.
- `base_thickness`: Thickness (mm) of the main branches. Larger values create chunkier coral.
- `num_initial_branches`: Number of main branches from the center. More branches improve distribution and complexity.

### coral_v2.py Parameters

- All parameters from `coral_v1.py`, plus:
- `num_additional_start_points`: Number of extra starting points for branches, distributed throughout the cube. More points increase coverage and complexity.
- Improved branch distribution using Fibonacci sphere and additional randomization for more natural appearance.
- Adds a solid shell around the perimeter of the cube (`shell_thickness` is derived from container size and resolution).
