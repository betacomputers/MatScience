"""
Gyroid Generator with Porosity Control and Bounding Box Enclosure.

This module implements a comprehensive gyroid lattice generator based on the
structure-generation methodology described in Section 2.1 of "Processing,
Microstructure, and Mechanical Properties of Graded Gyroid Structures
Manufactured by Fused Filament Fabrication" (Materials 2022, 15, 3730).

The generator produces watertight, enclosed gyroid structures suitable for:
- 3D printing (FDM/FFF manufacturing)
- Finite element analysis (FEA) simulations
- Mechanical testing and characterization

Key Features:
1. Implicit surface generation using the gyroid equation (Equation 1)
2. Porosity control with constant or graded profiles in the z-direction
3. Layer-wise threshold adjustment to achieve target porosity within tolerance
4. Bounding box enclosure for structural integrity and watertight meshes
5. STL export with marching cubes surface extraction
6. Interactive visualization with 3D preview, side projection, and cross-sections

The porosity profile can be constant (grad=0) or graded (grad=1) with polynomial
degrees of 0 (constant), 1 (linear), or 2 (quadratic) to create functionally
graded lattice structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage import measure
from stl import mesh


@dataclass
class GyroidParameters:
    """
    Parameter bundle for the graded TPMS generator.
    
    This dataclass encapsulates all parameters needed to generate TPMS
    (Triply Periodic Minimal Surface) lattice structures including:
    - Gyroid (Schoen gyroid)
    - Schwarz (Schwarz P-surface / Primitive)
    - Diamond (Schwarz D-surface)
    - Lidinoid
    - Split-P
    
    All geometric parameters are defined in physical units (millimetres)
    to ensure direct manufacturing compatibility.
    
    Attributes
    ----------
    numx : int
        Number of unit cells repeated along the x-axis.
        Together with unit_cell_size, determines Lx (domain length in x).
    numy : int
        Number of unit cells repeated along the y-axis.
        Together with unit_cell_size, determines Ly (domain length in y).
    numz : int
        Number of unit cells repeated along the z-axis (build direction).
        Together with unit_cell_size, determines Lz (domain length in z).
        This is the direction of porosity grading.
    unit_cell_size : float
        Physical dimension of a single unit cell in millimetres.
        The total domain size is: Lx = numx × unit_cell_size (and similarly for y, z).
    nsteps : int
        Number of voxels used to discretize the TPMS in each direction
        per unit cell. Higher values yield finer resolution but increase
        computation time and mesh complexity. Minimum: 4 (required for gradients).
    porosity_min : float
        Minimum porosity value in the range [0, 1]. For graded structures,
        this is typically achieved at the bottom (z=0) layer.
        0.0 = fully solid, 1.0 = fully void.
    porosity_max : float
        Maximum porosity value in the range [0, 1]. For graded structures,
        this is typically achieved at the top (z=Lz) layer.
        0.0 = fully solid, 1.0 = fully void.
        Must be >= porosity_min.
    grad : int
        Gradient flag: 0 = constant porosity (no grading),
                       1 = graded porosity along z-direction.
        When grad=1, porosity varies from porosity_max (at z=0) to porosity_min
        (at z=Lz) according to func_degree.
    func_degree : int
        Polynomial degree of the porosity gradient function:
        0 = constant (uniform porosity = porosity_max),
        1 = linear gradient (straight line from max to min),
        2 = quadratic gradient (parabolic transition).
        Only used when grad=1.
    delta : float
        Tolerance parameter controlling the maximum allowable deviation
        between actual and target porosity for each layer. Smaller values
        (e.g., 0.02 = 2%) yield more accurate porosity control but increase
        computation time. Typical range: 0.01-0.05.
    smoothness : float
        Standard deviation (in voxels) for optional Gaussian smoothing
        applied to the implicit TPMS field before thresholding. This
        reduces voxel-scale artifacts and stair-stepping. 0.0 = no smoothing.
        Typical range: 0.5-1.0 voxels.
    marching_step : int
        Step size parameter for the marching cubes algorithm. Higher values
        reduce triangle count (lighter mesh) but lower surface quality.
        step_size=1 uses all voxels (maximum quality).
        step_size=2 uses every other voxel (faster, lower resolution).
    wall_thickness : float, optional
        Thickness of the bounding box walls in millimetres. These solid walls
        enclose the TPMS lattice on all 6 faces, making the structure
        watertight and suitable for simulation/printing. Default: 0.5 mm.
        Set to 0.0 to disable bounding box (not recommended for printing).
    tpms_type : str, optional
        Type of TPMS structure to generate. Options:
        - 'gyroid': Schoen gyroid (default)
        - 'schwarz': Schwarz P-surface / Primitive
        - 'diamond': Schwarz D-surface / Diamond
        - 'lidinoid': Lidinoid surface
        - 'split-p': Split-P surface
    """

    numx: int
    numy: int
    numz: int
    unit_cell_size: float  # millimetres
    nsteps: int
    porosity_min: float  # 0–1
    porosity_max: float  # 0–1
    grad: int  # 0 = constant, 1 = graded in z
    func_degree: int  # 0 constant, 1 linear, 2 quadratic
    delta: float  # allowable porosity deviation
    smoothness: float  # Gaussian sigma (voxels)
    marching_step: int
    wall_thickness: float = 0.5  # wall thickness in mm for bounding box
    tpms_type: str = 'gyroid'  # TPMS structure type


# Default parameters for 3x3x3 configuration with 10mm × 10mm × 10mm cells.
# Each cell: 10mm × 10mm × 10mm = 1000 mm³
# Total size: 3 × 10mm = 30mm × 30mm × 30mm
DEFAULT_PARAMS = GyroidParameters(
    numx=3,                    # 3 unit cells in x-direction
    numy=3,                    # 3 unit cells in y-direction
    numz=3,                    # 3 unit cells in z-direction (build direction)
    unit_cell_size=10.0,       # 10.0 mm per unit cell → 30×30×30 mm total domain
    nsteps=50,                 # 50 voxels per unit cell → 150×150×150 voxel grid
    porosity_min=0.30,         # 30% porosity at bottom (z=0)
    porosity_max=0.55,         # 55% porosity at top (z=Lz)
    grad=1,                    # Enable linear porosity gradient
    func_degree=1,             # Linear gradient (degree 1 polynomial)
    delta=0.02,                # 2% porosity tolerance per layer (standard value)
    smoothness=0.8,            # 0.8 voxel Gaussian smoothing for artifact reduction
    marching_step=1,           # Full resolution marching cubes (step_size=1)
    wall_thickness=0.5,        # 0.5 mm bounding box walls for enclosure
)


def validate_params(params: GyroidParameters) -> GyroidParameters:
    """
    Validate and sanitize input parameters with appropriate clamping.
    
    This function performs comprehensive validation of all parameters,
    ensuring they fall within acceptable ranges for stable computation.
    Invalid inputs are clamped to safe defaults or raise descriptive errors.
    
    Parameters
    ----------
    params : GyroidParameters
        Input parameter bundle (may contain invalid values).
    
    Returns
    -------
    GyroidParameters
        Validated and sanitized parameter bundle with all values
        within acceptable ranges.
    
    Raises
    ------
    ValueError
        If nsteps < 4 (insufficient resolution for gradients),
        func_degree not in {0, 1, 2}, or grad not in {0, 1}.
    
    Notes
    -----
    - Porosity values are clamped to [0, 1] and automatically swapped
      if min > max to ensure porosity_min <= porosity_max.
    - delta is clamped to minimum 1e-4 to prevent numerical issues.
    - smoothness and wall_thickness are clamped to >= 0.0.
    - marching_step is clamped to >= 1 (step_size=0 is invalid).
    """
    # Check minimum resolution requirement for gradient computation
    if params.nsteps < 4:
        raise ValueError("nsteps must be at least 4 to support porosity gradients.")
    
    # Validate gradient function degree (must be 0, 1, or 2)
    if params.func_degree not in (0, 1, 2):
        raise ValueError("func_degree must be 0 (constant), 1 (linear), or 2 (quadratic).")
    
    # Validate gradient flag (must be 0 or 1)
    if params.grad not in (0, 1):
        raise ValueError("grad must be 0 (constant porosity) or 1 (graded porosity).")
    
    # Validate TPMS type if present
    if hasattr(params, 'tpms_type'):
        valid_types = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']
        if params.tpms_type.lower() not in valid_types:
            print(f"Warning: Unknown TPMS type '{params.tpms_type}', defaulting to 'gyroid'")
            params.tpms_type = 'gyroid'
    
    # Clamp porosity values to valid range [0, 1]
    poro_min = float(np.clip(params.porosity_min, 0.0, 1.0))
    poro_max = float(np.clip(params.porosity_max, 0.0, 1.0))
    
    # Ensure porosity_min <= porosity_max (swap if necessary)
    if poro_min > poro_max:
        poro_min, poro_max = poro_max, poro_min
    
    # Get TPMS type if present, default to 'gyroid'
    tpms_type = 'gyroid'
    if hasattr(params, 'tpms_type'):
        tpms_type = params.tpms_type.lower()
        valid_types = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']
        if tpms_type not in valid_types:
            tpms_type = 'gyroid'
    
    # Return validated parameter bundle with type coercion and clamping
    return GyroidParameters(
        numx=int(params.numx),                      # Ensure integer cell counts
        numy=int(params.numy),
        numz=int(params.numz),
        unit_cell_size=float(params.unit_cell_size),  # Ensure float physical size
        nsteps=int(params.nsteps),                  # Ensure integer resolution
        porosity_min=poro_min,                      # Clamped [0, 1]
        porosity_max=poro_max,                      # Clamped [0, 1]
        grad=int(params.grad),                      # Ensure integer (0 or 1)
        func_degree=int(params.func_degree),        # Ensure integer (0, 1, or 2)
        delta=float(max(params.delta, 1e-4)),       # Minimum tolerance to avoid div-by-zero
        smoothness=float(max(params.smoothness, 0.0)),  # Non-negative smoothing
        marching_step=int(max(params.marching_step, 1)),  # Minimum step_size=1
        wall_thickness=float(max(params.wall_thickness, 0.0)),  # Non-negative walls
        tpms_type=tpms_type,                        # TPMS structure type
    )


def compute_domain_lengths(params: GyroidParameters) -> Tuple[float, float, float]:
    """
    Compute the physical domain dimensions from cell counts and cell size.
    
    Lx, Ly, Lz are obtained by multiplying the number of unit cells by the
    physical size of each unit cell. These lengths correspond to the actual
    physical dimensions of the generated structure in millimetres.
    
    Parameters
    ----------
    params : GyroidParameters
        Parameter bundle containing numx, numy, numz, and unit_cell_size.
    
    Returns
    -------
    Tuple[float, float, float]
        Physical domain lengths (Lx, Ly, Lz) in millimetres, where:
        - Lx = numx × unit_cell_size (x-direction length)
        - Ly = numy × unit_cell_size (y-direction length)
        - Lz = numz × unit_cell_size (z-direction length, typically build direction)
    
    Examples
    --------
    >>> params = GyroidParameters(numx=5, numy=5, numz=6, unit_cell_size=3.0, ...)
    >>> lx, ly, lz = compute_domain_lengths(params)
    >>> print(f"Domain: {lx}×{ly}×{lz} mm")
    Domain: 15.0×15.0×18.0 mm
    """
    # Calculate total length in each direction: L = num_cells × cell_size
    lx = params.numx * params.unit_cell_size  # Domain length in x (mm)
    ly = params.numy * params.unit_cell_size  # Domain length in y (mm)
    lz = params.numz * params.unit_cell_size  # Domain length in z (mm, build direction)
    return lx, ly, lz


def generate_coordinate_grid(params: GyroidParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a periodic voxel grid covering the design domain."""
    nx = params.numx * params.nsteps
    ny = params.numy * params.nsteps
    nz = params.numz * params.nsteps

    # Exact spacing ensures perfect periodicity
    dx = params.unit_cell_size / params.nsteps
    dy = params.unit_cell_size / params.nsteps
    dz = params.unit_cell_size / params.nsteps
    
    # Generate grid with exact arithmetic
    x = np.array([i * dx for i in range(nx)])
    y = np.array([i * dy for i in range(ny)])
    z = np.array([i * dz for i in range(nz)])
    
    return np.meshgrid(x, y, z, indexing="ij")


def tpms_field(params: GyroidParameters, grid: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Evaluate the implicit TPMS field (normalised to [-1, 1]).
    
    Supports multiple TPMS lattice types: gyroid, schwarz, diamond, lidinoid, split-p.
    
    Parameters
    ----------
    params : GyroidParameters
        Parameters including tpms_type
    grid : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Coordinate grids (x, y, z)
    
    Returns
    -------
    np.ndarray
        Normalized TPMS field values in [-1, 1]
    """
    x, y, z = grid
    # Use unit_cell_size for periodicity to ensure proper repetition
    # Each unit cell should repeat, not the entire domain
    kx = 2.0 * np.pi / params.unit_cell_size
    ky = 2.0 * np.pi / params.unit_cell_size
    kz = 2.0 * np.pi / params.unit_cell_size
    
    # Apply modulo to ALL coordinates for perfect periodicity
    # This ensures the field repeats exactly at cell boundaries
    # Note: This means porosity gradient will also repeat per cell
    x_periodic = x % params.unit_cell_size
    y_periodic = y % params.unit_cell_size
    z_periodic = z % params.unit_cell_size

    # Get TPMS type (case-insensitive, default to gyroid)
    tpms_type = params.tpms_type.lower() if hasattr(params, 'tpms_type') else 'gyroid'
    
    if tpms_type == 'gyroid':
        # Schoen gyroid equation - use periodic coordinates for perfect tiling
        field = (
            np.sin(kx * x_periodic) * np.cos(ky * y_periodic)
            + np.sin(ky * y_periodic) * np.cos(kz * z_periodic)
            + np.sin(kz * z_periodic) * np.cos(kx * x_periodic)
        ) / 3.0
        
    elif tpms_type == 'schwarz':
        # Schwarz P-surface (Primitive) - use periodic coordinates
        field = (
            np.cos(kx * x_periodic) + np.cos(ky * y_periodic) + np.cos(kz * z_periodic)
        ) / 3.0
        
    elif tpms_type == 'diamond':
        # Schwarz D-surface (Diamond) - use periodic coordinates
        field = (
            np.sin(kx * x_periodic) * np.sin(ky * y_periodic) * np.sin(kz * z_periodic)
            + np.sin(kx * x_periodic) * np.cos(ky * y_periodic) * np.cos(kz * z_periodic)
            + np.cos(kx * x_periodic) * np.sin(ky * y_periodic) * np.cos(kz * z_periodic)
            + np.cos(kx * x_periodic) * np.cos(ky * y_periodic) * np.sin(kz * z_periodic)
        ) / 4.0
        
    elif tpms_type == 'lidinoid':
        # Lidinoid surface - use periodic coordinates
        field = (
            np.sin(2 * kx * x_periodic) * np.cos(ky * y_periodic) * np.sin(kz * z_periodic)
            + np.sin(kx * x_periodic) * np.sin(2 * ky * y_periodic) * np.cos(kz * z_periodic)
            + np.cos(kx * x_periodic) * np.sin(ky * y_periodic) * np.sin(2 * kz * z_periodic)
            - np.cos(2 * kx * x_periodic) * np.cos(2 * ky * y_periodic)
            - np.cos(2 * ky * y_periodic) * np.cos(2 * kz * z_periodic)
            - np.cos(2 * kz * z_periodic) * np.cos(2 * kx * x_periodic)
            + 0.3
        ) / 6.0
        
    elif tpms_type == 'split-p':
        # Split-P surface - use periodic coordinates
        field = (
            1.1 * (
                np.sin(2 * kx * x_periodic) * np.cos(ky * y_periodic) * np.sin(kz * z_periodic)
                + np.sin(kx * x_periodic) * np.sin(2 * ky * y_periodic) * np.cos(kz * z_periodic)
                + np.cos(kx * x_periodic) * np.sin(ky * y_periodic) * np.sin(2 * kz * z_periodic)
            )
            - 0.2 * (
                np.cos(2 * kx * x_periodic) * np.cos(2 * ky * y_periodic)
                + np.cos(2 * ky * y_periodic) * np.cos(2 * kz * z_periodic)
                + np.cos(2 * kz * z_periodic) * np.cos(2 * kx * x_periodic)
            )
            - 0.4 * (
                np.cos(2 * kx * x_periodic) + np.cos(2 * ky * y_periodic) + np.cos(2 * kz * z_periodic)
            )
        ) / 5.0
        
    else:
        # Default to gyroid if unknown type - use periodic coordinates
        print(f"Warning: Unknown TPMS type '{tpms_type}', defaulting to 'gyroid'")
        field = (
            np.sin(kx * x_periodic) * np.cos(ky * y_periodic)
            + np.sin(ky * y_periodic) * np.cos(kz * z_periodic)
            + np.sin(kz * z_periodic) * np.cos(kx * x_periodic)
        ) / 3.0

    # Apply Gaussian smoothing if requested
    # Use mode='wrap' for periodic boundary conditions to preserve periodicity
    if params.smoothness > 0:
        field = ndimage.gaussian_filter(field, sigma=params.smoothness, mode='wrap')
        max_abs = np.max(np.abs(field))
        if max_abs > 0:
            field = field / max_abs

    return np.clip(field, -1.0, 1.0)


# Alias for backward compatibility
gyroid_field = tpms_field


def compute_porosity_profile(params: GyroidParameters, nlayers: int) -> np.ndarray:
    """Compute target porosity profile along the z-direction."""
    phi_max = params.porosity_max
    phi_min = params.porosity_min if params.grad else params.porosity_max

    if params.grad == 0 or params.func_degree == 0:
        profile = np.full(nlayers, phi_max)
    elif params.func_degree == 1:
        q = np.arange(nlayers)
        denom = max(nlayers - 1, 1)
        profile = -((phi_max - phi_min) / denom) * q + phi_max
    elif params.func_degree == 2:
        if nlayers < 4:
            raise ValueError("Quadratic gradient requires at least 4 layers.")
        q = np.arange(nlayers)
        denom = (nlayers - 3) ** 2
        profile = -((phi_max - phi_min) / denom) * (q - 2) ** 2 + phi_max
    else:
        raise ValueError("Unsupported func_degree.")

    return np.clip(profile, params.porosity_min, params.porosity_max)


def solve_layer_threshold(layer_field: np.ndarray, solid_target: float, delta: float) -> Tuple[float, float]:
    """Binary search the threshold for a single voxel layer."""
    low, high = -1.0, 1.0
    tolerance = max(delta, 1e-6)
    best_mid = 0.0
    best_fraction = 0.0

    for _ in range(60):
        mid = 0.5 * (low + high)
        solid_fraction = float(np.mean(layer_field > mid))
        error = solid_fraction - solid_target
        best_mid = mid
        best_fraction = solid_fraction
        if abs(error) <= tolerance:
            break
        if error > 0:
            low = mid
        else:
            high = mid

    return best_mid, best_fraction


def add_bounding_box(volume: np.ndarray, spacing: Tuple[float, float, float], wall_thickness: float) -> np.ndarray:
    """Add solid bounding box walls to enclose the gyroid structure.
    
    Walls are added only at the outer boundaries, preserving internal periodicity.
    The walls should align with the grid spacing to avoid breaking tiling.
    """
    nx, ny, nz = volume.shape
    dx, dy, dz = spacing
    
    # Calculate wall thickness in voxels - use exact division to ensure alignment
    # Round to nearest integer to avoid misalignment issues
    wall_x = max(1, int(round(wall_thickness / dx)))
    wall_y = max(1, int(round(wall_thickness / dy)))
    wall_z = max(1, int(round(wall_thickness / dz)))
    
    # Create a copy to avoid modifying the original
    enclosed_volume = volume.copy()
    
    # Add walls on all 6 faces - only at the very edges
    # This preserves the internal periodic structure
    enclosed_volume[:, :, :wall_z] = True      # Bottom (z=0)
    enclosed_volume[:, :, -wall_z:] = True     # Top (z=max)
    enclosed_volume[:, :wall_y, :] = True       # Front (y=0)
    enclosed_volume[:, -wall_y:, :] = True      # Back (y=max)
    enclosed_volume[:wall_x, :, :] = True       # Left (x=0)
    enclosed_volume[-wall_x:, :, :] = True      # Right (x=max)
    
    return enclosed_volume


def generate_volume(params: GyroidParameters) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, np.ndarray]]:
    """Generate a binary TPMS volume enforcing the porosity profile."""
    params = validate_params(params)
    grid = generate_coordinate_grid(params)
    field = tpms_field(params, grid)

    nx = params.numx * params.nsteps
    ny = params.numy * params.nsteps
    nz = params.numz * params.nsteps

    porosity_targets = compute_porosity_profile(params, nz)
    solid_targets = np.clip(1.0 - porosity_targets, 0.0, 1.0)

    volume = np.zeros((nx, ny, nz), dtype=bool)
    thresholds = np.zeros(nz, dtype=float)
    achieved_porosity = np.zeros(nz, dtype=float)

    for layer_idx in range(nz):
        threshold, solid_fraction = solve_layer_threshold(
            field[:, :, layer_idx], solid_targets[layer_idx], params.delta
        )
        volume[:, :, layer_idx] = field[:, :, layer_idx] > threshold
        thresholds[layer_idx] = threshold
        achieved_porosity[layer_idx] = 1.0 - solid_fraction

    lx, ly, lz = compute_domain_lengths(params)
    spacing = (lx / nx, ly / ny, lz / nz)
    
    # Add bounding box walls to enclose the structure
    # Only add if wall_thickness > 0 to avoid interfering with periodicity
    if params.wall_thickness > 0:
        volume = add_bounding_box(volume, spacing, params.wall_thickness)

    metadata: Dict[str, np.ndarray] = {
        "porosity_targets": porosity_targets,
        "porosity_achieved": achieved_porosity,
        "thresholds": thresholds,
        "average_porosity": np.mean(achieved_porosity),
        "volume_fraction": np.mean(volume),
        "lengths": np.array([lx, ly, lz]),
        "spacing": np.array(spacing),
    }
    return volume, spacing, metadata


def marching_cubes_mesh(volume: np.ndarray, spacing: Tuple[float, float, float], params: GyroidParameters):
    """Extract a triangle mesh from the voxelised gyroid."""
    verts, faces, normals, values = measure.marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=spacing,
        step_size=params.marching_step,
    )
    return verts, faces, normals, values


def export_stl(verts: np.ndarray, faces: np.ndarray, output_dir: Path, tpms_type: str = 'gyroid') -> Path:
    """Write the mesh to an STL file and return the resulting path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stl_path = output_dir / f"{tpms_type}_{timestamp}.stl"

    gyroid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            gyroid_mesh.vectors[i][j] = verts[face[j], :]
    gyroid_mesh.save(stl_path.as_posix())
    return stl_path


def visualise(
    params: GyroidParameters,
    volume: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    spacing: Tuple[float, float, float],
    metadata: Dict[str, np.ndarray],
):
    lx, ly, lz = metadata["lengths"]

    fig = plt.figure(figsize=(14, 8))

    ax3d = fig.add_subplot(2, 2, (1, 3), projection="3d")
    poly = Poly3DCollection(verts[faces], alpha=0.65, facecolor="#1f77b4", edgecolor="none")
    ax3d.add_collection3d(poly)
    ax3d.set_xlim(0, lx)
    ax3d.set_ylim(0, ly)
    ax3d.set_zlim(0, lz)
    ax3d.set_box_aspect((lx, ly, lz))
    tpms_type = params.tpms_type.lower() if hasattr(params, 'tpms_type') else 'gyroid'
    ax3d.set_title(f"{tpms_type.capitalize()} TPMS STL Preview")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")

    ax_side = fig.add_subplot(2, 2, 2)
    side_projection = np.max(volume, axis=0).astype(float)
    im_side = ax_side.imshow(side_projection.T, origin="lower", cmap="viridis")
    ax_side.set_title("Side View (X Projection)")
    ax_side.set_xlabel("Y voxels")
    ax_side.set_ylabel("Z voxels")
    fig.colorbar(im_side, ax=ax_side, fraction=0.046, pad=0.04)

    ax_cut = fig.add_subplot(2, 2, 4)
    mid_z = volume.shape[2] // 2
    cut = volume[:, :, mid_z].astype(float)
    im_cut = ax_cut.imshow(cut.T, origin="lower", cmap="plasma")
    ax_cut.set_title(f"Cut View (Z={mid_z})")
    ax_cut.set_xlabel("X voxels")
    ax_cut.set_ylabel("Y voxels")
    fig.colorbar(im_cut, ax=ax_cut, fraction=0.046, pad=0.04)

    inputs_caption = (
        f"numx/numy/numz: {params.numx}/{params.numy}/{params.numz}\n"
        f"unit_cell_size: {params.unit_cell_size:.2f} mm\n"
        f"nsteps: {params.nsteps}, delta: {params.delta:.3f}\n"
        f"porosity_min/max: {params.porosity_min:.2f}/{params.porosity_max:.2f}\n"
        f"grad: {params.grad}, func_degree: {params.func_degree}"
    )
    outputs_caption = (
        f"avg_porosity: {metadata['average_porosity']:.3f}\n"
        f"volume_fraction: {metadata['volume_fraction']:.3f}\n"
        f"domain (mm): {lx:.1f} × {ly:.1f} × {lz:.1f}"
    )
    fig.text(0.05, 0.04, f"Inputs:\n{inputs_caption}", ha="left", va="bottom", fontsize=9, family="monospace")
    fig.text(0.55, 0.04, f"Outputs:\n{outputs_caption}", ha="left", va="bottom", fontsize=9, family="monospace")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def create_gyroid(params: GyroidParameters, output_dir: Path, show_plot: bool = True) -> Path:
    """Generate, export, and optionally visualise a TPMS lattice."""
    volume, spacing, metadata = generate_volume(params)
    
    # Validate volume has some solid material
    solid_voxels = np.sum(volume)
    total_voxels = volume.size
    solid_fraction = solid_voxels / total_voxels if total_voxels > 0 else 0.0
    
    if solid_voxels == 0:
        raise ValueError(
            f"Generated volume is empty! Check parameters:\n"
            f"  - porosity_min={params.porosity_min}, porosity_max={params.porosity_max}\n"
            f"  - nsteps={params.nsteps} (try increasing to 15-20)\n"
            f"  - unit_cell_size={params.unit_cell_size}mm (try increasing)\n"
            f"  - wall_thickness={params.wall_thickness}mm (try increasing)"
        )
    
    if solid_fraction < 0.01:  # Less than 1% solid is too sparse
        raise ValueError(
            f"Volume is too sparse (only {solid_fraction*100:.1f}% solid)! "
            f"Try reducing porosity (current: {params.porosity_min}-{params.porosity_max}) "
            f"or increasing nsteps (current: {params.nsteps})"
        )
    
    verts, faces, *_ = marching_cubes_mesh(volume, spacing, params)
    
    # Get TPMS type for filename
    tpms_type = params.tpms_type.lower() if hasattr(params, 'tpms_type') else 'gyroid'
    
    # Validate mesh has faces
    if len(faces) == 0:
        raise ValueError(
            f"Marching cubes produced no faces! Volume might be too sparse.\n"
            f"  - Solid fraction: {solid_fraction*100:.1f}%\n"
            f"  - Try: reducing porosity, increasing nsteps (current: {params.nsteps}), "
            f"or reducing marching_step (current: {params.marching_step})"
        )
    
    if len(faces) < 100:  # Very few faces might indicate a problem
        print(f"Warning: Mesh has only {len(faces)} faces, which is quite low. Consider increasing nsteps or reducing marching_step.")
    
    stl_path = export_stl(verts, faces, output_dir, tpms_type=tpms_type)
    
    # Validate STL file was created and has reasonable size
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file was not created at {stl_path}")
    
    file_size = stl_path.stat().st_size
    if file_size < 5000:  # Less than 5KB is suspicious for a valid mesh
        raise ValueError(
            f"STL file is too small ({file_size} bytes, {file_size/1024:.1f} KB)! "
            f"Mesh generation likely failed.\n"
            f"  - Number of faces: {len(faces)}\n"
            f"  - Solid fraction: {solid_fraction*100:.1f}%\n"
            f"  - Try: increasing nsteps (current: {params.nsteps}), "
            f"reducing marching_step (current: {params.marching_step}), "
            f"or reducing porosity"
        )
    
    if show_plot:
        visualise(params, volume, verts, faces, spacing, metadata)
    return stl_path


def main():
    params = DEFAULT_PARAMS
    output_dir = Path.cwd() / "gyroid_outputs"
    stl_path = create_gyroid(params, output_dir)

    tpms_type = params.tpms_type.lower() if hasattr(params, 'tpms_type') else 'gyroid'
    print(f"Generated {tpms_type} TPMS with the following settings:")
    print(f"  numx/numy/numz: {params.numx}/{params.numy}/{params.numz}")
    print(f"  unit cell size: {params.unit_cell_size:.2f} mm")
    print(f"  porosity range: [{params.porosity_min:.2f}, {params.porosity_max:.2f}]")
    print(f"  grad: {params.grad} (degree {params.func_degree})")
    print(f"  tolerance delta: {params.delta:.3f}")
    print(f"STL exported to: {stl_path}")


if __name__ == "__main__":
    main()