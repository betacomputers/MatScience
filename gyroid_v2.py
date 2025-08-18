import numpy as np
from skimage import measure
from stl import mesh
import matplotlib.pyplot as plt

class GyroidBrickGenerator:
    def __init__(self):
        # Brick parameters
        self.brick_length = 30.0  # mm
        self.brick_width = 20.0   # mm
        self.brick_height = 15.0  # mm
        self.casing_thickness = 2.0  # mm - thickness of outer casing
        
        # Gyroid parameters
        self.cell_size_mm = 4.0  # TPMS unit cell length (mm)
        self.num_cells_x = 6     # Number of cells along X axis
        self.num_cells_y = 4     # Number of cells along Y axis
        self.num_cells_z = 3     # Number of cells along Z axis
        self.wall_thickness = 1.0  # mm - thickness of gyroid walls
        self.isovalue = 0.0      # Gyroid threshold surface
        
        # Quality parameters
        self.points_per_cell = 40  # Resolution: higher = smoother mesh but heavier file
        
        # Output parameters
        self.output_filename = 'gyroid_brick.stl'
        
    def generate_gyroid_field(self, X, Y, Z):
        """Generate the gyroid implicit function field"""
        # Convert mm to radians for TPMS math
        Xr = 2 * np.pi * X / self.cell_size_mm
        Yr = 2 * np.pi * Y / self.cell_size_mm
        Zr = 2 * np.pi * Z / self.cell_size_mm
        
        # Gyroid implicit function
        F = np.sin(Xr) * np.cos(Yr) + np.sin(Yr) * np.cos(Zr) + np.sin(Zr) * np.cos(Xr)
        return F
    
    def create_brick_mask(self, X, Y, Z):
        """Create a mask for the brick shape"""
        # Create brick boundary
        brick_mask = (
            (X >= 0) & (X <= self.brick_length) &
            (Y >= 0) & (Y <= self.brick_width) &
            (Z >= 0) & (Z <= self.brick_height)
        )
        return brick_mask
    
    def create_casing_mask(self, X, Y, Z):
        """Create a mask for the casing (outer shell)"""
        # Outer boundary
        outer_mask = (
            (X >= 0) & (X <= self.brick_length) &
            (Y >= 0) & (Y <= self.brick_width) &
            (Z >= 0) & (Z <= self.brick_height)
        )
        
        # Inner boundary (casing thickness)
        inner_mask = (
            (X >= self.casing_thickness) & (X <= self.brick_length - self.casing_thickness) &
            (Y >= self.casing_thickness) & (Y <= self.brick_width - self.casing_thickness) &
            (Z >= self.casing_thickness) & (Z <= self.brick_height - self.casing_thickness)
        )
        
        # Casing is outer minus inner
        casing_mask = outer_mask & ~inner_mask
        return casing_mask
    
    def create_gyroid_mask(self, X, Y, Z, F):
        """Create a mask for the gyroid structure"""
        # Make shell by thickening surface
        outer_surface = F - self.isovalue
        inner_surface = F - self.isovalue - (self.wall_thickness / self.cell_size_mm) * (2 * np.pi)
        
        # Boolean mask of shell region
        gyroid_mask = (outer_surface >= 0) & (inner_surface <= 0)
        return gyroid_mask
    
    def generate_mesh(self):
        """Generate the complete brick with gyroid structure"""
        print("Generating gyroid brick...")
        
        # Calculate grid dimensions
        L_x = self.brick_length
        L_y = self.brick_width
        L_z = self.brick_height
        
        # Calculate resolution based on cell size and points per cell
        res_x = int(self.num_cells_x * self.points_per_cell)
        res_y = int(self.num_cells_y * self.points_per_cell)
        res_z = int(self.num_cells_z * self.points_per_cell)
        
        # Create grid
        x = np.linspace(0, L_x, res_x)
        y = np.linspace(0, L_y, res_y)
        z = np.linspace(0, L_z, res_z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        print(f"Grid resolution: {res_x} x {res_y} x {res_z}")
        
        # Generate gyroid field
        F = self.generate_gyroid_field(X, Y, Z)
        
        # Create masks
        brick_mask = self.create_brick_mask(X, Y, Z)
        casing_mask = self.create_casing_mask(X, Y, Z)
        gyroid_mask = self.create_gyroid_mask(X, Y, Z, F)
        
        # Combine masks: casing OR (brick interior AND gyroid)
        interior_mask = brick_mask & ~casing_mask
        final_mask = casing_mask | (interior_mask & gyroid_mask)
        
        # Marching cubes mesh
        print("Generating mesh with marching cubes...")
        verts, faces, normals, values = measure.marching_cubes(
            final_mask.astype(float), 
            0, 
            spacing=(L_x/res_x, L_y/res_y, L_z/res_z)
        )
        
        # Create STL mesh
        print(f"Creating STL mesh with {faces.shape[0]} faces...")
        gyroid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                gyroid_mesh.vectors[i][j] = verts[f[j], :]
        
        return gyroid_mesh
    
    def save_mesh(self, mesh_obj):
        """Save the mesh to STL file"""
        mesh_obj.save(self.output_filename)
        print(f"Saved {self.output_filename}")
    
    def preview_parameters(self):
        """Print current parameters for review"""
        print("=== Gyroid Brick Parameters ===")
        print(f"Brick dimensions: {self.brick_length} x {self.brick_width} x {self.brick_height} mm")
        print(f"Casing thickness: {self.casing_thickness} mm")
        print(f"Gyroid cell size: {self.cell_size_mm} mm")
        print(f"Gyroid cells: {self.num_cells_x} x {self.num_cells_y} x {self.num_cells_z}")
        print(f"Gyroid wall thickness: {self.wall_thickness} mm")
        print(f"Resolution: {self.points_per_cell} points per cell")
        print(f"Output file: {self.output_filename}")
        print("=" * 30)

def main():
    # Create generator instance
    generator = GyroidBrickGenerator()
    
    # Preview parameters
    generator.preview_parameters()
    
    # Generate and save mesh
    mesh_obj = generator.generate_mesh()
    generator.save_mesh(mesh_obj)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()
