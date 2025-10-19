import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from skimage import measure
from stl import mesh

class SimpleGyroidAnalysis:
    """Simple mechanical analysis of gyroid structures"""
    
    def __init__(self, size=20.0, resolution=60, unit_cell_size=4.0, wall_thickness=0.3):
        """
        Initialize gyroid analysis
        
        Parameters:
        - size: Physical size of the gyroid in mm
        - resolution: Grid resolution (points per dimension)
        - unit_cell_size: Size of one gyroid unit cell in mm
        - wall_thickness: Thickness of gyroid walls (0-1, where 1 is solid)
        """
        self.size = size
        self.resolution = resolution
        self.unit_cell_size = unit_cell_size
        self.wall_thickness = wall_thickness
        
        # Create 3D grid
        self.x = np.linspace(-size/2, size/2, resolution)
        self.y = np.linspace(-size/2, size/2, resolution)
        self.z = np.linspace(-size/2, size/2, resolution)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Gyroid volume
        self.gyroid_volume = None
        
    def generate_gyroid(self):
        """Generate gyroid structure using mathematical formula"""
        print("Generating gyroid structure...")
        
        # Scale coordinates to unit cell size
        scale = 2 * np.pi / self.unit_cell_size
        x_scaled = self.X * scale
        y_scaled = self.Y * scale
        z_scaled = self.Z * scale
        
        # Gyroid level set function
        gyroid_function = (np.sin(x_scaled) * np.cos(y_scaled) + 
                          np.sin(y_scaled) * np.cos(z_scaled) + 
                          np.sin(z_scaled) * np.cos(x_scaled))
        
        # Create volume by thresholding
        threshold = 2 * self.wall_thickness - 1
        self.gyroid_volume = gyroid_function > threshold
        
        print(f"Gyroid generated with {np.sum(self.gyroid_volume)} solid voxels")
        print(f"Volume fraction: {np.sum(self.gyroid_volume) / self.gyroid_volume.size:.3f}")
        return self.gyroid_volume
    
    def create_mesh(self):
        """Create mesh from gyroid volume using marching cubes"""
        if self.gyroid_volume is None:
            raise ValueError("Must generate gyroid first")
            
        print("Creating mesh with marching cubes...")
        
        # Marching cubes to create mesh
        verts, faces, normals, values = measure.marching_cubes(
            self.gyroid_volume.astype(float), 
            0.5, 
            spacing=(self.size/self.resolution, self.size/self.resolution, self.size/self.resolution)
        )
        
        print(f"Mesh created with {len(verts)} vertices and {len(faces)} faces")
        return verts, faces, normals, values
    
    def save_stl(self, verts, faces, filename=None):
        """Save gyroid structure to STL file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gyroid_{timestamp}.stl"
        
        # Create STL mesh
        gyroid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                gyroid_mesh.vectors[i][j] = verts[f[j], :]
        
        # Save STL file
        filepath = Path(filename)
        gyroid_mesh.save(str(filepath))
        print(f"Saved STL file: {filepath}")
        return filepath
    
    def calculate_mechanical_properties(self):
        """Calculate basic mechanical properties using analytical methods"""
        if self.gyroid_volume is None:
            raise ValueError("Must generate gyroid first")
        
        print("Calculating mechanical properties...")
        
        # Material properties (PLA plastic)
        E_solid = 3.5e9  # Pa - Young's modulus of solid material
        nu_solid = 0.35  # Poisson's ratio
        density_solid = 1250  # kg/m³
        
        # Calculate volume fraction
        volume_fraction = np.sum(self.gyroid_volume) / self.gyroid_volume.size
        
        # Calculate volume in m³
        voxel_volume = (self.size / self.resolution / 1000) ** 3  # Convert mm to m
        total_volume = np.sum(self.gyroid_volume) * voxel_volume
        
        # Calculate mass
        mass = total_volume * density_solid
        
        # Estimate effective properties using Gibson-Ashby scaling laws
        # For gyroid structures, the scaling is approximately:
        # E_eff = E_solid * (rho/rho_solid)^2.5
        # where rho/rho_solid is the relative density (volume fraction)
        
        relative_density = volume_fraction
        E_effective = E_solid * (relative_density ** 2.5)
        
        # Compressive strength scales similarly
        sigma_yield_solid = 50e6  # Pa - typical yield strength of PLA
        sigma_compressive = sigma_yield_solid * (relative_density ** 1.5)
        
        # Calculate specific properties (per unit mass)
        specific_stiffness = E_effective / density_solid
        specific_strength = sigma_compressive / density_solid
        
        return {
            'volume_fraction': volume_fraction,
            'total_volume': total_volume,
            'mass': mass,
            'E_effective': E_effective,
            'sigma_compressive': sigma_compressive,
            'specific_stiffness': specific_stiffness,
            'specific_strength': specific_strength,
            'relative_density': relative_density
        }
    
    def simulate_compressive_loading(self, applied_stress=1e6):
        """Simulate compressive loading using simplified analysis"""
        print(f"Simulating compressive loading at {applied_stress/1e6:.1f} MPa...")
        
        properties = self.calculate_mechanical_properties()
        
        # Calculate strain under applied stress
        strain = applied_stress / properties['E_effective']
        
        # Calculate displacement
        displacement = strain * self.size / 1000  # Convert mm to m
        
        # Check if stress exceeds compressive strength
        failure = applied_stress > properties['sigma_compressive']
        
        # Calculate safety factor
        safety_factor = properties['sigma_compressive'] / applied_stress if applied_stress > 0 else float('inf')
        
        return {
            'applied_stress': applied_stress,
            'strain': strain,
            'displacement': displacement,
            'failure': failure,
            'safety_factor': safety_factor,
            'properties': properties
        }
    
    def analyze_stress_distribution(self):
        """Analyze stress distribution using simplified approach"""
        print("Analyzing stress distribution...")
        
        if self.gyroid_volume is None:
            raise ValueError("Must generate gyroid first")
        
        # Create a simplified stress field based on geometry
        # Higher stress concentrations at thin sections and intersections
        
        # Calculate local thickness (simplified)
        stress_field = np.zeros_like(self.gyroid_volume, dtype=float)
        
        # For each solid voxel, calculate local stress concentration
        solid_indices = np.where(self.gyroid_volume)
        
        for i, j, k in zip(solid_indices[0], solid_indices[1], solid_indices[2]):
            # Count solid neighbors (6-connectivity)
            neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if abs(di) + abs(dj) + abs(dk) == 1:  # 6-connectivity
                            ni, nj, nk = i + di, j + dj, k + dk
                            if (0 <= ni < self.resolution and 
                                0 <= nj < self.resolution and 
                                0 <= nk < self.resolution and
                                self.gyroid_volume[ni, nj, nk]):
                                neighbors += 1
            
            # Stress concentration factor based on connectivity
            # Fewer neighbors = higher stress concentration
            if neighbors > 0:
                stress_concentration = 6.0 / neighbors  # Max stress concentration
            else:
                stress_concentration = 10.0  # Very high for isolated voxels
            
            stress_field[i, j, k] = stress_concentration
        
        return stress_field
    
    def visualize_results(self, stress_field, save_plot=True):
        """Visualize analysis results"""
        print("Creating visualization...")
        
        fig = plt.figure(figsize=(15, 12))
        
        # Original gyroid structure
        ax1 = fig.add_subplot(231)
        mid_z = self.resolution // 2
        im1 = ax1.imshow(self.gyroid_volume[:, :, mid_z], cmap='viridis', origin='lower')
        ax1.set_title('Gyroid Structure (XY slice)')
        ax1.set_xlabel('X (voxels)')
        ax1.set_ylabel('Y (voxels)')
        plt.colorbar(im1, ax=ax1)
        
        # Stress distribution
        ax2 = fig.add_subplot(232)
        im2 = ax2.imshow(stress_field[:, :, mid_z], cmap='plasma', origin='lower')
        ax2.set_title('Stress Distribution (XY slice)')
        ax2.set_xlabel('X (voxels)')
        ax2.set_ylabel('Y (voxels)')
        plt.colorbar(im2, ax=ax2)
        
        # 3D stress visualization
        ax3 = fig.add_subplot(233, projection='3d')
        stress_points = np.where(stress_field > 0)
        sample_indices = slice(0, len(stress_points[0]), 20)  # Sample for visualization
        px = self.x[stress_points[0][sample_indices]]
        py = self.y[stress_points[1][sample_indices]]
        pz = self.z[stress_points[2][sample_indices]]
        stress_values = stress_field[stress_points[0][sample_indices], 
                                   stress_points[1][sample_indices], 
                                   stress_points[2][sample_indices]]
        scatter = ax3.scatter(px, py, pz, c=stress_values, cmap='plasma', s=1, alpha=0.6)
        ax3.set_title('3D Stress Distribution')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        plt.colorbar(scatter, ax=ax3, shrink=0.5)
        
        # Cross-sections
        ax4 = fig.add_subplot(234)
        mid_y = self.resolution // 2
        im4 = ax4.imshow(stress_field[:, mid_y, :], cmap='plasma', origin='lower')
        ax4.set_title('Stress Distribution (XZ slice)')
        ax4.set_xlabel('X (voxels)')
        ax4.set_ylabel('Z (voxels)')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(235)
        mid_x = self.resolution // 2
        im5 = ax5.imshow(stress_field[mid_x, :, :], cmap='plasma', origin='lower')
        ax5.set_title('Stress Distribution (YZ slice)')
        ax5.set_xlabel('Y (voxels)')
        ax5.set_ylabel('Z (voxels)')
        plt.colorbar(im5, ax=ax5)
        
        # Stress histogram
        ax6 = fig.add_subplot(236)
        stress_values = stress_field[stress_field > 0]
        ax6.hist(stress_values, bins=50, alpha=0.7, color='orange')
        ax6.set_title('Stress Distribution Histogram')
        ax6.set_xlabel('Stress Concentration Factor')
        ax6.set_ylabel('Frequency')
        ax6.axvline(np.mean(stress_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(stress_values):.2f}')
        ax6.legend()
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"gyroid_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Saved analysis visualization: {plot_filename}")
        
        plt.show()
        return fig
    
    def report_analysis(self, loading_results):
        """Report analysis results"""
        print("\n" + "="*60)
        print("GYROID MECHANICAL ANALYSIS RESULTS")
        print("="*60)
        
        props = loading_results['properties']
        
        print(f"\nGEOMETRY:")
        print(f"  Size: {self.size} mm")
        print(f"  Resolution: {self.resolution}³ voxels")
        print(f"  Unit cell size: {self.unit_cell_size} mm")
        print(f"  Wall thickness: {self.wall_thickness}")
        
        print(f"\nVOLUME PROPERTIES:")
        print(f"  Volume fraction: {props['volume_fraction']:.3f} ({props['volume_fraction']*100:.1f}%)")
        print(f"  Total volume: {props['total_volume']*1e9:.1f} mm³")
        print(f"  Mass: {props['mass']*1000:.1f} g")
        
        print(f"\nMECHANICAL PROPERTIES:")
        print(f"  Effective Young's modulus: {props['E_effective']/1e6:.1f} MPa")
        print(f"  Compressive strength: {props['sigma_compressive']/1e6:.1f} MPa")
        print(f"  Specific stiffness: {props['specific_stiffness']:.1f} m²/s²")
        print(f"  Specific strength: {props['specific_strength']:.1f} m²/s²")
        
        print(f"\nLOADING SIMULATION:")
        print(f"  Applied stress: {loading_results['applied_stress']/1e6:.1f} MPa")
        print(f"  Strain: {loading_results['strain']*1000:.3f} mstrain")
        print(f"  Displacement: {loading_results['displacement']*1000:.3f} mm")
        print(f"  Safety factor: {loading_results['safety_factor']:.2f}")
        print(f"  Failure: {'YES' if loading_results['failure'] else 'NO'}")

def main():
    """Main function to run the complete analysis"""
    print("=== Simple Gyroid Mechanical Analysis ===")
    
    # Parameters
    size = 20.0  # mm
    resolution = 60  # Lower resolution for faster computation
    unit_cell_size = 4.0  # mm
    wall_thickness = 0.4  # 0-1, where 1 is solid
    
    print(f"Parameters:")
    print(f"  Size: {size} mm")
    print(f"  Resolution: {resolution}x{resolution}x{resolution}")
    print(f"  Unit cell size: {unit_cell_size} mm")
    print(f"  Wall thickness: {wall_thickness} ({wall_thickness*100:.0f}% volume fraction)")
    
    # Initialize analysis
    analysis = SimpleGyroidAnalysis(size=size, resolution=resolution, 
                                   unit_cell_size=unit_cell_size, wall_thickness=wall_thickness)
    
    # Generate gyroid
    gyroid_volume = analysis.generate_gyroid()
    
    # Create mesh and save STL file
    verts, faces, normals, values = analysis.create_mesh()
    stl_path = analysis.save_stl(verts, faces)
    
    # Simulate compressive loading
    loading_results = analysis.simulate_compressive_loading(applied_stress=2e6)  # 2 MPa
    
    # Analyze stress distribution
    stress_field = analysis.analyze_stress_distribution()
    
    # Visualize results
    analysis.visualize_results(stress_field)
    
    # Report results
    analysis.report_analysis(loading_results)
    
    print(f"\nAnalysis completed successfully!")
    print(f"STL file saved: {stl_path}")
    print("The gyroid structure is ready for 3D printing!")
    print("This approach avoids the mesh connectivity issues of NGSolve")
    print("while still providing useful mechanical property estimates.")

if __name__ == "__main__":
    main()
