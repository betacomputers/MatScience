import os
import sys
import csv
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import trimesh

# Import gyroid generation functions
from gyroid_gen import (
    fn_plot_tpms_eq,
    fn_check_face_normals,
    fn_generate_mesh,
    fn_export_stl_file
)

# Import simulation functions
from linear_elastic_sim import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    run_compression_test
)


# Configuration - modify these to change the parameter sweep
TPMS_SHELL_TYPES = [
    'Shell-TPMS Gyroid',
    'Shell-TPMS Diamond',
    'Shell-TPMS Lidinoid',
    'Shell-TPMS Split-P',  # Uncomment to include more types
    'Shell-TPMS Schwarz'
]

THICKNESS_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Thickness values to test

# Cell sizes to test (mm) - same value for all dimensions [x, y, z]
CELL_SIZES_VALUES = [3.0, 4.0, 5.0, 6.0]  # mm - cell sizes to iterate over

# Fixed parameters for all structures
TPMS_TYPE = 'Shell'
SIZES = [20.0, 20.0, 20.0]  # mm [x, y, z]
ORIGIN = [0.0, 0.0, 0.0]  # mm
UNIT_CELL_MESH_RESOLUTION = 50  # Points per unit cell
C = 0.0  # Threshold parameter (controls volume fraction)
FLIP_FACE_NORMALS = False

# Simulation parameters
SIM_ELEMENT_SIZE = 0.05  # m
SIM_MAX_FORCE = 20000000.0  # N (20 MN)
SIM_NUM_STEPS = 5

# Output settings
OUTPUT_CSV = 'dataset_full.csv'
TEMP_DIR = Path('temp_sweep_files')
TEMP_DIR.mkdir(exist_ok=True)


def generate_tpms_structure(tpms_design: str, thickness: float, cell_size: float, output_stl_path: Path) -> Tuple[bool, Path]:
    try:
        print(f"\n{'='*60}")
        print(f"Generating: {tpms_design}, thickness={thickness}, cell_size={cell_size}mm")
        print(f"{'='*60}")
        
        # Create cell_sizes array from single cell_size value
        cell_sizes = [cell_size, cell_size, cell_size]
        
        # Initial empty mesh
        mesh = trimesh.Trimesh()
        
        # Plot the TPMS equation (without visualization)
        mesh, vertices = fn_plot_tpms_eq(
            TPMS_TYPE, tpms_design, SIZES, cell_sizes, ORIGIN,
            UNIT_CELL_MESH_RESOLUTION, C, thickness, mesh, show_plot=False
        )
        
        # Check face normals (silent, no plot)
        mesh = fn_check_face_normals(mesh, silent=True, show_plot=False)
        
        # Generate the mesh
        final_mesh = fn_generate_mesh(
            TPMS_TYPE, tpms_design, C, thickness, SIZES, cell_sizes,
            ORIGIN, UNIT_CELL_MESH_RESOLUTION, mesh, FLIP_FACE_NORMALS, silent=True
        )
        
        # Export to STL file
        # fn_export_stl_file adds .stl extension automatically, so use stem
        file_name = output_stl_path.stem
        directory_path = output_stl_path.parent
        fn_export_stl_file(final_mesh, file_name, str(directory_path), silent=True)
        
        # Verify file was created (fn_export_stl_file adds .stl extension)
        actual_stl_path = output_stl_path.parent / f"{file_name}.stl"
        if actual_stl_path.exists():
            print(f"STL file created: {actual_stl_path}")
            return True, actual_stl_path
        else:
            print(f"Failed to create STL file: {actual_stl_path}")
            return False, output_stl_path
            
    except Exception as e:
        print(f"Error generating structure: {e}")
        import traceback
        traceback.print_exc()
        return False, output_stl_path


def run_simulation(stl_path: Path) -> Dict:
    try:
        print(f"Running simulation on: {stl_path.name}")
        
        # Material properties
        material = MaterialProperties()
        
        # Simulation parameters
        sim_params = SimulationParameters(
            element_size=SIM_ELEMENT_SIZE,
            max_force=SIM_MAX_FORCE,
            num_steps=SIM_NUM_STEPS,
        )
        
        # Load and mesh STL
        fenics_mesh = load_stl_and_create_mesh(stl_path, sim_params.element_size)
        
        # Run compression test
        results = run_compression_test(fenics_mesh, material, sim_params)
        
        print(f"✓ Simulation completed")
        return results
        
    except Exception as e:
        print(f"✗ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_results_summary(results: Dict) -> Dict:
    return {
        'compressive_strength_MPa': results['compressive_strength'] / 1e6,
        'max_force_N': results['max_force_N'],
        'cross_sectional_area_m2': results['cross_sectional_area_m2'],
        'energy_absorption_J': results['total_energy_absorption'],
        'max_displacement_mm': max([abs(d) for d in results['displacements']]) * 1000 if results['displacements'] else 0.0,
        'max_strain': max([abs(s) for s in results['strains']]) if results['strains'] else 0.0,
    }


def main():
    print("\n" + "="*60)
    print("TPMS PARAMETER SWEEP")
    print("="*60)
    print(f"Shell types: {len(TPMS_SHELL_TYPES)}")
    print(f"Thickness values: {THICKNESS_VALUES}")
    print(f"Cell sizes: {CELL_SIZES_VALUES} mm")
    print(f"Total combinations: {len(TPMS_SHELL_TYPES) * len(THICKNESS_VALUES) * len(CELL_SIZES_VALUES)}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("="*60 + "\n")
    
    # Prepare CSV output
    csv_rows: List[Dict] = []
    
    # Iterate through all parameter combinations
    total_combinations = len(TPMS_SHELL_TYPES) * len(THICKNESS_VALUES) * len(CELL_SIZES_VALUES)
    current_combination = 0
    
    for tpms_design in TPMS_SHELL_TYPES:
        for thickness in THICKNESS_VALUES:
            for cell_size in CELL_SIZES_VALUES:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Processing combination...")
                
                # Create unique filename for this combination
                design_short = tpms_design.replace('Shell-TPMS ', '').replace(' ', '_').lower()
                stl_filename_base = f"{design_short}_th{thickness:.2f}_cs{cell_size:.1f}"
                stl_path = TEMP_DIR / f"{stl_filename_base}.stl"
                
                # Generate TPMS structure
                success, actual_stl_path = generate_tpms_structure(tpms_design, thickness, cell_size, stl_path)
                if not success:
                    print(f"Skipping simulation due to generation failure")
                    # Still record in CSV with error flag
                    csv_rows.append({
                        'tpms_design': tpms_design,
                        'thickness': thickness,
                        'cell_size_mm': cell_size,
                        'compressive_strength_MPa': None,
                        'max_force_N': None,
                        'cross_sectional_area_m2': None,
                        'energy_absorption_J': None,
                        'max_displacement_mm': None,
                        'max_strain': None,
                        'status': 'generation_failed',
                    })
                    continue
            
                # Use the actual path that was created
                stl_path = actual_stl_path
                
                # Run simulation
                results = run_simulation(stl_path)
                
                if results is None:
                    print(f"⚠ Skipping CSV entry due to simulation failure")
                    # Still record in CSV with error flag
                    csv_rows.append({
                        'tpms_design': tpms_design,
                        'thickness': thickness,
                        'cell_size_mm': cell_size,
                        'compressive_strength_MPa': None,
                        'max_force_N': None,
                        'cross_sectional_area_m2': None,
                        'energy_absorption_J': None,
                        'max_displacement_mm': None,
                        'max_strain': None,
                        'status': 'simulation_failed',
                    })
                    continue
                
                # Extract summary results
                summary = extract_results_summary(results)
                
                # Add input parameters
                row = {
                    'tpms_design': tpms_design,
                    'thickness': thickness,
                    'cell_size_mm': cell_size,
                    **summary,
                    'status': 'success',
                }
                
                csv_rows.append(row)
                
                # Print summary
                print(f"Results: Compressive strength = {summary['compressive_strength_MPa']:.2f} MPa")
                
                # Clean up STL file to save space (optional - comment out if you want to keep them)
                # stl_path.unlink()
    
    # Write CSV file
    print(f"\n{'='*60}")
    print(f"Writing results to CSV: {OUTPUT_CSV}")
    print(f"{'='*60}")
    
    if csv_rows:
        fieldnames = [
            'tpms_design',
            'thickness',
            'cell_size_mm',
            'compressive_strength_MPa',
            'max_force_N',
            'cross_sectional_area_m2',
            'energy_absorption_J',
            'max_displacement_mm',
            'max_strain',
            'status',
        ]
        
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"✓ CSV file written: {OUTPUT_CSV}")
        print(f"✓ Total rows: {len(csv_rows)}")
        
        # Print summary statistics
        successful_rows = [r for r in csv_rows if r['status'] == 'success']
        if successful_rows:
            print(f"\nSummary Statistics:")
            print(f"  Successful simulations: {len(successful_rows)}/{len(csv_rows)}")
            strengths = [r['compressive_strength_MPa'] for r in successful_rows if r['compressive_strength_MPa'] is not None]
            if strengths:
                print(f"  Compressive strength range: {min(strengths):.2f} - {max(strengths):.2f} MPa")
                print(f"  Average compressive strength: {np.mean(strengths):.2f} MPa")
    else:
        print("No results to write!")
    
    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

