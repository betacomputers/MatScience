from numpy import pi
import csv
from pathlib import Path
import sys
import os

from gyroids.lib import run, name_of_file, STL_OUTPUT_DIR
from gyroids.lib.types import *

from gyroids.lib.surfaces import (
    gyroid,
    diamond,
    holes,
    schwarz_p,
    schwarz_g,
    l_surface,
    lamella,
)

# Import simulation functions
# Note: earthquake_simulator imports from mazars_model_sfepy, but the file is mazars_model.py
# Create an alias so earthquake_simulator can import it
import mazars_model as mazars_model_sfepy
sys.modules['mazars_model_sfepy'] = mazars_model_sfepy

from mazars_model import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    run_compression_test,
    run_tensile_test,
)

from earthquake_simulator import (
    GroundMotion,
    EarthquakeSimulationParameters,
    EarthquakeSimulator,
    run_earthquake_test,
)

thicknesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
thresholds = [-1.0, -0.5, 0.0, 0.5, 1.0]  

# Testing: limit number of combinations to process (set to None to process all)
max_combinations = 5  # Set to a number (e.g., 5) for testing, or None for all

# CSV output file
csv_output = Path("simulation_results.csv")

# Prepare CSV with all columns
csv_columns = [
    # Parameters
    "formula_name",
    "thickness",
    "threshold",
    "stl_path",
    "subdivisions",
    "span",
    "size",
    "granularity",
    # Mazars compression results
    "mazars_compressive_strength_MPa",
    "mazars_tensile_strength_MPa",
    "mazars_max_force_N",
    "mazars_cross_sectional_area_m2",
    "mazars_total_energy_absorption_J",
    "mazars_max_damage",
    # Earthquake results
    "earthquake_max_displacement_mm",
    "earthquake_residual_displacement_mm",
    "earthquake_max_damage",
    "earthquake_residual_damage",
    "earthquake_max_stress_MPa",
    "earthquake_residual_stress_MPa",
    "earthquake_peak_acceleration_g",
    "earthquake_response_amplification",
    "earthquake_failure_occurred",
    "earthquake_failure_time_s",
    "earthquake_max_kinetic_energy_J",
    "earthquake_max_strain_energy_J",
]

# Open CSV file for writing
with open(csv_output, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    
    total_combinations = len([gyroid, diamond, holes, schwarz_p, schwarz_g, l_surface, lamella]) * len(thicknesses) * len(thresholds)
    if max_combinations is not None:
        total_combinations = min(total_combinations, max_combinations)
        print(f"Testing mode: Processing {max_combinations} combinations (out of {len([gyroid, diamond, holes, schwarz_p, schwarz_g, l_surface, lamella]) * len(thicknesses) * len(thresholds)} total)")
    current = 0
    
    should_break = False
    for f in (
        gyroid,
        diamond,
        holes,
        schwarz_p,
        schwarz_g,
        l_surface,
        lamella,
    ):
        if should_break:
            break
        for thickness in thicknesses:
            if should_break:
                break
            for threshold in thresholds:
                # Check if we've reached the max combinations limit before processing
                if max_combinations is not None and current >= max_combinations:
                    should_break = True
                    break
                
                current += 1
                
                print(f"\n{'='*60}")
                print(f"Processing {current}/{total_combinations}: {f.__name__}, thickness={thickness}, threshold={threshold}")
                print(f"{'='*60}")
                
                # Initialize row data
                row_data = {
                    "formula_name": f.__name__,
                    "thickness": thickness,
                    "threshold": threshold,
                    "subdivisions": 150,
                    "span": pi * 2.0,
                    "size": 30,
                    "granularity": 0.2,
                }
                
                try:
                    # Create parameters and generate STL
                    params = PlotParams(
                        name=f"{f.__name__}_{thickness}_{threshold}",
                        subdivisions=150,
                        span=pi * 2.0,
                        formula=f,
                        size=30,
                        thickness=thickness,
                        granularity=0.2,
                        threshold=threshold,
                    )
                    
                    # Generate STL file (this will create the file if --generate-stl flag is set)
                    # We need to ensure the STL is generated
                    print(f"Generating STL file...")
                    # Mock sys.argv to provide --generate-stl flag
                    # parse_args() expects only the script name and flags, not the params.name
                    original_argv = sys.argv.copy()
                    sys.argv = ['multi.py', '--generate-stl']
                    try:
                        run(params)
                    finally:
                        sys.argv = original_argv
                    
                    # Construct STL path
                    stl_path = STL_OUTPUT_DIR / f"{params.name}.stl"
                    row_data["stl_path"] = str(stl_path)

                    
                    print(f"STL file: {stl_path}")
                    
                    # Run Mazars model simulation
                    print(f"\nRunning Mazars model simulation...")
                    try:
                        material = MaterialProperties()
                        sim_params = SimulationParameters(
                            element_size=0.05,
                            max_force=3500.0,
                            target_stress_mpa=35.0,
                            num_steps=10,
                        )
                        
                        domain = load_stl_and_create_mesh(stl_path, sim_params.element_size)
                        compression_results = run_compression_test(domain, material, sim_params)
                        tension_results = run_tensile_test(domain, material, sim_params)
                        
                        # Extract Mazars results
                        row_data["mazars_compressive_strength_MPa"] = compression_results.get('compressive_strength', 0.0) / 1e6
                        row_data["mazars_tensile_strength_MPa"] = tension_results.get('tensile_strength', 0.0) / 1e6
                        row_data["mazars_max_force_N"] = compression_results.get('max_force_N', 0.0)
                        row_data["mazars_cross_sectional_area_m2"] = compression_results.get('cross_sectional_area_m2', 0.0)
                        row_data["mazars_total_energy_absorption_J"] = compression_results.get('total_energy_absorption', 0.0)
                        row_data["mazars_max_damage"] = max(compression_results.get('damage_history', [0.0]) + tension_results.get('damage_history', [0.0]), default=0.0)
                        
                        print(f"  ✓ Mazars simulation completed")
                        print(f"    Compressive strength: {row_data['mazars_compressive_strength_MPa']:.2f} MPa")
                        print(f"    Tensile strength: {row_data['mazars_tensile_strength_MPa']:.2f} MPa")
                    except Exception as e:
                        print(f"  ✗ Error in Mazars simulation: {e}")
                        import traceback
                        traceback.print_exc()
                        # Set default values for failed simulation
                        row_data["mazars_compressive_strength_MPa"] = None
                        row_data["mazars_tensile_strength_MPa"] = None
                        row_data["mazars_max_force_N"] = None
                        row_data["mazars_cross_sectional_area_m2"] = None
                        row_data["mazars_total_energy_absorption_J"] = None
                        row_data["mazars_max_damage"] = None
                    
                    # Run Earthquake simulation
                    print(f"\nRunning Earthquake simulation...")
                    try:
                        # Create synthetic ground motion (0.5g PGA)
                        ground_motion = GroundMotion.synthetic(
                            duration=20.0,
                            dt=0.01,
                            pga=0.5 * 9.81,  # 0.5g in m/s²
                            name=f"earthquake_{params.name}"
                        )
                        
                        earthquake_results = run_earthquake_test(
                            stl_path,
                            ground_motion,
                            material=material,
                            element_size=0.05,
                            damping_ratio=0.05
                        )
                        
                        # Extract Earthquake results
                        row_data["earthquake_max_displacement_mm"] = earthquake_results.get('max_displacement_mm', 0.0)
                        row_data["earthquake_residual_displacement_mm"] = earthquake_results.get('residual_displacement_mm', 0.0)
                        row_data["earthquake_max_damage"] = earthquake_results.get('max_damage', 0.0)
                        row_data["earthquake_residual_damage"] = earthquake_results.get('residual_damage', 0.0)
                        row_data["earthquake_max_stress_MPa"] = earthquake_results.get('max_stress_MPa', 0.0)
                        row_data["earthquake_residual_stress_MPa"] = earthquake_results.get('residual_stress_MPa', 0.0)
                        row_data["earthquake_peak_acceleration_g"] = earthquake_results.get('peak_acceleration_g', 0.0)
                        row_data["earthquake_response_amplification"] = earthquake_results.get('response_amplification', 0.0)
                        row_data["earthquake_failure_occurred"] = earthquake_results.get('failure_occurred', False)
                        row_data["earthquake_failure_time_s"] = earthquake_results.get('failure_time_s', None)
                        row_data["earthquake_max_kinetic_energy_J"] = earthquake_results.get('max_kinetic_energy_J', 0.0)
                        row_data["earthquake_max_strain_energy_J"] = earthquake_results.get('max_strain_energy_J', 0.0)
                        
                        print(f"  ✓ Earthquake simulation completed")
                        print(f"    Max displacement: {row_data['earthquake_max_displacement_mm']:.2f} mm")
                        print(f"    Max damage: {row_data['earthquake_max_damage']:.3f}")
                    except Exception as e:
                        print(f"  ✗ Error in Earthquake simulation: {e}")
                        import traceback
                        traceback.print_exc()
                        # Set default values for failed simulation
                        row_data["earthquake_max_displacement_mm"] = None
                        row_data["earthquake_residual_displacement_mm"] = None
                        row_data["earthquake_max_damage"] = None
                        row_data["earthquake_residual_damage"] = None
                        row_data["earthquake_max_stress_MPa"] = None
                        row_data["earthquake_residual_stress_MPa"] = None
                        row_data["earthquake_peak_acceleration_g"] = None
                        row_data["earthquake_response_amplification"] = None
                        row_data["earthquake_failure_occurred"] = None
                        row_data["earthquake_failure_time_s"] = None
                        row_data["earthquake_max_kinetic_energy_J"] = None
                        row_data["earthquake_max_strain_energy_J"] = None
                    
                    # Write row to CSV
                    writer.writerow(row_data)
                    csvfile.flush()  # Ensure data is written immediately
                    print(f"  ✓ Results saved to CSV")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {f.__name__}, thickness={thickness}, threshold={threshold}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Write row with error info
                    row_data["stl_path"] = "ERROR"
                    writer.writerow(row_data)
                    csvfile.flush()
                
                # Check if we should break after processing this combination
                if should_break:
                    break

print(f"Results saved to: {csv_output}")