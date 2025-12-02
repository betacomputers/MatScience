import argparse
import os
import sys
from pathlib import Path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric conversions
    df = df.copy()
    numeric_cols = [
        'unit_cell_size_mm', 'wall_thickness_mm', 'porosity_min', 'porosity_max',
        'compressive_strength_MPa', 'max_force_N', 'cross_sectional_area_m2',
        'energy_absorption_J', 'max_displacement_mm', 'max_strain'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Convert compressive strength from MPa to Pa
    if 'compressive_strength_MPa' in df.columns:
        df['compressive_strength_Pa'] = df['compressive_strength_MPa'] * 1e6

    # Compute max stress (Pa) from force and cross-sectional area
    if 'max_force_N' in df.columns and 'cross_sectional_area_m2' in df.columns:
        df['max_stress_Pa'] = df['max_force_N'] / df['cross_sectional_area_m2']

    # Create a simple volume proxy (m^3) = area * length where length ~ unit cell size (mm -> m)
    if 'cross_sectional_area_m2' in df.columns and 'unit_cell_size_mm' in df.columns:
        df['unit_cell_size_m'] = df['unit_cell_size_mm'] / 1000.0
        df['volume_proxy_m3'] = df['cross_sectional_area_m2'] * df['unit_cell_size_m']
        # avoid division by zero
        df.loc[df['volume_proxy_m3'] <= 0, 'volume_proxy_m3'] = np.nan

        # Strength per volume proxy (Pa / m^3) — a proxy for strength-to-weight when density is constant
        if 'compressive_strength_Pa' in df.columns:
            df['strength_per_volume'] = df['compressive_strength_Pa'] / df['volume_proxy_m3']
        if 'max_force_N' in df.columns:
            df['force_per_volume'] = df['max_force_N'] / df['volume_proxy_m3']

    return df


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    df = compute_metrics(df)

    # Define inputs and outputs
    inputs = ['unit_cell_size_mm', 'wall_thickness_mm', 'porosity_min', 'porosity_max']
    outputs = ['compressive_strength_MPa', 'max_force_N', 'cross_sectional_area_m2',
               'energy_absorption_J', 'max_displacement_mm', 'max_strain']
    
    # Subset dataframe to include only inputs and outputs
    all_cols = inputs + outputs
    df_subset = df[[c for c in all_cols if c in df.columns]].dropna()
    
    # Create heatmap: input-output correlations
    if len(df_subset) > 0:
        corr_full = df_subset.corr()
        
        # Extract input-output submatrix
        inputs_in_df = [c for c in inputs if c in df_subset.columns]
        outputs_in_df = [c for c in outputs if c in df_subset.columns]
        corr_io = corr_full.loc[inputs_in_df, outputs_in_df]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_io, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, 
                    cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        ax.set_title('Input-Output Correlations', fontsize=14, fontweight='bold')
        ax.set_xlabel('Outputs', fontsize=12)
        ax.set_ylabel('Inputs', fontsize=12)
        plt.tight_layout()
        plt.show()
        save_plot(fig, Path('correlation_heatmap.png'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='dataset_full.csv')
    args = p.parse_args()
    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(2)
    main(args.csv)
