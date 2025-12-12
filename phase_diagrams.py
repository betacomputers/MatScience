#!/usr/bin/env python3
"""
Script to create phase diagrams showing which formula dominates in different regions
of parameter space (e.g., thickness vs threshold).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load simulation results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['thickness', 'threshold', 'span',
                   'mazars_cross_sectional_area_m2',
                   'mazars_compressive_strength_MPa',
                   'mazars_tensile_strength_MPa']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_mass(df: pd.DataFrame) -> pd.Series:
    """Calculate mass from available data."""
    MATERIAL_DENSITY = 2400  # kg/m³
    
    mass = pd.Series(index=df.index, dtype=float)
    
    if 'mazars_cross_sectional_area_m2' in df.columns:
        if 'size' in df.columns:
            height = df['size'].fillna(0.03)
        elif 'span' in df.columns:
            height = df['span'].fillna(0.02)
        else:
            height = 0.03
        
        volume = df['mazars_cross_sectional_area_m2'] * height
        mass = volume * MATERIAL_DENSITY
    elif 'size' in df.columns:
        porosity_factor = 0.3
        volume = (df['size'] ** 3) * porosity_factor
        mass = volume * MATERIAL_DENSITY
    
    return mass


def find_dominant_formula(df: pd.DataFrame, x_param: str, y_param: str, 
                         metric: str, aggregation: str = 'mean') -> pd.DataFrame:
    """
    Find which formula dominates at each (x_param, y_param) combination.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    x_param, y_param : str
        Parameter names for x and y axes
    metric : str
        Metric to use for determining dominance (e.g., 'mazars_compressive_strength_MPa')
    aggregation : str
        How to aggregate multiple values at same (x, y) point ('mean', 'max', 'min')
    
    Returns:
    --------
    pd.DataFrame with columns [x_param, y_param, 'formula_name', metric]
    """
    # Group by formula, x_param, y_param and aggregate
    grouped = df.groupby(['formula_name', x_param, y_param])[metric].agg(aggregation).reset_index()
    
    # For each (x, y) combination, find formula with best metric
    # Group by x, y and find max
    best_formula = grouped.loc[grouped.groupby([x_param, y_param])[metric].idxmax()]
    
    return best_formula[[x_param, y_param, 'formula_name', metric]]


def create_phase_diagram(df: pd.DataFrame, x_param: str, y_param: str, 
                        metric: str, output_dir: Path, smooth: bool = True):
    """
    Create a phase diagram showing which formula dominates in parameter space.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    x_param, y_param : str
        Parameter names for x and y axes
    metric : str
        Metric to use for determining dominance
    output_dir : Path
        Output directory
    smooth : bool
        Whether to smooth the phase boundaries
    """
    # Check required columns
    required_cols = ['formula_name', x_param, y_param, metric]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns: {required_cols}")
        return
    
    # Remove missing data
    data = df[required_cols].dropna()
    
    if len(data) == 0:
        print("No valid data for phase diagram")
        return
    
    # Find dominant formula at each point
    dominant = find_dominant_formula(data, x_param, y_param, metric)
    
    # Get unique formulas and create color map
    formulas = sorted(data['formula_name'].unique())
    n_formulas = len(formulas)
    
    if n_formulas == 0:
        print("No formulas found")
        return
    
    # Create color palette
    colors = sns.color_palette("husl", n_formulas)
    formula_colors = dict(zip(formulas, colors))
    
    # Create grid for interpolation
    x_min, x_max = data[x_param].min(), data[x_param].max()
    y_min, y_max = data[y_param].min(), data[y_param].max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    
    # Create fine grid
    grid_resolution = 200
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Create phase map: assign each formula a number
    formula_to_num = {formula: i for i, formula in enumerate(formulas)}
    
    # Interpolate dominant formula at each grid point
    # Use nearest neighbor interpolation for categorical data
    phase_map = np.zeros_like(xi_grid)
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            x_val = xi[i]
            y_val = yi[j]
            
            # Find closest point in dominant data
            distances = np.sqrt((dominant[x_param] - x_val)**2 + 
                              (dominant[y_param] - y_val)**2)
            closest_idx = distances.idxmin()
            closest_formula = dominant.loc[closest_idx, 'formula_name']
            phase_map[j, i] = formula_to_num[closest_formula]
    
    # Smooth phase boundaries if requested
    if smooth:
        # Apply Gaussian filter to smooth boundaries
        phase_map = gaussian_filter(phase_map, sigma=2.0)
    
    # Create figure with extra width for colorbar
    fig, ax = plt.subplots(figsize=(13, 10))
    
    # Create custom colormap for formulas
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([formula_colors[f] for f in formulas])
    
    # Plot phase diagram (formula regions) as main background - full opacity
    im = ax.contourf(xi_grid, yi_grid, phase_map, levels=n_formulas-1, 
                cmap=cmap, alpha=1.0, extend='neither')
    
    # Plot phase boundaries with darker lines
    ax.contour(xi_grid, yi_grid, phase_map, levels=n_formulas-1, 
             colors='black', linewidths=2, alpha=0.8)
    
    # Plot actual data points - remove duplicates and add jitter to avoid overlap
    for formula in formulas:
        formula_data = data[data['formula_name'] == formula]
        # Remove duplicates at same (x, y) location
        unique_data = formula_data.drop_duplicates(subset=[x_param, y_param])
        
        # Add small random jitter to avoid exact overlap
        np.random.seed(42)  # For reproducibility
        x_jitter = unique_data[x_param] + np.random.normal(0, 0.01 * (x_max - x_min), len(unique_data))
        y_jitter = unique_data[y_param] + np.random.normal(0, 0.01 * (y_max - y_min), len(unique_data))
        
        ax.scatter(x_jitter, y_jitter, 
                  c=[formula_colors[formula]], s=30, alpha=0.8, 
                  edgecolors='white', linewidths=1.5, label=formula, zorder=5)
    
    # Add colorbar showing formula colors (discrete)
    boundaries = np.arange(n_formulas + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    cbar = plt.colorbar(im, ax=ax, boundaries=boundaries, ticks=np.arange(n_formulas))
    cbar.set_ticklabels(formulas)
    cbar.set_label('Dominant Formula', fontsize=11)
    
    # Annotate maximum values for each formula region
    for formula in formulas:
        # Get all points where this formula dominates
        formula_dominant = dominant[dominant['formula_name'] == formula]
        
        if len(formula_dominant) == 0:
            continue
        
        # Find the maximum metric value for this formula
        max_value = formula_dominant[metric].max()
        max_row = formula_dominant.loc[formula_dominant[metric].idxmax()]
        
        # Find a good position for annotation (centroid of the region or max value location)
        # Use the location where max value occurs, or centroid if multiple points
        if len(formula_dominant) > 1:
            # Use centroid of the region
            annot_x = formula_dominant[x_param].mean()
            annot_y = formula_dominant[y_param].mean()
        else:
            # Use the max value location
            annot_x = max_row[x_param]
            annot_y = max_row[y_param]
        
        # Format the value based on metric type
        if 'ratio' in metric.lower():
            value_str = f'{max_value:.2f}'
        else:
            value_str = f'{max_value:.1f}'
        
        # Add annotation with white background for visibility
        ax.annotate(f'Max: {value_str}', 
                   xy=(annot_x, annot_y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='black', alpha=0.8, linewidth=1),
                   ha='left', va='bottom',
                   color='black', zorder=10)
    
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    ax.set_title(f'Phase Diagram: Dominant Formula by {metric}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Position legend inside plot area to avoid being cut off
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
    
    # Adjust layout to ensure everything is visible
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on right for colorbar
    
    # Save figure
    filename = f"phase_diagram_{x_param}_vs_{y_param}_{metric}.png"
    filename = filename.replace('/', '_').replace(' ', '_')
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Phase diagram saved to {output_path}")


def create_multi_metric_phase_diagrams(df: pd.DataFrame, x_param: str, y_param: str, 
                                      output_dir: Path):
    """Create phase diagrams for multiple metrics."""
    metrics = [
        'mazars_compressive_strength_MPa',
        'mazars_tensile_strength_MPa',
    ]
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        print("No metrics available for phase diagrams")
        return
    
    # Calculate compressive/tensile strength ratio
    if 'mazars_compressive_strength_MPa' in df.columns and 'mazars_tensile_strength_MPa' in df.columns:
        # Avoid division by zero
        df['compressive_tensile_ratio'] = df['mazars_compressive_strength_MPa'] / df['mazars_tensile_strength_MPa'].replace(0, np.nan)
        print("Calculated compressive/tensile strength ratio")
    
    # Calculate strength-to-mass ratio if possible
    if 'mazars_cross_sectional_area_m2' in df.columns:
        df['mass_kg'] = calculate_mass(df)
        if 'mazars_compressive_strength_MPa' in df.columns:
            df['compressive_strength_to_mass'] = df['mazars_compressive_strength_MPa'] / df['mass_kg']
        if 'mazars_tensile_strength_MPa' in df.columns:
            df['tensile_strength_to_mass'] = df['mazars_tensile_strength_MPa'] / df['mass_kg']
    
    # Create phase diagrams for each metric
    for metric in available_metrics:
        print(f"\nCreating phase diagram for {metric}...")
        create_phase_diagram(df, x_param, y_param, metric, output_dir)
    
    # Create compressive/tensile ratio phase diagram if available
    if 'compressive_tensile_ratio' in df.columns:
        print(f"\nCreating phase diagram for compressive_tensile_ratio...")
        create_phase_diagram(df, x_param, y_param, 'compressive_tensile_ratio', output_dir)
    
    # Create strength-to-mass phase diagrams if available
    if 'compressive_strength_to_mass' in df.columns:
        print(f"\nCreating phase diagram for compressive_strength_to_mass...")
        create_phase_diagram(df, x_param, y_param, 'compressive_strength_to_mass', output_dir)
    
    if 'tensile_strength_to_mass' in df.columns:
        print(f"\nCreating phase diagram for tensile_strength_to_mass...")
        create_phase_diagram(df, x_param, y_param, 'tensile_strength_to_mass', output_dir)


def create_combined_phase_diagram(df: pd.DataFrame, x_param: str, y_param: str, 
                                 output_dir: Path):
    """Create a combined phase diagram showing multiple metrics side by side."""
    metrics = [
        ('mazars_compressive_strength_MPa', 'Compressive Strength'),
        ('mazars_tensile_strength_MPa', 'Tensile Strength'),
    ]
    
    # Check which metrics are available
    available_metrics = [(m, label) for m, label in metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        print("No metrics available for combined phase diagram")
        return
    
    # Calculate compressive/tensile strength ratio
    if 'mazars_compressive_strength_MPa' in df.columns and 'mazars_tensile_strength_MPa' in df.columns:
        df['compressive_tensile_ratio'] = df['mazars_compressive_strength_MPa'] / df['mazars_tensile_strength_MPa'].replace(0, np.nan)
        available_metrics.append(('compressive_tensile_ratio', 'Compressive/Tensile Ratio'))
        print("Calculated compressive/tensile strength ratio")
    
    # Calculate strength-to-mass if possible
    if 'mazars_cross_sectional_area_m2' in df.columns:
        df['mass_kg'] = calculate_mass(df)
        if 'mazars_compressive_strength_MPa' in df.columns:
            df['compressive_strength_to_mass'] = df['mazars_compressive_strength_MPa'] / df['mass_kg']
            available_metrics.append(('compressive_strength_to_mass', 'Compressive Strength/Mass'))
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 8))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric, label) in zip(axes, available_metrics):
        # Find dominant formula
        required_cols = ['formula_name', x_param, y_param, metric]
        data = df[required_cols].dropna()
        
        if len(data) == 0:
            ax.axis('off')
            continue
        
        dominant = find_dominant_formula(data, x_param, y_param, metric)
        
        # Get unique formulas
        formulas = sorted(data['formula_name'].unique())
        n_formulas = len(formulas)
        colors = sns.color_palette("husl", n_formulas)
        formula_colors = dict(zip(formulas, colors))
        
        # Create grid
        x_min, x_max = data[x_param].min(), data[x_param].max()
        y_min, y_max = data[y_param].min(), data[y_param].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.05 * x_range
        x_max += 0.05 * x_range
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range
        
        grid_resolution = 200
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        formula_to_num = {formula: i for i, formula in enumerate(formulas)}
        phase_map = np.zeros_like(xi_grid)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                x_val = xi[i]
                y_val = yi[j]
                distances = np.sqrt((dominant[x_param] - x_val)**2 + 
                                  (dominant[y_param] - y_val)**2)
                closest_idx = distances.idxmin()
                closest_formula = dominant.loc[closest_idx, 'formula_name']
                phase_map[j, i] = formula_to_num[closest_formula]
        
        phase_map = gaussian_filter(phase_map, sigma=2.0)
        
        from matplotlib.colors import ListedColormap, BoundaryNorm
        
        # Create custom colormap for formulas
        cmap = ListedColormap([formula_colors[f] for f in formulas])
        
        # Plot phase diagram (formula regions) as main background - full opacity
        im = ax.contourf(xi_grid, yi_grid, phase_map, levels=n_formulas-1, 
                   cmap=cmap, alpha=1.0, extend='neither')
        
        # Plot phase boundaries with darker lines
        ax.contour(xi_grid, yi_grid, phase_map, levels=n_formulas-1, 
                 colors='black', linewidths=1.5, alpha=0.8)
        
        # Plot data points with jitter to avoid overlap
        for formula in formulas:
            formula_data = data[data['formula_name'] == formula]
            # Remove duplicates at same (x, y) location
            unique_data = formula_data.drop_duplicates(subset=[x_param, y_param])
            
            # Add small random jitter
            np.random.seed(42)
            x_jitter = unique_data[x_param] + np.random.normal(0, 0.01 * (x_max - x_min), len(unique_data))
            y_jitter = unique_data[y_param] + np.random.normal(0, 0.01 * (y_max - y_min), len(unique_data))
            
            ax.scatter(x_jitter, y_jitter, 
                      c=[formula_colors[formula]], s=25, alpha=0.8, 
                      edgecolors='white', linewidths=1, label=formula, zorder=5)
        
        # Add colorbar showing formula colors
        boundaries = np.arange(n_formulas + 1) - 0.5
        norm = BoundaryNorm(boundaries, cmap.N)
        cbar = plt.colorbar(im, ax=ax, boundaries=boundaries, ticks=np.arange(n_formulas))
        cbar.set_ticklabels(formulas)
        cbar.set_label('Dominant Formula', fontsize=9)
        
        # Annotate maximum values for each formula region
        for formula in formulas:
            # Get all points where this formula dominates
            formula_dominant = dominant[dominant['formula_name'] == formula]
            
            if len(formula_dominant) == 0:
                continue
            
            # Find the maximum metric value for this formula
            max_value = formula_dominant[metric].max()
            
            # Find a good position for annotation (centroid of the region)
            if len(formula_dominant) > 1:
                annot_x = formula_dominant[x_param].mean()
                annot_y = formula_dominant[y_param].mean()
            else:
                max_row = formula_dominant.loc[formula_dominant[metric].idxmax()]
                annot_x = max_row[x_param]
                annot_y = max_row[y_param]
            
            # Format the value based on metric type
            if 'ratio' in metric.lower():
                value_str = f'{max_value:.2f}'
            else:
                value_str = f'{max_value:.1f}'
            
            # Add annotation with white background for visibility
            ax.annotate(f'Max: {value_str}', 
                       xy=(annot_x, annot_y),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='black', alpha=0.8, linewidth=0.8),
                       ha='left', va='bottom',
                       color='black', zorder=10)
        
        ax.set_xlabel(x_param, fontsize=11)
        ax.set_ylabel(y_param, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if ax == axes[0]:  # Only show legend on first subplot
            # Position legend inside plot to avoid being cut off
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9, fancybox=True, shadow=True)
    
    plt.suptitle(f'Phase Diagrams: Dominant Formula by Metric\n{x_param} vs {y_param}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle and colorbars
    
    filename = f"phase_diagram_combined_{x_param}_vs_{y_param}.png"
    filename = filename.replace('/', '_').replace(' ', '_')
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Combined phase diagram saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create phase diagrams showing dominant formulas in parameter space'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to simulation_results.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    parser.add_argument(
        '--x-param',
        type=str,
        default='thickness',
        help='X-axis parameter (default: thickness)'
    )
    parser.add_argument(
        '--y-param',
        type=str,
        default='threshold',
        help='Y-axis parameter (default: threshold)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=None,
        help='Specific metric to use (default: create for all available metrics)'
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable smoothing of phase boundaries'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Create combined phase diagram with multiple metrics'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_path}...")
    df = load_data(args.csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create phase diagrams
    if args.combined:
        print(f"\nCreating combined phase diagram for {args.x_param} vs {args.y_param}...")
        create_combined_phase_diagram(df, args.x_param, args.y_param, output_dir)
    elif args.metric:
        print(f"\nCreating phase diagram for {args.metric}...")
        create_phase_diagram(df, args.x_param, args.y_param, args.metric, 
                           output_dir, smooth=not args.no_smooth)
    else:
        print(f"\nCreating phase diagrams for all metrics...")
        create_multi_metric_phase_diagrams(df, args.x_param, args.y_param, output_dir)
    
    print(f"\nPhase diagram analysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()

