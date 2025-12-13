#!/usr/bin/env python3
"""
Script to create line graphs showing relationships between any input and output parameters.
Automatically detects when other variables are constant and creates separate lines for each group.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load simulation results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['thickness', 'span', 'radial_center_threshold', 'radial_edge_threshold',
                   'mazars_compressive_strength_MPa', 'mazars_tensile_strength_MPa',
                   'mazars_total_energy_absorption_J', 'mazars_max_damage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def find_constant_variables(df: pd.DataFrame, x_param: str, y_param: str) -> list:
    """
    Find which variables are constant (or nearly constant) across the dataset
    when considering x_param and y_param.
    
    Returns list of column names that are constant.
    """
    # Exclude x_param, y_param, and non-parameter columns
    exclude_cols = [x_param, y_param, 'stl_path']
    param_cols = [col for col in df.columns if col not in exclude_cols]
    
    constant_vars = []
    for col in param_cols:
        if col in df.columns:
            # Check if column has low variance (nearly constant)
            if df[col].dtype in ['float64', 'int64']:
                if df[col].nunique() <= 1:
                    constant_vars.append(col)
            else:
                # For categorical, check if all values are the same
                if df[col].nunique() <= 1:
                    constant_vars.append(col)
    
    return constant_vars


def create_line_graph(df: pd.DataFrame, x_param: str, y_param: str, output_dir: Path):
    """
    Create line graphs showing relationship between x_param and y_param.
    Automatically groups by constant variables and creates separate lines.
    """
    # Check if parameters exist
    if x_param not in df.columns:
        print(f"Error: {x_param} not found in CSV columns: {df.columns.tolist()}")
        return
    
    if y_param not in df.columns:
        print(f"Error: {y_param} not found in CSV columns: {df.columns.tolist()}")
        return
    
    # Remove rows with missing values and handle infinities
    data = df[[x_param, y_param]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(data) < 2:
        print(f"Not enough data for {x_param} vs {y_param}")
        return
    
    # Check for valid numeric ranges
    x_min, x_max = data[x_param].min(), data[x_param].max()
    y_min, y_max = data[y_param].min(), data[y_param].max()
    
    if x_max == x_min:
        print(f"Warning: {x_param} has no variation (all values = {x_min})")
        return
    
    if y_max == y_min:
        print(f"Warning: {y_param} has no variation (all values = {y_min})")
        return
    
    print(f"Data range: {x_param} [{x_min:.3f}, {x_max:.3f}], {y_param} [{y_min:.3f}, {y_max:.3f}]")
    
    # Find all parameter columns (excluding x, y, and metadata)
    exclude_cols = [x_param, y_param, 'stl_path']
    param_cols = [col for col in df.columns 
                 if col not in exclude_cols and col in df.columns]
    
    # Group by all other parameters to create separate lines
    # But be selective - only group by parameters that create meaningful groups
    group_cols = []
    for col in param_cols:
        if col in df.columns:
            # Only group by columns that have variation (not constant)
            # And have reasonable number of unique values (not too many)
            n_unique = df[col].nunique()
            if n_unique > 1 and n_unique <= 20:  # Limit to reasonable number of groups
                group_cols.append(col)
    
    # If no grouping columns, just plot all data
    if len(group_cols) == 0:
        # Simple line graph
        data_sorted = data.sort_values(x_param)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data_sorted[x_param], data_sorted[y_param], 
               'o-', linewidth=2, markersize=5, alpha=0.7)
        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)
        ax.set_title(f'{y_param} vs {x_param}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits with padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        plt.tight_layout()
        filename = f"line_{x_param}_vs_{y_param}.png"
        filename = filename.replace('/', '_').replace(' ', '_')
        output_path = output_dir / filename
        
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Line graph saved to {output_path}")
        except Exception as e:
            plt.close()
            print(f"Error saving graph: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Group data by other parameters
    grouped = df.groupby(group_cols)
    
    # Create color palette
    n_groups = len(grouped)
    colors = sns.color_palette("husl", n_groups) if n_groups > 1 else ['blue']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Track overall ranges for axis limits
    all_x_values = []
    all_y_values = []
    lines_plotted = 0
    
    # Plot each group as a separate line
    for idx, (group_key, group_data) in enumerate(grouped):
        group_data_clean = group_data[[x_param, y_param]].copy()
        group_data_clean = group_data_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(group_data_clean) < 2:
            continue
        
        # If multiple points have same x value, aggregate (take mean of y values)
        if group_data_clean[x_param].duplicated().any():
            group_data_clean = group_data_clean.groupby(x_param)[y_param].mean().reset_index()
        
        group_data_clean = group_data_clean.sort_values(x_param)
        
        if len(group_data_clean) < 2:
            continue
        
        # Collect values for axis limits
        all_x_values.extend(group_data_clean[x_param].values)
        all_y_values.extend(group_data_clean[y_param].values)
        
        # Create label from group values
        if isinstance(group_key, tuple):
            label_parts = [f"{col}={val}" for col, val in zip(group_cols, group_key)]
        else:
            label_parts = [f"{group_cols[0]}={group_key}"]
        
        label = ", ".join(label_parts)
        if len(label) > 50:  # Truncate long labels
            label = label[:47] + "..."
        
        color = colors[idx % len(colors)]
        ax.plot(group_data_clean[x_param], group_data_clean[y_param],
               'o-', linewidth=2.5, markersize=5, alpha=0.8, 
               color=color, label=label, zorder=3)
        lines_plotted += 1
    
    # If no lines were plotted, fall back to simple plot without grouping
    if lines_plotted == 0:
        print(f"No valid data groups found. Plotting all data together...")
        plt.close()
        
        # Fall back to simple plot
        data_sorted = data.sort_values(x_param)
        
        # Aggregate duplicate x values
        if data_sorted[x_param].duplicated().any():
            data_sorted = data_sorted.groupby(x_param)[y_param].mean().reset_index()
            data_sorted = data_sorted.sort_values(x_param)
        
        if len(data_sorted) < 2:
            print(f"Not enough data points after aggregation")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data_sorted[x_param], data_sorted[y_param], 
               'o-', linewidth=2.5, markersize=5, alpha=0.8)
        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)
        ax.set_title(f'{y_param} vs {x_param}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        plt.tight_layout()
        filename = f"line_{x_param}_vs_{y_param}.png"
        filename = filename.replace('/', '_').replace(' ', '_')
        output_path = output_dir / filename
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Line graph saved to {output_path}")
        except Exception as e:
            plt.close()
            print(f"✗ Error saving graph: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Set axis limits with padding
    if len(all_x_values) > 0 and len(all_y_values) > 0:
        x_min_plot = min(all_x_values)
        x_max_plot = max(all_x_values)
        y_min_plot = min(all_y_values)
        y_max_plot = max(all_y_values)
        
        x_range_plot = x_max_plot - x_min_plot
        y_range_plot = y_max_plot - y_min_plot
        
        # Add padding (5% on each side)
        if x_range_plot > 0:
            ax.set_xlim(x_min_plot - 0.05 * x_range_plot, x_max_plot + 0.05 * x_range_plot)
        if y_range_plot > 0:
            ax.set_ylim(y_min_plot - 0.05 * y_range_plot, y_max_plot + 0.05 * y_range_plot)
    
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    ax.set_title(f'{y_param} vs {x_param}', fontsize=14, fontweight='bold')
    
    if lines_plotted > 1:  # Only show legend if multiple lines
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"line_{x_param}_vs_{y_param}.png"
    filename = filename.replace('/', '_').replace(' ', '_')
    output_path = output_dir / filename
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Attempting to save to: {output_path.absolute()}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Line graph saved to {output_path}")
        print(f"  Grouped by: {', '.join(group_cols) if group_cols else 'none'}")
        print(f"  Number of groups: {n_groups}, Lines plotted: {lines_plotted}")
        plt.close()
    except Exception as e:
        print(f"✗ Error saving graph: {e}")
        import traceback
        traceback.print_exc()
        plt.close()
        raise


def create_all_line_graphs(df: pd.DataFrame, output_dir: Path):
    """Create line graphs for common parameter-output combinations."""
    # Input parameters
    input_params = ['thickness', 'span', 'radial_center_threshold', 'radial_edge_threshold']
    
    # Output parameters
    output_params = [
        'mazars_compressive_strength_MPa',
        'mazars_tensile_strength_MPa',
        'mazars_total_energy_absorption_J',
        'mazars_max_damage',
        'earthquake_max_displacement_mm',
        'earthquake_max_damage',
    ]
    
    # Get available parameters
    available_inputs = [p for p in input_params if p in df.columns]
    available_outputs = [p for p in output_params if p in df.columns]
    
    print(f"Creating line graphs for {len(available_inputs)} inputs × {len(available_outputs)} outputs...")
    
    for input_param in available_inputs:
        for output_param in available_outputs:
            create_line_graph(df, input_param, output_param, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Create line graphs showing relationships between input and output parameters'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to simulation_results.csv file'
    )
    parser.add_argument(
        '--x-param',
        type=str,
        default=None,
        help='X-axis parameter (input, e.g., thickness)'
    )
    parser.add_argument(
        '--y-param',
        type=str,
        default=None,
        help='Y-axis parameter (output, e.g., mazars_compressive_strength_MPa)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Create line graphs for all common parameter combinations'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_path}...")
    df = load_data(args.csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.all:
        # Create all line graphs
        print("\nCreating line graphs for all parameter combinations...")
        create_all_line_graphs(df, output_dir)
    elif args.x_param and args.y_param:
        # Create specific line graph
        print(f"\nCreating line graph for {args.x_param} vs {args.y_param}...")
        create_line_graph(df, args.x_param, args.y_param, output_dir)
    else:
        print("Error: Either specify --x-param and --y-param, or use --all")
        print("\nAvailable columns:")
        print(df.columns.tolist())
        return
    
    print(f"\nLine graphs saved to {output_dir}")


if __name__ == '__main__':
    main()

