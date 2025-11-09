"""
XGBoost-based optimization for gyroid structures to maximize 
compressive strength to mass ratio.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])
    import xgboost as xgb
except Exception as e:
    error_msg = str(e)
    if 'libomp' in error_msg or 'OpenMP' in error_msg:
        print("\n" + "="*60)
        print("ERROR: XGBoost requires OpenMP runtime library")
        print("="*60)
        print("On macOS, install it with:")
        print("  brew install libomp")
        print("\nOn Linux, install it with:")
        print("  sudo apt-get install libgomp1  # Debian/Ubuntu")
        print("  sudo yum install libgomp      # CentOS/RHEL")
        print("\nOn Windows, install Visual C++ Redistributable")
        print("="*60)
    raise

from gyroid_simple_simulation import SimpleGyroidAnalysis
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt


class GyroidXGBoostOptimizer:
    """Optimize gyroid structures using XGBoost to predict strength-to-mass ratio"""
    
    def __init__(self, 
                 size_range: Tuple[float, float] = (15.0, 25.0),
                 resolution: int = 60,
                 unit_cell_size_range: Tuple[float, float] = (2.0, 6.0),
                 wall_thickness_range: Tuple[float, float] = (0.1, 0.8),
                 smoothness_range: Tuple[float, float] = (0.3, 1.0)):
        """
        Initialize optimizer with parameter ranges
        
        Parameters:
        - size_range: Range for physical size (mm)
        - resolution: Grid resolution (fixed for training)
        - unit_cell_size_range: Range for unit cell size (mm)
        - wall_thickness_range: Range for wall thickness (0-1)
        - smoothness_range: Range for smoothness parameter
        """
        self.size_range = size_range
        self.resolution = resolution
        self.unit_cell_size_range = unit_cell_size_range
        self.wall_thickness_range = wall_thickness_range
        self.smoothness_range = smoothness_range
        
        self.model = None
        self.training_data = None
        self.feature_names = ['size', 'unit_cell_size', 'wall_thickness', 'smoothness']
        
    def generate_training_data(self, n_samples: int = 200, verbose: bool = True) -> pd.DataFrame:
        """
        Generate training data by sampling parameter combinations and evaluating them
        
        Parameters:
        - n_samples: Number of samples to generate
        - verbose: Whether to print progress
        """
        if verbose:
            print(f"Generating {n_samples} training samples...")
        
        data = []
        
        # Generate random parameter combinations
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_samples):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{n_samples}")
            
            # Sample parameters
            size = np.random.uniform(*self.size_range)
            unit_cell_size = np.random.uniform(*self.unit_cell_size_range)
            wall_thickness = np.random.uniform(*self.wall_thickness_range)
            smoothness = np.random.uniform(*self.smoothness_range)
            
            # Evaluate this combination
            try:
                result = self._evaluate_parameters(
                    size, unit_cell_size, wall_thickness, smoothness
                )
                
                if result is not None:
                    data.append({
                        'size': size,
                        'unit_cell_size': unit_cell_size,
                        'wall_thickness': wall_thickness,
                        'smoothness': smoothness,
                        **result
                    })
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to evaluate sample {i+1}: {e}")
                continue
        
        df = pd.DataFrame(data)
        self.training_data = df
        
        if verbose:
            print(f"\nGenerated {len(df)} successful samples")
            print(f"Strength-to-mass ratio range: {df['strength_to_mass_ratio'].min():.2e} - {df['strength_to_mass_ratio'].max():.2e}")
        
        return df
    
    def _evaluate_parameters(self, size: float, unit_cell_size: float, 
                            wall_thickness: float, smoothness: float) -> Optional[Dict]:
        """
        Evaluate a parameter combination and return properties
        
        Returns:
        Dictionary with mechanical properties or None if evaluation fails
        """
        try:
            # Create gyroid analysis instance
            analysis = SimpleGyroidAnalysis(
                size=size,
                resolution=self.resolution,
                unit_cell_size=unit_cell_size,
                wall_thickness=wall_thickness,
                smoothness=smoothness
            )
            
            # Generate gyroid
            analysis.generate_gyroid()
            
            # Calculate properties
            properties = analysis.calculate_mechanical_properties()
            
            # Calculate strength-to-mass ratio (compressive strength per unit mass)
            # Units: Pa / kg = m²/s²
            strength_to_mass_ratio = properties['sigma_compressive'] / properties['mass']
            
            return {
                'volume_fraction': properties['volume_fraction'],
                'mass': properties['mass'],
                'sigma_compressive': properties['sigma_compressive'],
                'E_effective': properties['E_effective'],
                'strength_to_mass_ratio': strength_to_mass_ratio
            }
        except Exception as e:
            return None
    
    def train_model(self, n_estimators: int = 100, max_depth: int = 6, 
                   learning_rate: float = 0.1, verbose: bool = True) -> xgb.XGBRegressor:
        """
        Train XGBoost model on training data
        
        Parameters:
        - n_estimators: Number of boosting rounds
        - max_depth: Maximum tree depth
        - learning_rate: Learning rate
        - verbose: Whether to print training progress
        """
        if self.training_data is None or len(self.training_data) == 0:
            raise ValueError("No training data available. Call generate_training_data() first.")
        
        if verbose:
            print(f"\nTraining XGBoost model on {len(self.training_data)} samples...")
        
        # Prepare features and target
        X = self.training_data[self.feature_names]
        y = self.training_data['strength_to_mass_ratio']
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y, verbose=False)
        
        # Evaluate model
        train_score = self.model.score(X, y)
        
        if verbose:
            print(f"Training R² score: {train_score:.4f}")
            
            # Feature importance
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            print("\nFeature importances:")
            for feature, importance in sorted(feature_importance.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")
        
        return self.model
    
    def predict_strength_to_mass(self, size: float, unit_cell_size: float,
                                wall_thickness: float, smoothness: float) -> float:
        """
        Predict strength-to-mass ratio for given parameters using trained model
        
        Returns:
        Predicted strength-to-mass ratio
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X = pd.DataFrame({
            'size': [size],
            'unit_cell_size': [unit_cell_size],
            'wall_thickness': [wall_thickness],
            'smoothness': [smoothness]
        })
        
        prediction = self.model.predict(X)[0]
        return prediction
    
    def optimize_parameters(self, method: str = 'differential_evolution',
                          n_iterations: int = 50) -> Dict:
        """
        Find optimal parameters using the trained model
        
        Parameters:
        - method: 'differential_evolution' or 'random_search'
        - n_iterations: Number of optimization iterations
        
        Returns:
        Dictionary with optimal parameters and predicted value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print(f"\nOptimizing parameters using {method}...")
        
        if method == 'differential_evolution':
            bounds = [
                self.size_range,
                self.unit_cell_size_range,
                self.wall_thickness_range,
                self.smoothness_range
            ]
            
            def objective(params):
                size, unit_cell_size, wall_thickness, smoothness = params
                # Minimize negative of strength-to-mass ratio
                return -self.predict_strength_to_mass(size, unit_cell_size, 
                                                     wall_thickness, smoothness)
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=n_iterations,
                seed=42,
                polish=True
            )
            
            optimal_params = {
                'size': result.x[0],
                'unit_cell_size': result.x[1],
                'wall_thickness': result.x[2],
                'smoothness': result.x[3],
                'predicted_strength_to_mass_ratio': -result.fun,
                'optimization_success': result.success
            }
            
        elif method == 'random_search':
            # Random search
            best_params = None
            best_value = -np.inf
            
            np.random.seed(42)
            for i in range(n_iterations):
                size = np.random.uniform(*self.size_range)
                unit_cell_size = np.random.uniform(*self.unit_cell_size_range)
                wall_thickness = np.random.uniform(*self.wall_thickness_range)
                smoothness = np.random.uniform(*self.smoothness_range)
                
                value = self.predict_strength_to_mass(size, unit_cell_size, 
                                                     wall_thickness, smoothness)
                
                if value > best_value:
                    best_value = value
                    best_params = {
                        'size': size,
                        'unit_cell_size': unit_cell_size,
                        'wall_thickness': wall_thickness,
                        'smoothness': smoothness
                    }
            
            optimal_params = {
                **best_params,
                'predicted_strength_to_mass_ratio': best_value,
                'optimization_success': True
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"\nOptimal parameters found:")
        for key, value in optimal_params.items():
            if key != 'optimization_success':
                print(f"  {key}: {value:.4f}")
        
        return optimal_params
    
    def find_best_structures(self, n_best: int = 10, 
                            save_stl: bool = True,
                            output_dir: str = "optimized_gyroids") -> List[Dict]:
        """
        Find and save the best structures from training data
        
        Parameters:
        - n_best: Number of best structures to save
        - save_stl: Whether to save STL files
        - output_dir: Directory to save STL files
        
        Returns:
        List of dictionaries with best structures' information
        """
        if self.training_data is None or len(self.training_data) == 0:
            raise ValueError("No training data available.")
        
        print(f"\nFinding {n_best} best structures...")
        
        # Sort by strength-to-mass ratio
        sorted_data = self.training_data.nlargest(n_best, 'strength_to_mass_ratio')
        
        best_structures = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for idx, (_, row) in enumerate(sorted_data.iterrows(), 1):
            print(f"\nProcessing best structure #{idx}...")
            print(f"  Strength-to-mass ratio: {row['strength_to_mass_ratio']:.2e} m²/s²")
            print(f"  Parameters: size={row['size']:.2f}, unit_cell={row['unit_cell_size']:.2f}, "
                  f"wall_thickness={row['wall_thickness']:.2f}, smoothness={row['smoothness']:.2f}")
            
            structure_info = {
                'rank': idx,
                'size': row['size'],
                'unit_cell_size': row['unit_cell_size'],
                'wall_thickness': row['wall_thickness'],
                'smoothness': row['smoothness'],
                'strength_to_mass_ratio': row['strength_to_mass_ratio'],
                'mass': row['mass'],
                'sigma_compressive': row['sigma_compressive'],
                'volume_fraction': row['volume_fraction']
            }
            
            if save_stl:
                # Recreate analysis object to generate mesh and save STL
                try:
                    analysis = SimpleGyroidAnalysis(
                        size=row['size'],
                        resolution=self.resolution,
                        unit_cell_size=row['unit_cell_size'],
                        wall_thickness=row['wall_thickness'],
                        smoothness=row['smoothness']
                    )
                    analysis.generate_gyroid()
                    
                    verts, faces, normals, values = analysis.create_mesh()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = output_path / f"gyroid_best_{idx:02d}_{timestamp}.stl"
                    stl_path = analysis.save_stl(verts, faces, filename=str(filename))
                    structure_info['stl_path'] = str(stl_path)
                    print(f"  Saved STL: {stl_path}")
                except Exception as e:
                    print(f"  Warning: Failed to save STL: {e}")
                    structure_info['stl_path'] = None
            
            best_structures.append(structure_info)
        
        # Save summary JSON
        summary_path = output_path / f"best_structures_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = best_structures.copy()  # All fields are JSON-serializable
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nSaved summary to: {summary_path}")
        print(f"Found and saved {len(best_structures)} best structures")
        
        return best_structures
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from: {filepath}")
    
    def plot_training_results(self, save_path: Optional[str] = None):
        """Plot training data and model predictions"""
        if self.training_data is None or len(self.training_data) == 0:
            raise ValueError("No training data available.")
        
        if self.model is None:
            raise ValueError("Model not trained.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        X = self.training_data[self.feature_names]
        y_true = self.training_data['strength_to_mass_ratio']
        y_pred = self.model.predict(X)
        
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Strength-to-Mass Ratio')
        ax.set_ylabel('Predicted Strength-to-Mass Ratio')
        ax.set_title('Model Predictions vs Actual')
        
        # Feature importance
        ax = axes[0, 1]
        importances = self.model.feature_importances_
        ax.barh(self.feature_names, importances)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance')
        
        # Strength-to-mass vs wall thickness
        ax = axes[1, 0]
        ax.scatter(self.training_data['wall_thickness'], 
                  self.training_data['strength_to_mass_ratio'], alpha=0.6)
        ax.set_xlabel('Wall Thickness')
        ax.set_ylabel('Strength-to-Mass Ratio')
        ax.set_title('Strength-to-Mass vs Wall Thickness')
        
        # Strength-to-mass vs unit cell size
        ax = axes[1, 1]
        ax.scatter(self.training_data['unit_cell_size'], 
                  self.training_data['strength_to_mass_ratio'], alpha=0.6)
        ax.set_xlabel('Unit Cell Size')
        ax.set_ylabel('Strength-to-Mass Ratio')
        ax.set_title('Strength-to-Mass vs Unit Cell Size')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


def main():
    """Main function to run optimization"""
    print("="*60)
    print("XGBoost Optimization for Gyroid Structures")
    print("Maximizing Compressive Strength to Mass Ratio")
    print("="*60)
    
    # Initialize optimizer
    optimizer = GyroidXGBoostOptimizer(
        size_range=(15.0, 25.0),
        resolution=60,  # Lower resolution for faster training
        unit_cell_size_range=(2.0, 6.0),
        wall_thickness_range=(0.1, 0.8),
        smoothness_range=(0.3, 1.0)
    )
    
    # Generate training data
    print("\n" + "="*60)
    print("STEP 1: Generating Training Data")
    print("="*60)
    training_data = optimizer.generate_training_data(n_samples=150, verbose=True)
    
    # Train model
    print("\n" + "="*60)
    print("STEP 2: Training XGBoost Model")
    print("="*60)
    model = optimizer.train_model(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbose=True
    )
    
    # Plot results
    print("\n" + "="*60)
    print("STEP 3: Visualizing Results")
    print("="*60)
    optimizer.plot_training_results(save_path="xgboost_training_results.png")
    
    # Optimize parameters
    print("\n" + "="*60)
    print("STEP 4: Optimizing Parameters")
    print("="*60)
    optimal_params = optimizer.optimize_parameters(
        method='differential_evolution',
        n_iterations=30
    )
    
    # Evaluate optimal parameters
    print("\n" + "="*60)
    print("STEP 5: Evaluating Optimal Parameters")
    print("="*60)
    result = optimizer._evaluate_parameters(
        optimal_params['size'],
        optimal_params['unit_cell_size'],
        optimal_params['wall_thickness'],
        optimal_params['smoothness']
    )
    
    if result:
        print(f"\nActual strength-to-mass ratio: {result['strength_to_mass_ratio']:.2e} m²/s²")
        print(f"Predicted: {optimal_params['predicted_strength_to_mass_ratio']:.2e} m²/s²")
        
        # Recreate analysis object to save STL
        analysis = SimpleGyroidAnalysis(
            size=optimal_params['size'],
            resolution=optimizer.resolution,
            unit_cell_size=optimal_params['unit_cell_size'],
            wall_thickness=optimal_params['wall_thickness'],
            smoothness=optimal_params['smoothness']
        )
        analysis.generate_gyroid()
        
        # Save optimal structure
        verts, faces, normals, values = analysis.create_mesh()
        stl_path = analysis.save_stl(verts, faces, 
                                     filename="gyroid_optimized_xgboost.stl")
        print(f"\nSaved optimal structure to: {stl_path}")
    
    # Find and save best structures from training data
    print("\n" + "="*60)
    print("STEP 6: Saving Best Structures")
    print("="*60)
    best_structures = optimizer.find_best_structures(
        n_best=10,
        save_stl=True,
        output_dir="optimized_gyroids"
    )
    
    # Save model
    print("\n" + "="*60)
    print("STEP 7: Saving Model")
    print("="*60)
    optimizer.save_model("gyroid_xgboost_model.pkl")
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"\nBest structures saved in 'optimized_gyroids/' directory")
    print(f"Model saved as 'gyroid_xgboost_model.pkl'")
    print(f"Training results saved as 'xgboost_training_results.png'")


if __name__ == "__main__":
    main()

