#!/usr/bin/env python3
"""
Script to predict simulation results based on parameter inputs using machine learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Target metrics to predict
TARGET_METRICS = [
    'mazars_compressive_strength_MPa',
    'mazars_tensile_strength_MPa',
    'mazars_total_energy_absorption_J',
    'mazars_max_damage',
    'earthquake_max_displacement_mm',
    'earthquake_max_damage',
    'earthquake_max_stress_MPa',
]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load simulation results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['thickness', 'threshold', 'span', 'size',
                   'mazars_cross_sectional_area_m2',
                   'mazars_compressive_strength_MPa',
                   'mazars_tensile_strength_MPa',
                   'mazars_total_energy_absorption_J',
                   'mazars_max_damage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for machine learning.
    Returns X (features) and y (targets) arrays, plus encoders.
    """
    # Feature columns
    feature_cols = ['formula_name', 'thickness', 'threshold', 'span']
    
    # Add radial gradient parameters if available
    if 'radial_center_threshold' in df.columns:
        feature_cols.append('radial_center_threshold')
    if 'radial_edge_threshold' in df.columns:
        feature_cols.append('radial_edge_threshold')
    
    # Get available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Prepare X (features)
    X = df[available_features].copy()
    
    # Encode categorical variables
    label_encoders = {}
    if 'formula_name' in X.columns:
        le = LabelEncoder()
        X['formula_name_encoded'] = le.fit_transform(X['formula_name'])
        label_encoders['formula_name'] = le
        X = X.drop('formula_name', axis=1)
    
    # Handle None/NaN values in radial gradient parameters
    for col in X.columns:
        if 'radial' in col.lower():
            X[col] = X[col].fillna(-999)  # Use -999 as a sentinel for None
    
    # Prepare y (targets) - only use available metrics
    available_targets = [m for m in TARGET_METRICS if m in df.columns]
    y = df[available_targets].copy()
    
    # Remove rows with missing target values
    valid_mask = ~y.isna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y, label_encoders, available_features


def train_models(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
    """
    Train models for each target metric.
    Returns dictionary of trained models and their performance metrics.
    """
    models = {}
    performance = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"Features: {list(X.columns)}\n")
    
    for metric in y.columns:
        print(f"Training model for {metric}...")
        
        # Get target values
        y_train_metric = y_train[metric].values
        y_test_metric = y_test[metric].values
        
        # Remove any remaining NaN values
        valid_train = ~np.isnan(y_train_metric)
        valid_test = ~np.isnan(y_test_metric)
        
        if valid_train.sum() < 10:  # Need at least 10 samples
            print(f"  Skipping {metric}: insufficient data")
            continue
        
        X_train_clean = X_train[valid_train]
        y_train_clean = y_train_metric[valid_train]
        X_test_clean = X_test[valid_test]
        y_test_clean = y_test_metric[valid_test]
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        # Make predictions
        y_pred_train = model.predict(X_train_clean)
        y_pred_test = model.predict(X_test_clean)
        
        # Calculate metrics
        train_r2 = r2_score(y_train_clean, y_pred_train)
        test_r2 = r2_score(y_test_clean, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_test))
        test_mae = mean_absolute_error(y_test_clean, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_clean, y_train_clean, 
                                   cv=5, scoring='r2')
        
        models[metric] = model
        performance[metric] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_samples': len(y_train_clean)
        }
        
        print(f"  Train R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}")
        print(f"  Test MAE: {test_mae:.3f}")
        print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")
    
    return models, performance, X.columns.tolist()


def predict(model_dict: dict, feature_names: list, label_encoders: dict,
           formula_name: str, thickness: float, threshold: float = 0.0,
           span: float = None, radial_center_threshold: float = None,
           radial_edge_threshold: float = None) -> dict:
    """
    Make predictions for a given set of parameters.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary of trained models
    feature_names : list
        List of feature column names
    label_encoders : dict
        Label encoders for categorical variables
    formula_name : str
        Formula name (e.g., 'gyroid', 'diamond')
    thickness : float
        Thickness parameter
    threshold : float
        Threshold parameter (default: 0.0)
    span : float
        Span parameter (optional)
    radial_center_threshold : float
        Radial center threshold (optional, use -999 for None)
    radial_edge_threshold : float
        Radial edge threshold (optional, use -999 for None)
    
    Returns:
    --------
    dict : Predictions for all metrics
    """
    # Prepare feature vector
    features = {}
    
    # Encode formula name
    if 'formula_name' in label_encoders:
        le = label_encoders['formula_name']
        if formula_name in le.classes_:
            features['formula_name_encoded'] = le.transform([formula_name])[0]
        else:
            raise ValueError(f"Unknown formula: {formula_name}. Available: {list(le.classes_)}")
    
    # Add numeric features
    features['thickness'] = thickness
    if 'threshold' in feature_names:
        features['threshold'] = threshold
    
    if span is not None and 'span' in feature_names:
        features['span'] = span
    
    if radial_center_threshold is not None and 'radial_center_threshold' in feature_names:
        features['radial_center_threshold'] = radial_center_threshold if radial_center_threshold != -999 else -999
    elif 'radial_center_threshold' in feature_names:
        features['radial_center_threshold'] = -999
    
    if radial_edge_threshold is not None and 'radial_edge_threshold' in feature_names:
        features['radial_edge_threshold'] = radial_edge_threshold if radial_edge_threshold != -999 else -999
    elif 'radial_edge_threshold' in feature_names:
        features['radial_edge_threshold'] = -999
    
    # Create feature array in correct order
    X_pred = np.array([[features.get(col, 0) for col in feature_names]])
    
    # Make predictions
    predictions = {}
    for metric, model in model_dict.items():
        pred = model.predict(X_pred)[0]
        predictions[metric] = float(pred)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Train models to predict simulation results from parameters'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to simulation_results.csv file'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train new models (default: load existing models)'
    )
    parser.add_argument(
        '--predict',
        type=str,
        nargs='+',
        help='Make prediction: --predict formula_name thickness [threshold] [span] [radial_center] [radial_edge]'
    )
    parser.add_argument(
        '--predict-file',
        type=str,
        help='CSV file with parameters to predict (columns: formula_name, thickness, threshold, span, etc.)'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)
    
    if args.train or not (model_dir / 'models.pkl').exists():
        # Load data and train models
        print(f"Loading data from {args.csv_path}...")
        df = load_data(args.csv_path)
        print(f"Loaded {len(df)} rows\n")
        
        # Prepare features
        X, y, label_encoders, feature_names = prepare_features(df)
        print(f"Prepared {len(X)} samples with {len(X.columns)} features")
        print(f"Target metrics: {list(y.columns)}\n")
        
        # Train models
        models, performance, feature_names = train_models(X, y)
        
        # Save models
        with open(model_dir / 'models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        with open(model_dir / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        with open(model_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        
        with open(model_dir / 'performance.json', 'w') as f:
            json.dump(performance, f, indent=2)
        
        print(f"\nModels saved to {model_dir}/")
        print(f"Performance metrics saved to {model_dir}/performance.json")
    
    # Load models for prediction
    if args.predict or args.predict_file:
        print("\nLoading models...")
        with open(model_dir / 'models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        with open(model_dir / 'label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open(model_dir / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        if args.predict:
            # Single prediction from command line
            if len(args.predict) < 2:
                print("Error: Need at least formula_name and thickness")
                return
            
            formula_name = args.predict[0]
            thickness = float(args.predict[1])
            threshold = float(args.predict[2]) if len(args.predict) > 2 else 0.0
            span = float(args.predict[3]) if len(args.predict) > 3 else None
            radial_center = float(args.predict[4]) if len(args.predict) > 4 else None
            radial_edge = float(args.predict[5]) if len(args.predict) > 5 else None
            
            predictions = predict(models, feature_names, label_encoders,
                                formula_name, thickness, threshold, span,
                                radial_center, radial_edge)
            
            print(f"\nPredictions for {formula_name}, thickness={thickness}, threshold={threshold}:")
            for metric, value in predictions.items():
                print(f"  {metric}: {value:.3f}")
        
        elif args.predict_file:
            # Batch prediction from file
            pred_df = pd.read_csv(args.predict_file)
            results = []
            
            for idx, row in pred_df.iterrows():
                formula_name = row['formula_name']
                thickness = row['thickness']
                threshold = row.get('threshold', 0.0)
                span = row.get('span', None)
                radial_center = row.get('radial_center_threshold', None)
                radial_edge = row.get('radial_edge_threshold', None)
                
                predictions = predict(models, feature_names, label_encoders,
                                    formula_name, thickness, threshold, span,
                                    radial_center, radial_edge)
                
                result = {
                    'formula_name': formula_name,
                    'thickness': thickness,
                    'threshold': threshold,
                    'span': span,
                    'radial_center_threshold': radial_center,
                    'radial_edge_threshold': radial_edge,
                    **predictions
                }
                results.append(result)
            
            output_df = pd.DataFrame(results)
            output_path = Path(args.predict_file).stem + '_predictions.csv'
            output_df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")


if __name__ == '__main__':
    main()

