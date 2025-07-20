'''
Metrics calculation module for f21 inference using power spectrum data.
Handles calculation of performance metrics for regression models.
'''

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from logging import info 


def calculate_metrics(y_test, y_pred):
    """
    Calculate performance metrics for regression results.
    
    Args:
        y_test (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate metrics for each output variable separately
    metrics_per_output = {}
    for i, output_name in enumerate(['xHI', 'logfX']):
        r2_single = r2_score(y_test[:, i], y_pred[:, i])
        mse_single = mean_squared_error(y_test[:, i], y_pred[:, i])
        rmse_single = np.sqrt(mse_single)
        
        metrics_per_output[output_name] = {
            'r2': r2_single,
            'mse': mse_single,
            'rmse': rmse_single
        }
    
    metrics = {
        'overall': {
            'r2': r2,
            'mse': mse,
            'rmse': rmse
        },
        'per_output': metrics_per_output
    }
    
    return metrics


def print_metrics(metrics, logger=None):
    """
    Print performance metrics to console or logger.
    
    Args:
        metrics (dict): Dictionary containing performance metrics
        logger: Logger object (optional)
    """
    output_func = logger.info if logger else print
    
    output_func("\nModel Performance:")
    output_func(f"Overall R2 Score: {metrics['overall']['r2']:.4f}")
    output_func(f"Overall MSE: {metrics['overall']['mse']:.4f}")
    output_func(f"Overall RMSE: {metrics['overall']['rmse']:.4f}")
    
    output_func("\nPer-output metrics:")
    for output_name, output_metrics in metrics['per_output'].items():
        output_func(f"{output_name}:")
        output_func(f"  R2 Score: {output_metrics['r2']:.4f}")
        output_func(f"  MSE: {output_metrics['mse']:.4f}")
        output_func(f"  RMSE: {output_metrics['rmse']:.4f}")


def save_metrics(metrics, output_dir):
    """
    Save metrics to a file.
    
    Args:
        metrics (dict): Dictionary containing performance metrics
        output_dir (str): Directory to save metrics
    """
    import os
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    metrics_serializable = convert_numpy_types(metrics)
    
    metrics_file = os.path.join(output_dir, "performance_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    info(f"Metrics saved to: {metrics_file}") 