'''
Model testing module for f21 inference using power spectrum data.
Handles making predictions with trained models.
'''

import numpy as np


def test_model(model, X_test, y_test):
    """
    Test the trained model and make predictions.
    
    Args:
        model: Trained model (XGBRegressor)
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        np.ndarray: Predicted values
    """
    y_pred = model.predict(X_test)
    return y_pred


def save_test_results(y_pred, y_test, output_dir):
    """
    Save test results to files.
    
    Args:
        y_pred (np.ndarray): Predicted values
        y_test (np.ndarray): True values
        output_dir (str): Directory to save results
    """
    import f21_predict_base as base
    base.save_test_results(y_pred, y_test, output_dir) 