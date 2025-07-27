'''
Model testing module for f21 inference using power spectrum data.
Handles making predictions with trained models.
'''

import numpy as np
import tempfile
import os
from logging import info


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


def save_test_results(y_pred, y_test, output_dir, label = ''):
    """
    Save test results to files.
    
    Args:
        y_pred (np.ndarray): Predicted values
        y_test (np.ndarray): True values
        output_dir (str): Directory to save results
    """
    filename = f"{output_dir}/test_results{label}.csv"
    
    # Combine predictions and test values
    add_columns = ''
    if y_pred.shape[1] > 2: add_columns = ",sigma" * (y_pred.shape[1] - 2)
    header = "pred_xHI,pred_logfX" + add_columns + ",test_xHI,test_logfX" + add_columns
    combined = np.hstack((y_pred, y_test))
    
    # Save to CSV
    np.savetxt(filename, combined, delimiter=',', header=header, comments='')
    info(f"Saved test results to {filename}")

def sample_2d_point(mu_x, mu_y, stdev):
    """
    Generate a 2D point from a normal distribution centered at (mu_x, mu_y)
    with standard deviation stdev for both axes.
    """
    point = np.random.normal(loc=[mu_x, mu_y], scale=stdev, size=2)
    return point


def generate_posterior_samples(y_pred, y_test, numsamples, output_dir, label = ''):
    """
    Generate posterior samples based on predicted uncertainty and save results to file.
    
    Args:
        y_pred (np.ndarray): Predicted values with sigma 
        y_test (np.ndarray): True values
        output_dir (str): Directory to save results
    """
    filename = f"{output_dir}/generated_posterior_samples{label}.npy"
    
    # Combine predictions and test values
    if y_pred.shape[1] <= 2: raise error('expected sigma column in prediction')
    #header = "pred_xHI,pred_logfX," + add_columns + "test_xHI,test_logfX" + add_columns
    y_gen = np.zeros((y_pred.shape[0]*numsamples, 4))
    for i, (pred_x, pred_f, pred_sigma) in enumerate(y_pred):
        for j in range(numsamples):
            y_gen[i*numsamples+j,:2] = sample_2d_point(pred_x, pred_f, pred_sigma)
            y_gen[i*numsamples+j,2:] = y_test[i,:2]
            if i == 0 and j == 0: 
                info(f"Generated: {pred_x}, {pred_f}, {pred_sigma}: {y_gen[i*numsamples+j]}")

    
    # Save to CSV
    #np.save(filename, y_gen)
    #info(f"Saved generated posterior samples to {filename}")
    #print(f"Saved generated posterior samples to {filename}")
    info(f"Generated {len(y_gen)} posterior samples")
    return y_gen
    

