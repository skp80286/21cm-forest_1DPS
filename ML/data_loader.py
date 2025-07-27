'''
Data loading module for f21 inference using power spectrum data.
Handles loading of training and test data from CSV files.
'''

import numpy as np
import f21_predict_base as base
import F21Stats as f21stats
from logging import info 

def load_training_data(override_path, samples, args):
    """
    Load training data from CSV files.
    
    Args:
        override_path (str): Path to override the default data path
        samples (int): Number of samples to load
        args: Arguments object containing configuration
        
    Returns:
        tuple: (X_train, y_train) - Features and labels for training
    """
    files = base.get_datafile_list('noisy', args, extn='csv', override_path=override_path)
    numgroups = samples//args.training_sample_group_size
    X_train = np.zeros((numgroups*len(files), 16))
    y_train = np.zeros((numgroups*len(files), 2))
    
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        y_train[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        y_train[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        currps = np.loadtxt(file)[:samples,:16]
        currps_grouped = currps.reshape(-1, 10, currps.shape[1]).mean(axis=1)

        if i == 0:
            info(f"Original array shape: {currps.shape}")
            info(f"Shape after grouping and taking mean: {currps_grouped.shape}")
            info(f"currps sample:\n{currps[:10,2]}")
            info(f"currps sample grouped:\n{currps_grouped[0][3]}")
        X_train[i*numgroups:(i+1)*numgroups, :] = currps_grouped[:,:]
    return X_train, y_train


def load_test_data(override_path, samples, args):
    """
    Load test data from CSV files.
    
    Args:
        override_path (str): Path to override the default data path
        samples (int): Number of samples to load
        args: Arguments object containing configuration
        
    Returns:
        tuple: (X_test, y_test) - Features and labels for testing
    """
    files = base.get_datafile_list('noisy', args, extn='csv', override_path=override_path)
    X_test = np.zeros((10000*len(files), 16))
    y_test = np.zeros((10000*len(files), 2))
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])
        
        y_test[i*10000:(i+1)*10000, 0] = curr_xHI
        y_test[i*10000:(i+1)*10000, 1] = curr_logfX
        currps = np.loadtxt(file)[samples:,:16]
        # info shape and parameters for the first file
        if i == 0:
            info(f"Loading test data. File: {file}")
            info(f"samples: {samples}, curr_xHI: {curr_xHI}, curr_logfX: {curr_logfX}")
            info(f"currps shape: {currps.shape}")
            info("-" * 50)
        
        currps_boot = f21stats.bootstrap(ps=currps, reps=10000, size=10)
        X_test[i*10000:(i+1)*10000, :] = currps_boot
    return X_test, y_test 