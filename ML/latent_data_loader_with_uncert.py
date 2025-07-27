'''
Latent Data loading module for f21 inference.
Handles loading of training and test data from npy files.
Calculates std deviation as a measure of uncertainty in the features.
'''

import numpy as np
import f21_predict_base as base
import F21Stats as f21stats
from logging import info 

def load_features_data(override_path, samples, samples_start_index, group_size, args):
    """
    Load training data from npy files.
    
    Args:
        override_path (str): Path to override the default data path
        samples (int): Number of samples to load
        args: Arguments object containing configuration
        
    Returns:
        tuple: (X, y) - Features and labels for training or testing
    """
    files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)
    numgroups = samples//group_size
    X = np.zeros((numgroups*len(files), 256))
    y = np.zeros((numgroups*len(files), 3))
    
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        ##
        ## Override the xHI and logfX with the accurate values from the simulation data file
        ##
        if i == 0: info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        data = np.fromfile('%sF21_signalonly_21cmFAST_200Mpc_z6.0_fX%.2f_xHI%.2f_8kHz.dat' % (args.path,curr_logfX,curr_xHI),dtype=np.float32)
        #print(f'###data:{data[:20]}')
        curr_xHI = data[1]
        curr_logfX = data[2]
        if i == 0: info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')

        y[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        y[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        currfeat = np.load(file)[samples_start_index:samples_start_index+samples]
        currfeat_grouped = currfeat.reshape(-1, 10, currfeat.shape[1]).mean(axis=1)
        currfeat_sigma = currfeat.reshape(-1, 10, currfeat.shape[1]).std(axis=1)/np.abs(currfeat_grouped)
        y[i*numgroups:(i+1)*numgroups, 2] = currfeat_sigma.mean()

        if y[i*numgroups:(i+1)*numgroups, 2].any() < 0:
            info(f"Original array shape: {currfeat.shape}")
            info(f"Shape after grouping and taking mean: {currfeat_grouped.shape}")
            info(f"currps sample:\n{currfeat[0]}")
            info(f"currps sample grouped:\n{currfeat_grouped}")
        X[i*numgroups:(i+1)*numgroups, :] = currfeat_grouped[:,:]
    return X, y

