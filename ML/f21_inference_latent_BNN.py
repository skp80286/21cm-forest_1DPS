'''
Use Latent features data to infer cosmological parameters.
XGBoostRegressor is trained based on labelled data of 529 parameter combinations. 
The same is then tested on 5 parameter combinations specially selected as test points.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import f21_predict_base as base
import F21Stats as f21stats
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats

import numpy as np
import sys

import matplotlib.pyplot as plt
from BayesianNNRegressor import BayesianNNRegressor

import optuna
import os

def load_training_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    args.extra_file_tag=''
    files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)
    numgroups = samples//args.training_sample_group_size
    X_train = np.zeros((numgroups*len(files), 256))
    y_train = np.zeros((numgroups*len(files), 2))
    
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        ##
        ## Override the xHI and logfX with the accurate values from the simulation data file
        ##
        #logger.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        data = np.fromfile('%sF21_signalonly_21cmFAST_200Mpc_z6.0_fX%.2f_xHI%.2f_8kHz.dat' % (args.path,curr_logfX,curr_xHI),dtype=np.float32)
        #logger.info(f'###data:{data[:20]}')
        curr_xHI = data[1]
        curr_logfX = data[2]
        #logger.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')

        y_train[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        y_train[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        currps = np.load(file)[:samples,:256]
        #logger.info(f'loaded training data from file. shape: {currps.shape}')
        currps_grouped = currps.reshape(-1, 10, currps.shape[1]).mean(axis=1)

        if i == 0:
            logger.info(f"Original array shape: {currps.shape}")
            logger.info(f"Shape after grouping and taking mean: {currps_grouped.shape}")
            logger.info(f"currps sample:\n{currps[:10,2]}")
            logger.info(f"currps sample grouped:\n{currps_grouped[0][3]}")
        X_train[i*numgroups:(i+1)*numgroups, :] = currps_grouped[:,:]
    return X_train, y_train

def load_test_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    args.extra_file_tag='_diffseed'
    files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)
    X_test = np.zeros((10000*len(files), 256))
    y_test = np.zeros((10000*len(files), 2))
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        ##
        ## Override the xHI and logfX with the accurate values from the simulation data file
        ##
        logger.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        data = np.fromfile('%sF21_signalonly_21cmFAST_200Mpc_z6.0_fX%.2f_xHI%.2f_8kHz.dat' % (args.path,curr_logfX,curr_xHI),dtype=np.float32)
        #logger.info(f'###data:{data[:20]}')
        curr_xHI = data[1]
        curr_logfX = data[2]
        logger.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        
        y_test[i*10000:(i+1)*10000, 0] = curr_xHI
        y_test[i*10000:(i+1)*10000, 1] = curr_logfX
        currps = np.load(file)[samples:,:256]
        logger.info(f'loaded training data from file. shape: {currps.shape}')

        currps_boot = f21stats.bootstrap(ps=currps, reps=10000, size=10)
        X_test[i*10000:(i+1)*10000, :] = currps_boot
    return X_test, y_test

logger = None
output_dir = None
args = None
# main code starts here
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.determinisitc=True
    torch.backends.cudnn.benchmark=False

    parser = base.setup_args_parser()
    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_500/f21_ps_dum_train_test_uGMRT_t500.0_20250511105815/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_500/f21_ps_dum_train_test_uGMRT_t500.0_20250511105815/test_ps/", help='')
    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_g50/f21_ps_dum_train_test_uGMRT_t50.0_20250410153928/ps/", help='model file')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_g50/f21_ps_dum_train_test_uGMRT_t50.0_20250410153928/test_ps/", help='model file')
    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_ska/f21_ps_dum_train_test_SKA1-low_t50.0_20250511105922/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_ska/f21_ps_dum_train_test_SKA1-low_t50.0_20250511105922/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_500/f21_unet_ps_dum_train_test_uGMRT_t500.0_20250511164401/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_500/f21_unet_ps_dum_train_test_uGMRT_t500.0_20250511164401/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_500/mixed_f21_unet_ps_dum_train_test_uGMRT_t500.0_20250604091744/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_500/mixed_f21_unet_ps_dum_train_test_uGMRT_t500.0_20250604091744/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_50/mixed_f21_unet_ps_dum_train_test_uGMRT_t50.0_20250607223018/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_50/mixed_f21_unet_ps_dum_train_test_uGMRT_t50.0_20250607223018/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_ska/f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250511164401/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_ska/f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250511164401/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_ska/mixed_f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250608062755/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_ska/mixed_f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250608062755/test_ps/", help='')

    #../data/denoised_gmrt50h/f21_unet_ps_dum_train_test_uGMRT_t50.0_20250417191012/denoised_ps

    parser.add_argument('--datapath', type=str, help='PS data path')
    parser.add_argument('--testdatapath', type=str, help='test PS data path')
    parser.add_argument('--training_sample_group_size', type=int, default=10, help='Number of samples of spectrum to be grouped')
    
    args = parser.parse_args()

    if args.datapath is None:
        ## Set the datapath

        # noisy
        if args.telescope == 'uGMRT' and args.t_int == 50:
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/latent_g50/f21_unet_latent_dum_train_test_uGMRT_t50.0_20250720131101/latent/"
        if args.telescope == 'uGMRT' and args.t_int == 500:
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/latent_g500/f21_unet_latent_dum_train_test_uGMRT_t500.0_20250720193911/latent/"
        if args.telescope == 'SKA1-low' and args.t_int == 50:
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/latent_ska50/f21_unet_latent_dum_train_test_SKA1-low_t50.0_20250720181435/latent/"

        ## Set the testdatapath

        # noisy
        if args.telescope == 'uGMRT' and args.t_int == 50:
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/latent_g50/f21_unet_latent_dum_train_test_uGMRT_t50.0_20250822101747_diffseed/test_latent/"
        if args.telescope == 'uGMRT' and args.t_int == 500:
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/latent_g500/f21_unet_latent_dum_train_test_uGMRT_t500.0_20250822070209_diffseed/test_latent/"
        if args.telescope == 'SKA1-low' and args.t_int == 50:
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/latent_ska50/f21_unet_latent_dum_train_test_SKA1-low_t50.0_20250822101929_diffseed/test_latent/"

    output_dir = base.create_output_dir(args=args)
    global logger
    logger = base.setup_logging(output_dir)

    # Initialize the network
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info("####")
    logger.info(f"### Using \"{device}\" device ###")
    logger.info("####")

    # Load training and test data
    logger.info(f"Loading training data from {args.datapath}...")
    X_train, y_train = load_training_data(override_path=args.datapath, samples=args.limitsamplesize, args=args)
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    logger.info(f"Loading test data from {args.testdatapath}...")
    X_test, y_test = load_test_data(override_path=args.testdatapath, samples=args.limitsamplesize, args=args)
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Initialize and train BNN model
    logger.info("Training BNN model...")
    model = BayesianNNRegressor(
        hidden_dims=[128, 64, 32],
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        random_state=42,
        device=device
    )
    model.fit(X_train, y_train)

    logger.info(f"Fitted regressor: {model}")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    logger.info("\nModel Performance:")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    base.save_test_results(y_pred, y_test, output_dir)

    # Generate posterior distribution plot
    logger.info("Generating posterior distribution plot...")
    model.plot_posterior_distribution(X_test, y_test, output_dir=output_dir, save_plot=True)
    logger.info(f"\nResults saved to {output_dir}")

    # Save the trained model
    model_path = os.path.join(output_dir, 'bayesian_nn_model.pth')
    model.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    """
    pltr.summarize_test_1000(y_pred, y_test, output_dir=output_dir, showplots=False, saveplots=True, label=f"{args.telescope}, {args.t_int:.0f}h")
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot xHI predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    plt.xlabel('True xHI')
    plt.ylabel('Predicted xHI')
    plt.title('xHI Predictions')
    
    # Plot logfX predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 'r--')
    plt.xlabel('True logfX')
    plt.ylabel('Predicted logfX')
    plt.title('logfX Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.pdf'), format='pdf')

    """

if __name__ == '__main__':
    main() 