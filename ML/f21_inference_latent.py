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

import optuna
from xgboost import XGBRegressor
import os

def load_training_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    #print(f'Loading training data for target {args.target}')
    if args.target.startswith('PSOJ352'): 
        files = base.get_rms_datafile_list('signalandnoise', args, extn='npy', override_path=override_path)
    else:
        args.extra_file_tag=''
        files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)

    numgroups = samples//args.training_sample_group_size
    X_train = np.zeros((numgroups*len(files), args.latentdim))
    y_train = np.zeros((numgroups*len(files), 2))
    logger.info(f'Created X_train.shape={X_train.shape}, y_train.shape={y_train.shape}')
    
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        """
        ##
        ## Override the xHI and logfX with the accurate values from the simulation data file
        ##
        if curr_xHI > 1 or curr_xHI < 0 or curr_logfX > 1 or curr_logfX < -4:
            logger.error(f'Invalid: curr_xHI={curr_xHI}, curr_logfx={curr_logfX}, file={file}')
            logger.error(file.split('xHI')[1])
            logger.error(file.split('xHI')[1].split('_')[0])
        data = np.fromfile(file)
        logger.info(f'####data:{np.array2string(data[:30], formatter={'float_kind':lambda x: f"{x:.2f},"})}')
        sofilepattern = file.replace('^.*/F21_noisy',f'{args.path}/F21_signalonly').replace('^.*/F21_signalandnoise', '{args.path}/F21_signalonly')
        sofiles = glob.glob(sofilepattern)
        if len(sofiles) == 1:
            data = np.fromfile(sofiles[0])
            #data = np.fromfile(str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%s_rms%.4fmJy_%.1fkHz%s.%s' %
            #       (path, args.redshift,args.log_fx, args.xHI, args.telescope, args.target, args.rms, args.spec_res,args.extra_file_tag,extn))
            logger.info(f'####sodata:{np.array2string(data[:30], formatter={'float_kind':lambda x: f"{x:.2f},"})}')
            curr_logfX = data[2]
            curr_xHI = data[3]
            if curr_xHI > 1 or curr_xHI < 0 or curr_logfX > 1 or curr_logfX < -4:
                logger.error(f'Invalid from sofile: curr_xHI={curr_xHI}, curr_logfx={curr_logfX}, sofile={sofiles[0]}')
        else:
            logger.info(f'Did not find signalonly file for curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        """

        y_train[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        y_train[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        currps = np.load(file)[:samples,:args.latentdim]
        logger.info(f'loaded training data from file {file}. shape: {currps.shape}')
        logger.info(f'Loading data into X_train from rows: {i*numgroups} to {(i+1)*numgroups}')
        if args.training_sample_group_size > 1:
            currps_grouped = currps.reshape(-1, 10, currps.shape[1]).mean(axis=1)
        else:
            currps_grouped = currps

        if i == 0:
            logger.info(f"Original array shape: {currps.shape}")
            logger.info(f"Shape after grouping and taking mean: {currps_grouped.shape}")
            logger.info(f"currps sample:\n{currps[:10,2]}")
            logger.info(f"currps sample grouped:\n{currps_grouped[0][3]}")
        X_train[i*numgroups:(i+1)*numgroups, :] = currps_grouped[:,:]
    return X_train, y_train

def load_test_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    #args.extra_file_tag=''
    args.extra_file_tag='_diffseed'
    if args.target.startswith('PSOJ352'): 
        #args.extra_file_tag='_seed370'
        files = base.get_rms_datafile_list('signalandnoise', args, extn='npy', filter="test_only", override_path=override_path)
    else:
        files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)

    num_samples = 10000
    if args.training_sample_group_size == 1:
        num_samples = args.limitsamplesize
    X_test = np.zeros((num_samples*len(files), args.latentdim))
    y_test = np.zeros((num_samples*len(files), 2))
    logger.info(f'Created X_test.shape={X_test.shape}, y_test.shape={y_test.shape}')
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        """
        ##
        ## Override the xHI and logfX with the accurate values from the simulation data file
        ##
        #logger.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        sofilepattern = file.replace('^.*/F21_noisy',f'{args.path}/F21_signalonly').replace('^.*/F21_signalandnoise', '{args.path}/F21_signalonly')
        sofiles = glob.glob(sofilepattern)
        if len(sofiles) == 1:
            data = np.fromfile(sofiles[0])
            #data = np.fromfile(str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%s_rms%.4fmJy_%.1fkHz%s.%s' %
            #       (path, args.redshift,args.log_fx, args.xHI, args.telescope, args.target, args.rms, args.spec_res,args.extra_file_tag,extn)) 
            logger.info(f'Changing xHI, logfX. From filename: curr_xHI={curr_xHI}, curr_logfx={curr_logfX}, inside SO file: curr_xHI={data[3]}, curr_logfx={data[2]}')
            #curr_xHI = data[1]
            #curr_logfX = data[2]
        else:
            logger.info(f'Did not find signalonly file for curr_xHI={curr_xHI}, curr_logfx={curr_logfX}, pattern used: {sofilepattern}')
        """

        y_test[i*num_samples:(i+1)*num_samples, 0] = curr_xHI
        y_test[i*num_samples:(i+1)*num_samples, 1] = curr_logfX
        currps = np.load(file)[:num_samples,:args.latentdim]
        logger.info(f'loaded testing data from file {file}. shape: {currps.shape}')
        logger.info(f'Loading data into X_test from rows: {i*num_samples} to {(i+1)*num_samples}')

        if args.training_sample_group_size > 1:
            currps_boot = f21stats.bootstrap(ps=currps, reps=10000, size=args.training_sample_group_size)
            X_test[i*num_samples:(i+1)*num_samples, :] = currps_boot
        else:
            X_test[i*num_samples:(i+1)*num_samples, :] = currps
    return X_test, y_test

def save_model(model, modelfile):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {modelfile}')
    model_json = model.save_model(modelfile)

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
    parser.add_argument('--latentdim', type=int, default=256, help='256, 512, etc')

    
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

    # Initialize and train XGBoost model
    logger.info("Training XGBoost model...")
    model = XGBRegressor(
        random_state=42
    )

    logger.info(f"dtype: {X_train.dtype}")
    logger.info(f"min: {np.nanmin(X_train)}")
    logger.info(f"max: {np.nanmax(X_train)}")
    logger.info(f"contains NaN: {np.isnan(X_train).any()}")
    logger.info(f"contains Inf: {np.isinf(X_train).any()}")
    logger.info(f"dtype: {y_train.dtype}")
    logger.info(f"min: {np.nanmin(y_train)}")
    logger.info(f"max: {np.nanmax(y_train)}")
    logger.info(f"contains NaN: {np.isnan(y_train).any()}")
    logger.info(f"contains Inf: {np.isinf(y_train).any()}")

    model.fit(X_train, y_train)

    logger.info(f"Fitted regressor: {model}")
    logger.info(f"Booster: {model.get_booster()}")
    feature_importance = model.feature_importances_
    save_model(model, f'{output_dir}/xgb-f21-inf-ps.json')
    np.savetxt(f"{output_dir}/feature_importance.csv", feature_importance, delimiter=',')
    logger.info(f"Feature importance: {feature_importance}")
    for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
        logger.info(f"Importance type {imp_type}: {model.get_booster().get_score(importance_type=imp_type)}")


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
    logger.info(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main() 
