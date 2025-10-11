'''
Use Latent features data to infer cosmological parameters using Simulation Based Inference (SBI).
XGBoostRegressor is trained based on labelled data of 529 parameter combinations. 
The same is then used for Bayesian inference with MCMC sampling to generate posterior distributions.
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
import corner

import optuna
from xgboost import XGBRegressor
import os
import emcee
from scipy.optimize import minimize

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
        logger.info(f'loaded training data from file. shape: {currps.shape}')
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

def save_model(model, modelfile):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {modelfile}')
    model_json = model.save_model(modelfile)

def log_prior(theta):
    """
    Define uniform prior for xHI (0 to 1) and logfX (-4 to +1)
    """
    xHI, logfX = theta
    if 0.0 <= xHI <= 1.0 and -4.0 <= logfX <= 1.0:
        return 0.0
    return -np.inf

def log_likelihood(theta, model, observed_features):
    """
    Define likelihood function using XGBoost model predictions
    """
    xHI, logfX = theta
    
    # Predict using the trained model
    predicted_params = model.predict(observed_features.reshape(1, -1))[0]
    pred_xHI, pred_logfX = predicted_params
    
    # Calculate likelihood based on prediction error
    # Using a simple Gaussian likelihood with fixed variance
    sigma = 0.1  # Fixed uncertainty for simplicity
    log_like = -0.5 * ((xHI - pred_xHI)**2 + (logfX - pred_logfX)**2) / (sigma**2)
    
    return log_like

def log_posterior(theta, model, observed_features):
    """
    Define posterior function
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, observed_features)

def run_mcmc_inference(model, observed_features, true_params, n_walkers=32, n_steps=10000, burn_in=1000):
    """
    Run MCMC inference for a single test point
    """
    ndim = 2
    
    # Initialize walkers around the true parameters with some scatter
    initial_pos = true_params + 0.1 * np.random.randn(n_walkers, ndim)
    
    # Ensure initial positions are within prior bounds
    initial_pos[:, 0] = np.clip(initial_pos[:, 0], 0.0, 1.0)  # xHI
    initial_pos[:, 1] = np.clip(initial_pos[:, 1], -4.0, 1.0)  # logfX
    
    # Set up the sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, 
                                  args=(model, observed_features))
    
    # Run MCMC
    logger.info(f"Running MCMC with {n_walkers} walkers for {n_steps} steps...")
    state = sampler.run_mcmc(initial_pos, n_steps, progress=True)
    
    # Get samples after burn-in
    samples = sampler.get_chain(discard=burn_in, thin=10, flat=True)
    
    return samples, sampler

def plot_posterior_distributions(samples_list, true_params_list, output_dir):
    """
    Plot posterior distributions for all test points
    """
    n_test_points = len(samples_list)
    
    # Create corner plots for each test point
    for i, (samples, true_params) in enumerate(zip(samples_list, true_params_list)):
        fig = corner.corner(samples, 
                          labels=[r'$x_{\rm HI}$', r'$\log_{10}(f_{\rm X})$'],
                          truths=true_params,
                          truth_color='red',
                          show_titles=True,
                          title_fmt='.3f',
                          levels=[0.68, 0.95],
                          smooth=True)
        
        fig.suptitle(f'Posterior Distribution - Test Point {i+1}\nTrue: xHI={true_params[0]:.3f}, logfX={true_params[1]:.3f}', 
                    fontsize=14)
        
        plt.savefig(f'{output_dir}/posterior_test_point_{i+1}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create combined plot showing all posteriors
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot xHI distributions
    ax1 = axes[0, 0]
    for i, (samples, true_params) in enumerate(zip(samples_list, true_params_list)):
        ax1.hist(samples[:, 0], bins=50, alpha=0.6, label=f'Test {i+1}', density=True)
        ax1.axvline(true_params[0], color=f'C{i}', linestyle='--', alpha=0.8)
    ax1.set_xlabel(r'$x_{\rm HI}$')
    ax1.set_ylabel('Density')
    ax1.set_title('Posterior Distributions - xHI')
    ax1.legend()
    
    # Plot logfX distributions
    ax2 = axes[0, 1]
    for i, (samples, true_params) in enumerate(zip(samples_list, true_params_list)):
        ax2.hist(samples[:, 1], bins=50, alpha=0.6, label=f'Test {i+1}', density=True)
        ax2.axvline(true_params[1], color=f'C{i}', linestyle='--', alpha=0.8)
    ax2.set_xlabel(r'$\log_{10}(f_{\rm X})$')
    ax2.set_ylabel('Density')
    ax2.set_title('Posterior Distributions - logfX')
    ax2.legend()
    
    # Plot 2D scatter plots
    ax3 = axes[1, 0]
    for i, (samples, true_params) in enumerate(zip(samples_list, true_params_list)):
        ax3.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1, label=f'Test {i+1}')
        ax3.scatter(true_params[0], true_params[1], color=f'C{i}', marker='x', s=100, linewidth=3)
    ax3.set_xlabel(r'$x_{\rm HI}$')
    ax3.set_ylabel(r'$\log_{10}(f_{\rm X})$')
    ax3.set_title('2D Posterior Samples')
    ax3.legend()
    
    # Plot summary statistics
    ax4 = axes[1, 1]
    xHI_means = [np.mean(samples[:, 0]) for samples in samples_list]
    xHI_stds = [np.std(samples[:, 0]) for samples in samples_list]
    logfX_means = [np.mean(samples[:, 1]) for samples in samples_list]
    logfX_stds = [np.std(samples[:, 1]) for samples in samples_list]
    
    x = np.arange(n_test_points)
    width = 0.35
    
    ax4.bar(x - width/2, xHI_means, width, yerr=xHI_stds, label=r'$x_{\rm HI}$', alpha=0.7)
    ax4.bar(x + width/2, logfX_means, width, yerr=logfX_stds, label=r'$\log_{10}(f_{\rm X})$', alpha=0.7)
    ax4.set_xlabel('Test Point')
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Posterior Means and Uncertainties')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Test {i+1}' for i in range(n_test_points)])
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/posterior_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def save_mcmc_results(samples_list, true_params_list, output_dir):
    """
    Save MCMC results to files
    """
    for i, (samples, true_params) in enumerate(zip(samples_list, true_params_list)):
        # Save samples
        np.save(f'{output_dir}/mcmc_samples_test_point_{i+1}.npy', samples)
        
        # Calculate and save summary statistics
        xHI_mean = np.mean(samples[:, 0])
        xHI_std = np.std(samples[:, 0])
        xHI_16, xHI_50, xHI_84 = np.percentile(samples[:, 0], [16, 50, 84])
        
        logfX_mean = np.mean(samples[:, 1])
        logfX_std = np.std(samples[:, 1])
        logfX_16, logfX_50, logfX_84 = np.percentile(samples[:, 1], [16, 50, 84])
        
        summary = {
            'test_point': i + 1,
            'true_xHI': true_params[0],
            'true_logfX': true_params[1],
            'xHI_mean': xHI_mean,
            'xHI_std': xHI_std,
            'xHI_16': xHI_16,
            'xHI_50': xHI_50,
            'xHI_84': xHI_84,
            'logfX_mean': logfX_mean,
            'logfX_std': logfX_std,
            'logfX_16': logfX_16,
            'logfX_50': logfX_50,
            'logfX_84': logfX_84
        }
        
        # Save summary to text file
        with open(f'{output_dir}/mcmc_summary_test_point_{i+1}.txt', 'w') as f:
            f.write(f"MCMC Results - Test Point {i+1}\n")
            f.write("=" * 50 + "\n")
            f.write(f"True parameters: xHI={true_params[0]:.4f}, logfX={true_params[1]:.4f}\n")
            f.write(f"\nInferred xHI:\n")
            f.write(f"  Mean: {xHI_mean:.4f} ± {xHI_std:.4f}\n")
            f.write(f"  Median: {xHI_50:.4f}\n")
            f.write(f"  16th percentile: {xHI_16:.4f}\n")
            f.write(f"  84th percentile: {xHI_84:.4f}\n")
            f.write(f"\nInferred logfX:\n")
            f.write(f"  Mean: {logfX_mean:.4f} ± {logfX_std:.4f}\n")
            f.write(f"  Median: {logfX_50:.4f}\n")
            f.write(f"  16th percentile: {logfX_16:.4f}\n")
            f.write(f"  84th percentile: {logfX_84:.4f}\n")

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
    parser.add_argument('--datapath', type=str, help='PS data path')
    parser.add_argument('--testdatapath', type=str, help='test PS data path')
    parser.add_argument('--training_sample_group_size', type=int, default=10, help='Number of samples of spectrum to be grouped')
    parser.add_argument('--n_test_points', type=int, default=5, help='Number of test points for inference')
    parser.add_argument('--n_walkers', type=int, default=32, help='Number of MCMC walkers')
    parser.add_argument('--n_steps', type=int, default=10000, help='Number of MCMC steps')
    parser.add_argument('--burn_in', type=int, default=1000, help='Number of burn-in steps')
    
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

    # Select 5 test points for MCMC inference
    logger.info(f"Selecting {args.n_test_points} test points for MCMC inference...")
    
    # Select test points with different parameter combinations
    test_indices = np.linspace(0, len(X_test)-1, args.n_test_points, dtype=int)
    test_features = X_test[test_indices]
    test_params = y_test[test_indices]
    
    logger.info(f"Selected test points:")
    for i, (features, params) in enumerate(zip(test_features, test_params)):
        logger.info(f"  Test point {i+1}: xHI={params[0]:.4f}, logfX={params[1]:.4f}")

    # Run MCMC inference for each test point
    logger.info("Starting MCMC inference...")
    samples_list = []
    true_params_list = []
    
    for i, (features, true_params) in enumerate(zip(test_features, test_params)):
        logger.info(f"Running MCMC for test point {i+1}/{args.n_test_points}...")
        
        # Run MCMC
        samples, sampler = run_mcmc_inference(
            model, features, true_params, 
            n_walkers=args.n_walkers, 
            n_steps=args.n_steps, 
            burn_in=args.burn_in
        )
        
        samples_list.append(samples)
        true_params_list.append(true_params)
        
        # Log autocorrelation time
        try:
            tau = sampler.get_autocorr_time()
            logger.info(f"Autocorrelation time for test point {i+1}: xHI={tau[0]:.1f}, logfX={tau[1]:.1f}")
        except:
            logger.info(f"Could not compute autocorrelation time for test point {i+1}")

    # Save MCMC results
    logger.info("Saving MCMC results...")
    save_mcmc_results(samples_list, true_params_list, output_dir)

    # Plot posterior distributions
    logger.info("Creating posterior distribution plots...")
    plot_posterior_distributions(samples_list, true_params_list, output_dir)

    logger.info(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main() 