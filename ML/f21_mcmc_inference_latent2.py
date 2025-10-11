'''
Use Latent features data to infer cosmological parameters using Simulation Based Inference (SBI).
XGBoostRegressor is trained to predict latent features from parameters (xHI, logfX).
The trained regressor acts as a simulator for likelihood evaluation in MCMC sampling.
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
    X_train = np.zeros((numgroups*len(files), 2))  # Parameters as input
    y_train = np.zeros((numgroups*len(files), 256))  # Latent features as output
    
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

        # Parameters as input (X_train)
        X_train[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        X_train[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        
        # Load latent features
        currps = np.load(file)[:samples,:256]
        logger.info(f'loaded training data from file. shape: {currps.shape}')
        currps_grouped = currps.reshape(-1, 10, currps.shape[1]).mean(axis=1)

        if i == 0:
            logger.info(f"Original array shape: {currps.shape}")
            logger.info(f"Shape after grouping and taking mean: {currps_grouped.shape}")
            logger.info(f"currps sample:\n{currps[:10,2]}")
            logger.info(f"currps sample grouped:\n{currps_grouped[0][3]}")
        
        # Latent features as output (y_train)
        y_train[i*numgroups:(i+1)*numgroups, :] = currps_grouped[:,:]
    return X_train, y_train

def load_test_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    args.extra_file_tag='_diffseed'
    files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)
    X_test = np.zeros((10000*len(files), 2))  # Parameters as input
    y_test = np.zeros((10000*len(files), 256))  # Latent features as output
    
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
        
        # Parameters as input (X_test)
        X_test[i*10000:(i+1)*10000, 0] = curr_xHI
        X_test[i*10000:(i+1)*10000, 1] = curr_logfX
        
        # Load latent features
        currps = np.load(file)[samples:,:256]
        logger.info(f'loaded test data from file. shape: {currps.shape}')

        currps_boot = f21stats.bootstrap(ps=currps, reps=10000, size=10)
        # Latent features as output (y_test)
        y_test[i*10000:(i+1)*10000, :] = currps_boot
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

def log_likelihood(theta, simulator, observed_features, noise_std=0.1):
    """
    Define likelihood function using the trained simulator (XGBoost model)
    """
    xHI, logfX = theta
    
    # Use the simulator to predict latent features from parameters
    predicted_features = simulator.predict(np.array([[xHI, logfX]]))[0]
    
    # Calculate likelihood based on the difference between predicted and observed features
    # Using Gaussian likelihood with fixed noise standard deviation
    diff = observed_features - predicted_features
    log_like = -0.5 * np.sum(diff**2) / (noise_std**2)
    
    return log_like

def log_posterior(theta, simulator, observed_features, noise_std=0.1):
    """
    Define posterior function
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, simulator, observed_features, noise_std)

def run_mcmc_inference(simulator, observed_features, true_params, n_walkers=32, n_steps=10000, burn_in=1000, noise_std=0.1):
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
                                  args=(simulator, observed_features, noise_std))
    
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

def evaluate_simulator_performance(simulator, X_test, y_test, output_dir):
    """
    Evaluate the performance of the simulator by comparing predicted vs true latent features
    """
    logger.info("Evaluating simulator performance...")
    
    # Make predictions
    y_pred = simulator.predict(X_test)
    
    # Calculate metrics for each latent feature dimension
    mse_per_feature = np.mean((y_test - y_pred)**2, axis=0)
    r2_per_feature = np.array([r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    
    # Overall metrics
    mse_overall = np.mean(mse_per_feature)
    r2_overall = np.mean(r2_per_feature)
    
    logger.info(f"Simulator Performance:")
    logger.info(f"Overall MSE: {mse_overall:.6f}")
    logger.info(f"Overall R2: {r2_overall:.4f}")
    logger.info(f"Mean MSE per feature: {np.mean(mse_per_feature):.6f}")
    logger.info(f"Mean R2 per feature: {np.mean(r2_per_feature):.4f}")
    
    # Plot performance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MSE per feature
    ax1 = axes[0, 0]
    ax1.plot(mse_per_feature)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE per Latent Feature')
    ax1.set_yscale('log')
    
    # R2 per feature
    ax2 = axes[0, 1]
    ax2.plot(r2_per_feature)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score per Latent Feature')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Sample predictions vs true for first few features
    ax3 = axes[1, 0]
    for i in range(min(5, y_test.shape[1])):
        ax3.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=1, label=f'Feature {i}')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
    ax3.set_xlabel('True Latent Features')
    ax3.set_ylabel('Predicted Latent Features')
    ax3.set_title('Predicted vs True (First 5 Features)')
    ax3.legend()
    
    # Distribution of residuals
    ax4 = axes[1, 1]
    residuals = y_test - y_pred
    ax4.hist(residuals.flatten(), bins=50, alpha=0.7, density=True)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/simulator_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save performance metrics
    np.savetxt(f'{output_dir}/simulator_mse_per_feature.csv', mse_per_feature, delimiter=',')
    np.savetxt(f'{output_dir}/simulator_r2_per_feature.csv', r2_per_feature, delimiter=',')

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
    parser.add_argument('--noise_std', type=float, default=0.1, help='Noise standard deviation for likelihood')
    
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
    logger.info(f"X_train (parameters): xHI range=[{X_train[:, 0].min():.3f}, {X_train[:, 0].max():.3f}], logfX range=[{X_train[:, 1].min():.3f}, {X_train[:, 1].max():.3f}]")
    logger.info(f"y_train (latent features): shape={y_train.shape}")

    logger.info(f"Loading test data from {args.testdatapath}...")
    X_test, y_test = load_test_data(override_path=args.testdatapath, samples=args.limitsamplesize, args=args)
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Initialize and train XGBoost simulator (parameters -> latent features)
    logger.info("Training XGBoost simulator (parameters -> latent features)...")
    simulator = XGBRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    simulator.fit(X_train, y_train)

    logger.info(f"Fitted simulator: {simulator}")
    logger.info(f"Booster: {simulator.get_booster()}")
    feature_importance = simulator.feature_importances_
    save_model(simulator, f'{output_dir}/xgb-simulator.json')
    np.savetxt(f"{output_dir}/simulator_feature_importance.csv", feature_importance, delimiter=',')
    logger.info(f"Simulator feature importance: {feature_importance}")
    for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
        logger.info(f"Importance type {imp_type}: {simulator.get_booster().get_score(importance_type=imp_type)}")

    # Evaluate simulator performance
    evaluate_simulator_performance(simulator, X_test, y_test, output_dir)

    # Select 5 test points for MCMC inference
    logger.info(f"Selecting {args.n_test_points} test points for MCMC inference...")
    
    # Select test points with different parameter combinations
    test_indices = np.linspace(0, len(X_test)-1, args.n_test_points, dtype=int)
    test_features = y_test[test_indices]  # Observed latent features
    test_params = X_test[test_indices]    # True parameters
    
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
            simulator, features, true_params, 
            n_walkers=args.n_walkers, 
            n_steps=args.n_steps, 
            burn_in=args.burn_in,
            noise_std=args.noise_std
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