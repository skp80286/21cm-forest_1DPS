'''
Results plotting module for f21 inference using power spectrum data.
Handles creation of plots and visualizations for model results.
'''

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_predictions(y_test, y_pred, output_dir, args=None, showplots=False, saveplots=True):
    """
    Create prediction plots comparing true vs predicted values.
    
    Args:
        y_test (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save plots
        args: Arguments object containing configuration (optional)
        showplots (bool): Whether to display plots
        saveplots (bool): Whether to save plots
    """
    # Create basic prediction plots
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
    
    if saveplots:
        plt.savefig(os.path.join(output_dir, 'predictions.pdf'), format='pdf')
        plt.savefig(os.path.join(output_dir, 'predictions.png'), format='png', dpi=300)
    
    if showplots:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(feature_importance, output_dir, saveplots=True):
    """
    Plot feature importance from the trained model.
    
    Args:
        feature_importance (np.ndarray): Feature importance array
        output_dir (str): Directory to save plots
        saveplots (bool): Whether to save plots
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.xticks(range(len(feature_importance)))
    
    if saveplots:
        plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'), format='pdf')
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), format='png', dpi=300)
    
    plt.close()


def create_summary_plots(y_test, y_pred, output_dir, args=None, showplots=False, saveplots=True):
    """
    Create comprehensive summary plots including the original plot_results functionality.
    
    Args:
        y_test (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save plots
        args: Arguments object containing configuration (optional)
        showplots (bool): Whether to display plots
        saveplots (bool): Whether to save plots
    """
    # Import plot_results module for additional plotting functionality
    try:
        import plot_results as pltr
        label = f"{args.telescope}, {args.t_int:.0f}h" if args else "Model Results"
        pltr.summarize_test_1000(y_pred, y_test, output_dir=output_dir, 
                                showplots=showplots, saveplots=saveplots, label=label)
    except ImportError:
        print("Warning: plot_results module not found. Using basic plotting only.")
        plot_predictions(y_test, y_pred, output_dir, args, showplots, saveplots)


def plot_residuals(y_test, y_pred, output_dir, saveplots=True):
    """
    Plot residuals (prediction errors) for each output variable.
    
    Args:
        y_test (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save plots
        saveplots (bool): Whether to save plots
    """
    residuals = y_test - y_pred
    output_names = ['xHI', 'logfX']
    
    plt.figure(figsize=(12, 5))
    
    for i, name in enumerate(output_names):
        plt.subplot(1, 2, i+1)
        plt.scatter(y_test[:, i], residuals[:, i], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'True {name}')
        plt.ylabel(f'Residual ({name})')
        plt.title(f'{name} Residuals')
    
    plt.tight_layout()
    
    if saveplots:
        plt.savefig(os.path.join(output_dir, 'residuals.pdf'), format='pdf')
        plt.savefig(os.path.join(output_dir, 'residuals.png'), format='png', dpi=300)
    
    plt.close()


def plot_histograms(y_test, y_pred, output_dir, saveplots=True):
    """
    Plot histograms of true vs predicted values.
    
    Args:
        y_test (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save plots
        saveplots (bool): Whether to save plots
    """
    output_names = ['xHI', 'logfX']
    
    plt.figure(figsize=(12, 5))
    
    for i, name in enumerate(output_names):
        plt.subplot(1, 2, i+1)
        plt.hist(y_test[:, i], alpha=0.7, label=f'True {name}', bins=30)
        plt.hist(y_pred[:, i], alpha=0.7, label=f'Predicted {name}', bins=30)
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.title(f'{name} Distribution')
        plt.legend()
    
    plt.tight_layout()
    
    if saveplots:
        plt.savefig(os.path.join(output_dir, 'histograms.pdf'), format='pdf')
        plt.savefig(os.path.join(output_dir, 'histograms.png'), format='png', dpi=300)
    
    plt.close() 