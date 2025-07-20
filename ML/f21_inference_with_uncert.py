'''
f21 inference using power spectrum data.
This script uses modular components for better organization and maintainability.
'''

import argparse
import sys
import os

# Import our modular components
import config_manager as cm
import data_loader as dl
import regression_trainer as rt
import model_tester as mt
import metrics_calculator as mc
import results_plotter as rp
import f21_predict_base as base


def main():
    """
    Main function that orchestrates the entire f21 inference pipeline.
    """
    # Setup argument parser and parse arguments
    parser = cm.setup_argument_parser()
    args = parser.parse_args()
    
    # Set default paths if not provided
    cm.set_default_paths(args)
    
    # Setup environment and device
    device = cm.setup_environment()
    
    # Create output directory and setup logging
    output_dir = base.create_output_dir(args=args)
    logger = base.setup_logging(output_dir)
    
    logger.info("####")
    logger.info(f"### Using \"{device}\" device ###")
    logger.info("####")
    
    # Load training data
    logger.info("Loading training data...")
    X_train, y_train = dl.load_training_data(
        override_path=args.datapath, 
        samples=args.limitsamplesize, 
        args=args
    )
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Load test data
    logger.info("Loading test data...")
    X_test, y_test = dl.load_test_data(
        override_path=args.testdatapath, 
        samples=args.limitsamplesize, 
        args=args
    )
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    model = rt.train_xgboost_model(X_train, y_train)
    logger.info(f"Fitted regressor: {model}")
    logger.info(f"Booster: {model.get_booster()}")
    
    # Get and save feature importance
    importance_dict = rt.get_feature_importance(model)
    feature_importance = importance_dict['feature_importance']
    rt.save_feature_importance(feature_importance, output_dir)
    logger.info(f"Feature importance: {feature_importance}")
    
    # Log different types of importance
    for imp_type, scores in importance_dict['booster_scores'].items():
        logger.info(f"Importance type {imp_type}: {scores}")
    
    # Save the trained model
    rt.save_model(model, f'{output_dir}/xgb-f21-inf-ps.json')
    
    # Test the model
    logger.info("Making predictions...")
    y_pred = mt.test_model(model, X_test, y_test)
    
    # Calculate and display metrics
    metrics = mc.calculate_metrics(y_test, y_pred)
    mc.print_metrics(metrics, logger)
    mc.save_metrics(metrics, output_dir)
    
    # Save test results
    mt.save_test_results(y_pred, y_test, output_dir)
    
    # Create plots
    logger.info("Creating plots...")
    rp.plot_predictions(y_test, y_pred, output_dir, args, showplots=False, saveplots=True)
    rp.plot_feature_importance(feature_importance, output_dir, saveplots=True)
    rp.plot_residuals(y_test, y_pred, output_dir, saveplots=True)
    rp.plot_histograms(y_test, y_pred, output_dir, saveplots=True)
    rp.create_summary_plots(y_test, y_pred, output_dir, args, showplots=False, saveplots=True)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main() 