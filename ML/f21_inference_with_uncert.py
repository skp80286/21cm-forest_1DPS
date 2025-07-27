'''
f21 inference.
'''

import argparse
import sys
import os
import numpy as np 

# Import our modular components
import config_manager as cm
import latent_data_loader_with_uncert as dl
import regression_trainer as rt
import model_tester_with_uncert as mt
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
    X_train, y_train = dl.load_features_data(
        override_path=args.datapath, 
        samples=args.limitsamplesize, 
        samples_start_index=0,
        group_size=args.training_sample_group_size,
        args=args,
    )
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Load test data
    logger.info("Loading test data...")
    X_test, y_test = dl.load_features_data(
        override_path=args.testdatapath, 
        samples=800, 
        samples_start_index=args.limitsamplesize,
        group_size=args.testing_sample_group_size,
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
    logger.info(f"Sample predictions: {y_pred[0]} , true: {y_test[0]}")
    logger.info(f"Sample predictions: {y_pred[80]} , true: {y_test[80]}")
    logger.info(f"Sample predictions: {y_pred[160]} , true: {y_test[160]}")
    logger.info(f"Sample predictions: {y_pred[240]} , true: {y_test[240]}")
    logger.info(f"Sample predictions: {y_pred[320]} , true: {y_test[320]}")
    
    # Calculate and display metrics
    metrics = mc.calculate_metrics(y_test[:,:2], y_pred[:,:2])
    mc.print_metrics(metrics, logger)
    mc.save_metrics(metrics, output_dir)
    
    # Save test results
    mt.save_test_results(y_pred, y_test, output_dir)
    y_gen = mt.generate_posterior_samples(y_pred, y_test, 100, output_dir, label = '')
    np.save(f'{output_dir}/generated_posterior_samples.npy', y_gen)
    
    # Create plots
    logger.info("Creating plots...")
    rp.plot_predictions(y_gen[:,:2], y_gen[:,2:], output_dir, args, showplots=False, saveplots=True)
    rp.plot_feature_importance(feature_importance, output_dir, saveplots=True)
    rp.plot_residuals(y_test[:,:2], y_pred[:,:2], output_dir, saveplots=True)
    rp.plot_histograms(y_test[:,:2], y_pred[:,:2], output_dir, saveplots=True)
    rp.create_summary_plots(y_test[:,:2], y_pred[:,:2], output_dir, args, showplots=False, saveplots=True)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main() 