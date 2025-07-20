'''
Regression training module for f21 inference using power spectrum data.
Handles training of XGBoost regression models.
'''

import numpy as np
from xgboost import XGBRegressor


def train_xgboost_model(X_train, y_train, random_state=42):
    """
    Train an XGBoost regression model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        random_state (int): Random seed for reproducibility
        
    Returns:
        XGBRegressor: Trained XGBoost model
    """
    model = XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model, modelfile):
    """
    Save the trained model to a file.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        modelfile (str): Path to save the model
    """
    print(f'Saving model to: {modelfile}')
    model.save_model(modelfile)


def get_feature_importance(model):
    """
    Get feature importance from the trained model.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        
    Returns:
        dict: Dictionary containing different types of feature importance
    """
    feature_importance = model.feature_importances_
    
    importance_dict = {
        'feature_importance': feature_importance,
        'booster_scores': {}
    }
    
    # Get different types of importance from booster
    for imp_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
        importance_dict['booster_scores'][imp_type] = model.get_booster().get_score(importance_type=imp_type)
    
    return importance_dict


def save_feature_importance(feature_importance, output_dir):
    """
    Save feature importance to a CSV file.
    
    Args:
        feature_importance (np.ndarray): Feature importance array
        output_dir (str): Directory to save the file
    """
    import os
    np.savetxt(os.path.join(output_dir, "feature_importance.csv"), feature_importance, delimiter=',') 