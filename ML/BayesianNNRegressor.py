import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
import os

class BayesianNNRegressor:
    """
    Bayesian Neural Network Regressor using PyTorch.
    Maps 256 features to 2 parameter values with uncertainty quantification.
    """
    
    def __init__(self, hidden_dims=[128, 64, 32], dropout_rate=0.1, 
                 learning_rate=0.001, batch_size=32, epochs=100, 
                 random_state=42, device=None):
        """
        Initialize the Bayesian Neural Network.
        
        Args:
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            random_state (int): Random seed for reproducibility
            device (str): Device to use ('cuda', 'mps', or 'cpu')
        """
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        # Set device
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device
            
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted = False
        
    def _build_network(self, input_dim, output_dim):
        """Build the neural network architecture."""
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with dropout
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _prepare_data(self, X, y=None, fit_scalers=False):
        """Prepare data for training/prediction."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor
    
    def fit(self, X, y, validation_split=0.2):
        """
        Train the Bayesian Neural Network.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, 256)
            y (np.ndarray): Training targets of shape (n_samples, 2)
            validation_split (float): Fraction of data to use for validation
        """
        # Split data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state
        )
        
        # Prepare data
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_dim = X.shape[1]  # 256
        output_dim = y.shape[1]  # 2
        self.model = self._build_network(input_dim, output_dim).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Train Loss: {train_losses[-1]:.6f}, "
                      f"Val Loss: {val_losses[-1]:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X, n_samples=100):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, 256)
            n_samples (int): Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            np.ndarray: Predictions of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = self._prepare_data(X)
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout for uncertainty estimation
                self.model.train()
                pred = self.model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        # Return mean predictions
        return np.mean(predictions, axis=0)
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, 256)
            n_samples (int): Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            tuple: (mean_predictions, std_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = self._prepare_data(X)
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout for uncertainty estimation
                self.model.train()
                pred = self.model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def plot_posterior_distribution(self, X_test, y_test, output_dir=None, save_plot=True):
        """
        Generate and plot posterior distribution for test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test values
            output_dir (str): Directory to save the plot
            save_plot (bool): Whether to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting posterior distribution")
        
        # Get predictions with uncertainty
        y_pred_mean, y_pred_std = self.predict_with_uncertainty(X_test, n_samples=100)
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot xHI parameter
        axes[0].scatter(y_test[:, 0], y_pred_mean[:, 0], 
                       alpha=0.6, s=50, c='blue', label='Predictions')
        
        # Add error bars
        axes[0].errorbar(y_test[:, 0], y_pred_mean[:, 0], 
                        yerr=y_pred_std[:, 0], fmt='none', 
                        alpha=0.3, capsize=3, color='blue')
        
        # Add perfect prediction line
        min_val = min(y_test[:, 0].min(), y_pred_mean[:, 0].min())
        max_val = max(y_test[:, 0].max(), y_pred_mean[:, 0].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                     alpha=0.8, label='Perfect Prediction')
        
        axes[0].set_xlabel('True xHI')
        axes[0].set_ylabel('Predicted xHI')
        axes[0].set_title('xHI Parameter: True vs Predicted with Uncertainty')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot logfX parameter
        axes[1].scatter(y_test[:, 1], y_pred_mean[:, 1], 
                       alpha=0.6, s=50, c='green', label='Predictions')
        
        # Add error bars
        axes[1].errorbar(y_test[:, 1], y_pred_mean[:, 1], 
                        yerr=y_pred_std[:, 1], fmt='none', 
                        alpha=0.3, capsize=3, color='green')
        
        # Add perfect prediction line
        min_val = min(y_test[:, 1].min(), y_pred_mean[:, 1].min())
        max_val = max(y_test[:, 1].max(), y_pred_mean[:, 1].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', 
                     alpha=0.8, label='Perfect Prediction')
        
        axes[1].set_xlabel('True logfX')
        axes[1].set_ylabel('Predicted logfX')
        axes[1].set_title('logfX Parameter: True vs Predicted with Uncertainty')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            plot_path = os.path.join(output_dir, 'posterior_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Posterior distribution plot saved to: {plot_path}")
        
        plt.show()
        
        return fig, axes
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }, filepath)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Rebuild the model
        input_dim = 256  # Fixed input dimension
        output_dim = 2   # Fixed output dimension
        self.model = self._build_network(input_dim, output_dim).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore other parameters
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        self.random_state = checkpoint['random_state']
        
        self.is_fitted = True
        print(f"Model loaded from: {filepath}")

