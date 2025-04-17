"""
Linear models for supervised learning.

This module implements various linear models for classification and regression tasks,
including mathematical foundations and detailed explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    LogisticRegression, SGDRegressor, SGDClassifier
)
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


class LinearRegressionModel:
    """
    Linear Regression model implementation with detailed mathematical explanation.
    
    Linear regression finds the relationship between a dependent variable y and one
    or more independent variables X by fitting a linear equation:
        y = X * w + b
    
    where:
        - y is the target variable
        - X is the feature matrix
        - w is the weight vector (coefficients)
        - b is the bias term (intercept)
    
    The model is trained by minimizing the cost function:
        J(w, b) = (1/2m) * sum((y_pred - y)^2)
    
    This is also known as the Mean Squared Error (MSE).
    """
    
    def __init__(self, fit_intercept: bool = True, normalize: bool = False,
               copy_X: bool = True, n_jobs: Optional[int] = None):
        """
        Initialize a LinearRegressionModel instance.
        
        Args:
            fit_intercept (bool, optional): Whether to fit the intercept term. Defaults to True.
            normalize (bool, optional): Whether to normalize the features. Defaults to False.
            copy_X (bool, optional): Whether to copy X. Defaults to True.
            n_jobs (Optional[int], optional): Number of jobs for parallel computation. Defaults to None.
        """
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            normalize=normalize if hasattr(LinearRegression, 'normalize') else False,
            copy_X=copy_X,
            n_jobs=n_jobs
        )
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series]) -> 'LinearRegressionModel':
        """
        Fit the linear regression model.
        
        The model computes the coefficients w and intercept b by solving:
            min_w ||Xw - y||_2^2
        
        Using the normal equation:
            w = (X^T X)^(-1) X^T y
            
        Or using gradient descent:
            w = w - alpha * (1/m) * X^T * (X*w - y)
            b = b - alpha * (1/m) * sum(X*w - y)
            
        where alpha is the learning rate and m is the number of samples.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            y (Union[np.ndarray, pd.Series]): Target values.
            
        Returns:
            LinearRegressionModel: Fitted model.
        """
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the linear model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Samples to predict.
            
        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.predict(X)
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.
        
        R^2 = 1 - SS_res / SS_tot
        
        where:
            - SS_res is the sum of squares of residuals: sum((y_pred - y)^2)
            - SS_tot is the total sum of squares: sum((y - y_mean)^2)
        
        R^2 ranges from 0 to 1, with 1 indicating a perfect fit.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Test samples.
            y (Union[np.ndarray, pd.Series]): True values.
            
        Returns:
            float: R^2 score.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters.
        """
        return {
            'coef_': self.coef_,
            'intercept_': self.intercept_
        }
    
    def visualize(self, X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series],
                 feature_idx: int = 0,
                 title: str = 'Linear Regression',
                 xlabel: str = 'X',
                 ylabel: str = 'y') -> plt.Figure:
        """
        Visualize the linear regression model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            y (Union[np.ndarray, pd.Series]): Target values.
            feature_idx (int, optional): Index of the feature to plot (for multivariate regression). 
                                        Defaults to 0.
            title (str, optional): Plot title. Defaults to 'Linear Regression'.
            xlabel (str, optional): X-axis label. Defaults to 'X'.
            ylabel (str, optional): Y-axis label. Defaults to 'y'.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y
            
        # For multivariate regression, select one feature
        if X_np.shape[1] > 1:
            X_plot = X_np[:, feature_idx].reshape(-1, 1)
            ax.set_title(f"{title} (Feature {feature_idx})")
        else:
            X_plot = X_np
            ax.set_title(title)
            
        # Sort X for clean line plot
        sort_idx = np.argsort(X_plot.flatten())
        X_sorted = X_plot[sort_idx]
        
        # Predict values
        if X_np.shape[1] > 1:
            # For multivariate, create a copy of the data with only the selected feature
            X_pred = np.zeros_like(X_np)
            X_pred[:, feature_idx] = X_sorted.flatten()
            y_pred = self.model.predict(X_pred)
        else:
            y_pred = self.model.predict(X_sorted)
            
        # Plot original data points
        ax.scatter(X_plot, y_np, color='blue', alpha=0.6, label='Data points')
        
        # Plot regression line
        ax.plot(X_sorted, y_pred[sort_idx] if X_np.shape[1] > 1 else y_pred, 
               color='red', linewidth=2, label='Linear regression')
        
        # Add formula
        if X_np.shape[1] == 1:
            coef = self.coef_[0] if isinstance(self.coef_, np.ndarray) else self.coef_
            intercept = self.intercept_
            formula = f"y = {coef:.4f}x + {intercept:.4f}"
            ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        # Add R^2 score
        r2 = self.score(X, y)
        ax.text(0.05, 0.85, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class RidgeRegressionModel:
    """
    Ridge Regression model implementation with detailed mathematical explanation.
    
    Ridge regression adds L2 regularization to linear regression:
        y = X * w + b
        
    The cost function with L2 regularization is:
        J(w, b) = (1/2m) * sum((y_pred - y)^2) + (alpha/2) * sum(w_j^2)
        
    where alpha is the regularization parameter.
    
    This penalizes large coefficient values, helping prevent overfitting.
    The Ridge estimator solves:
        min_w ||Xw - y||_2^2 + alpha * ||w||_2^2
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, 
               normalize: bool = False, copy_X: bool = True, 
               max_iter: Optional[int] = None, tol: float = 1e-3, 
               solver: str = 'auto', random_state: Optional[int] = None):
        """
        Initialize a RidgeRegressionModel instance.
        
        Args:
            alpha (float, optional): Regularization strength. Defaults to 1.0.
            fit_intercept (bool, optional): Whether to fit the intercept term. Defaults to True.
            normalize (bool, optional): Whether to normalize the features. Defaults to False.
            copy_X (bool, optional): Whether to copy X. Defaults to True.
            max_iter (Optional[int], optional): Maximum number of iterations. Defaults to None.
            tol (float, optional): Tolerance for stopping criteria. Defaults to 1e-3.
            solver (str, optional): Solver to use. Defaults to 'auto'.
            random_state (Optional[int], optional): Random seed. Defaults to None.
        """
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize=normalize if hasattr(Ridge, 'normalize') else False,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state
        )
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series]) -> 'RidgeRegressionModel':
        """
        Fit the ridge regression model.
        
        The closed-form solution for ridge regression is:
            w = (X^T X + alpha * I)^(-1) X^T y
            
        where I is the identity matrix.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            y (Union[np.ndarray, pd.Series]): Target values.
            
        Returns:
            RidgeRegressionModel: Fitted model.
        """
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the ridge regression model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Samples to predict.
            
        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.predict(X)
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Test samples.
            y (Union[np.ndarray, pd.Series]): True values.
            
        Returns:
            float: R^2 score.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters.
        """
        return {
            'alpha': self.alpha,
            'coef_': self.coef_,
            'intercept_': self.intercept_
        }
    
    def visualize(self, X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series],
                 feature_idx: int = 0,
                 title: str = 'Ridge Regression',
                 xlabel: str = 'X',
                 ylabel: str = 'y',
                 compare_with_linear: bool = True) -> plt.Figure:
        """
        Visualize the ridge regression model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            y (Union[np.ndarray, pd.Series]): Target values.
            feature_idx (int, optional): Index of the feature to plot (for multivariate regression). 
                                        Defaults to 0.
            title (str, optional): Plot title. Defaults to 'Ridge Regression'.
            xlabel (str, optional): X-axis label. Defaults to 'X'.
            ylabel (str, optional): Y-axis label. Defaults to 'y'.
            compare_with_linear (bool, optional): Whether to compare with linear regression. 
                                                 Defaults to True.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y
            
        # For multivariate regression, select one feature
        if X_np.shape[1] > 1:
            X_plot = X_np[:, feature_idx].reshape(-1, 1)
            ax.set_title(f"{title} (Feature {feature_idx}, Alpha={self.alpha})")
        else:
            X_plot = X_np
            ax.set_title(f"{title} (Alpha={self.alpha})")
            
        # Sort X for clean line plot
        sort_idx = np.argsort(X_plot.flatten())
        X_sorted = X_plot[sort_idx]
        
        # Predict values
        if X_np.shape[1] > 1:
            # For multivariate, create a copy of the data with only the selected feature changing
            X_pred = np.zeros_like(X_np)
            X_pred[:, feature_idx] = X_sorted.flatten()
            y_pred_ridge = self.model.predict(X_pred)
            
            # Compare with linear regression if requested
            if compare_with_linear:
                lr = LinearRegression()
                lr.fit(X, y)
                y_pred_linear = lr.predict(X_pred)
        else:
            y_pred_ridge = self.model.predict(X_sorted)
            
            # Compare with linear regression if requested
            if compare_with_linear:
                lr = LinearRegression()
                lr.fit(X, y)
                y_pred_linear = lr.predict(X_sorted)
            
        # Plot original data points
        ax.scatter(X_plot, y_np, color='blue', alpha=0.6, label='Data points')
        
        # Plot ridge regression line
        ax.plot(X_sorted, y_pred_ridge[sort_idx] if X_np.shape[1] > 1 else y_pred_ridge, 
               color='red', linewidth=2, label='Ridge regression')
        
        # Plot linear regression line if requested
        if compare_with_linear:
            ax.plot(X_sorted, y_pred_linear[sort_idx] if X_np.shape[1] > 1 else y_pred_linear, 
                   color='green', linewidth=2, linestyle='--', label='Linear regression')
            
        # Add formula for single-feature case
        if X_np.shape[1] == 1:
            coef = self.coef_[0] if isinstance(self.coef_, np.ndarray) else self.coef_
            intercept = self.intercept_
            formula = f"y = {coef:.4f}x + {intercept:.4f}, α={self.alpha}"
            ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        # Add R^2 score
        r2 = self.score(X, y)
        ax.text(0.05, 0.85, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add L2 norm of coefficients
        l2_norm = np.sum(self.coef_**2)
        ax.text(0.05, 0.75, f"||w||₂² = {l2_norm:.4f}", transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class LassoRegressionModel:
    """
    Lasso Regression model implementation with detailed mathematical explanation.
    
    Lasso regression adds L1 regularization to linear regression:
        y = X * w + b
        
    The cost function with L1 regularization is:
        J(w, b) = (1/2m) * sum((y_pred - y)^2) + alpha * sum(|w_j|)
        
    where alpha is the regularization parameter.
    
    This penalizes the absolute value of coefficients, promoting sparsity
    by driving some coefficients to exactly zero, effectively performing
    feature selection. The Lasso estimator solves:
        min_w (1/2) * ||Xw - y||_2^2 + alpha * ||w||_1
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
               normalize: bool = False, precompute: bool = False,
               copy_X: bool = True, max_iter: int = 1000,
               tol: float = 1e-4, warm_start: bool = False,
               positive: bool = False, random_state: Optional[int] = None,
               selection: str = 'cyclic'):
        """
        Initialize a LassoRegressionModel instance.
        
        Args:
            alpha (float, optional): Regularization strength. Defaults to 1.0.
            fit_intercept (bool, optional): Whether to fit the intercept term. Defaults to True.
            normalize (bool, optional): Whether to normalize the features. Defaults to False.
            precompute (bool, optional): Whether to precompute the Gram matrix. Defaults to False.
            copy_X (bool, optional): Whether to copy X. Defaults to True.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Tolerance for stopping criteria. Defaults to 1e-4.
            warm_start (bool, optional): Whether to reuse the solution of the previous call. 
                                        Defaults to False.
            positive (bool, optional): Whether to force positive coefficients. Defaults to False.
            random_state (Optional[int], optional): Random seed. Defaults to None.
            selection (str, optional): Feature selection method. Defaults to 'cyclic'.
        """
        self.model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize=normalize if hasattr(Lasso, 'normalize') else False,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection
        )
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series]) -> 'LassoRegressionModel':
        """
        Fit the lasso regression model.
        
        Unlike ridge regression, there is no closed-form solution for lasso.
        The optimization is performed using coordinate descent.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            y (Union[np.ndarray, pd.Series]): Target values.
            
        Returns:
            LassoRegressionModel: Fitted model.
        """
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the lasso regression model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Samples to predict.
            
        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.predict(X)
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Test samples.
            y (Union[np.ndarray, pd.Series]): True values.
            
        Returns:
            float: R^2 score.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters.
        """
        return {
            'alpha': self.alpha,
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'n_iter_': self.model.n_iter_ if hasattr(self.model, 'n_iter_') else None
        }
    
    def visualize_coefficients(self, feature_names: Optional[List[str]] = None,
                              title: str = 'Lasso Coefficients') -> plt.Figure:
        """
        Visualize the lasso coefficients to show feature selection.
        
        Args:
            feature_names (Optional[List[str]], optional): Names of features. Defaults to None.
            title (str, optional): Plot title. Defaults to 'Lasso Coefficients'.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.coef_))]
        
        # Get non-zero coefficients
        non_zero_mask = self.coef_ != 0
        non_zero_coefs = self.coef_[non_zero_mask]
        selected_features = np.array(feature_names)[non_zero_mask]
        
        # Count non-zero and zero coefficients
        n_selected = sum(non_zero_mask)
        n_discarded = len(self.coef_) - n_selected
        
        # Sort coefficients by absolute value for better visualization
        sorted_idxs = np.argsort(np.abs(non_zero_coefs))
        non_zero_coefs = non_zero_coefs[sorted_idxs]
        selected_features = selected_features[sorted_idxs]
        
        # Plot coefficients as horizontal bars
        bars = ax.barh(np.arange(len(non_zero_coefs)), non_zero_coefs)
        
        # Color bars based on sign
        for i, bar in enumerate(bars):
            if non_zero_coefs[i] > 0:
                bar.set_color('blue')
            else:
                bar.set_color('red')
                
        # Add feature names to y-axis
        ax.set_yticks(np.arange(len(non_zero_coefs)))
        ax.set_yticklabels(selected_features)
        
        # Set title and labels
        ax.set_title(f"{title} (Alpha={self.alpha}, {n_selected}/{len(self.coef_)} features selected)")
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Features')
        
        # Add zero line
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add a summary text
        summary_text = (f"Non-zero coefficients: {n_selected}\n"
                       f"Zero coefficients: {n_discarded}\n"
                       f"L1 norm: {np.sum(np.abs(self.coef_)):.4f}")
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


class LogisticRegressionModel:
    """
    Logistic Regression model implementation with detailed mathematical explanation.
    
    Logistic regression models the probability that a sample belongs to a particular class
    using the logistic (sigmoid) function:
        P(y=1|x) = sigmoid(z) = 1 / (1 + e^(-z))
    
    where z = X * w + b, also known as the 'logit' or log-odds.
    
    The cost function for logistic regression is the negative log-likelihood:
        J(w, b) = -(1/m) * sum(y*log(p) + (1-y)*log(1-p))
        
    where p is the predicted probability.
    """
    
    def __init__(self, penalty: str = 'l2', dual: bool = False,
               tol: float = 1e-4, C: float = 1.0,
               fit_intercept: bool = True, intercept_scaling: float = 1,
               class_weight: Optional[Union[Dict, str]] = None, 
               random_state: Optional[int] = None,
               solver: str = 'lbfgs', max_iter: int = 100,
               multi_class: str = 'auto', verbose: int = 0,
               warm_start: bool = False, n_jobs: Optional[int] = None,
               l1_ratio: Optional[float] = None):
        """
        Initialize a LogisticRegressionModel instance.
        
        Args:
            penalty (str, optional): Norm used in the penalization. Defaults to 'l2'.
            dual (bool, optional): Dual or primal formulation. Defaults to False.
            tol (float, optional): Tolerance for stopping criteria. Defaults to 1e-4.
            C (float, optional): Inverse of regularization strength. Defaults to 1.0.
            fit_intercept (bool, optional): Whether to fit intercept. Defaults to True.
            intercept_scaling (float, optional): Scaling factor. Defaults to 1.
            class_weight (Optional[Union[Dict, str]], optional): Weights for classes. Defaults to None.
            random_state (Optional[int],
