"""
Linear Regression Example

This example demonstrates how to use the Linear Regression models 
from the ml_concepts library on a sample dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Add the parent directory to the path to import the ml_concepts library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from ml_concepts library
from ml_concepts.datasets.loaders import DatasetLoader
from ml_concepts.exploratory.statistical_analysis import generate_summary_statistics
from ml_concepts.models.supervised.linear_models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel
)

def main():
    """Run the linear regression example."""
    print("Linear Regression Example\n")
    
    # 1. Load a dataset from scikit-learn
    print("Loading the diabetes dataset...")
    dataset = DatasetLoader.load_dataset('diabetes')
    X, y = dataset['X'], dataset['y']
    feature_names = dataset['feature_names']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {', '.join(feature_names)}")
    print("\n")
    
    # 2. Perform statistical analysis
    print("Generating summary statistics for features:")
    stats = generate_summary_statistics(X)
    print(stats)
    print("\n")
    
    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print("\n")
    
    # 4. Train and evaluate Linear Regression model
    print("Training Linear Regression model...")
    linear_model = LinearRegressionModel()
    linear_model.fit(X_train, y_train)
    
    y_pred_linear = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    
    print(f"Linear Regression - MSE: {mse_linear:.4f}, R²: {r2_linear:.4f}")
    print("Coefficients:")
    for feature, coef in zip(feature_names, linear_model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {linear_model.intercept_:.4f}")
    print("\n")
    
    # 5. Train and evaluate Ridge Regression model
    print("Training Ridge Regression model...")
    ridge_model = RidgeRegressionModel(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    print(f"Ridge Regression (alpha=1.0) - MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")
    print("Coefficients:")
    for feature, coef in zip(feature_names, ridge_model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {ridge_model.intercept_:.4f}")
    print("\n")
    
    # 6. Train and evaluate Lasso Regression model
    print("Training Lasso Regression model...")
    lasso_model = LassoRegressionModel(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    
    print(f"Lasso Regression (alpha=0.1) - MSE: {mse_lasso:.4f}, R²: {r2_lasso:.4f}")
    print("Coefficients:")
    for feature, coef in zip(feature_names, lasso_model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {lasso_model.intercept_:.4f}")
    print("\n")
    
    # 7. Visualize coefficient comparison between models
    print("Visualizing coefficient comparison...")
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    plt.bar(x - width, linear_model.coef_, width, label='Linear')
    plt.bar(x, ridge_model.coef_, width, label='Ridge (α=1.0)')
    plt.bar(x + width, lasso_model.coef_, width, label='Lasso (α=0.1)')
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Comparison of Coefficients for Different Regression Models')
    plt.xticks(x, feature_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('regression_coefficients_comparison.png')
    print("Figure saved as 'regression_coefficients_comparison.png'")
    
    # 8. Feature importance visualization for Lasso
    print("Visualizing Lasso feature importance...")
    fig = lasso_model.visualize_coefficients(feature_names=feature_names)
    fig.savefig('lasso_feature_importance.png')
    print("Figure saved as 'lasso_feature_importance.png'")
    
    # 9. Visualize predictions vs actual
    print("Visualizing predictions vs actual values...")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, y_pred_linear, alpha=0.5, label='Linear')
    plt.scatter(y_test, y_pred_ridge, alpha=0.5, label='Ridge')
    plt.scatter(y_test, y_pred_lasso, alpha=0.5, label='Lasso')
    
    # Plot the ideal line (y=x)
    min_val = min(y_test.min(), min(y_pred_linear.min(), y_pred_ridge.min(), y_pred_lasso.min()))
    max_val = max(y_test.max(), max(y_pred_linear.max(), y_pred_ridge.max(), y_pred_lasso.max()))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values for Different Regression Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('regression_predictions_comparison.png')
    print("Figure saved as 'regression_predictions_comparison.png'")
    
    print("\nExample completed successfully!")
    
    
if __name__ == "__main__":
    main()
