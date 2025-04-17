"""
Linear model interview questions.

This module contains common interview questions about linear models
including linear regression, logistic regression, and regularization techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split


class LinearModelQuestions:
    """Class containing linear model interview questions and answers."""

    @staticmethod
    def linear_regression_assumptions():
        """
        Q: What are the assumptions of linear regression and how to validate them?
        
        Returns:
            Dict[str, Any]: Question and detailed answer with examples.
        """
        question = "What are the assumptions of linear regression and how to validate them?"
        
        answer = """
        # Linear Regression Assumptions

        Linear regression is based on several assumptions that should be satisfied for the model to provide unbiased and efficient estimates. When these assumptions are violated, the regression results may be misleading or biased.

        ## The Main Assumptions

        ### 1. Linearity
        
        **Assumption:** There exists a linear relationship between the independent variables and the dependent variable.
        
        **Validation Techniques:**
        - **Scatter plots:** Plot each independent variable against the dependent variable to visually inspect if the relationship appears linear.
        - **Residual plots:** Plot residuals against predicted values. If the relationship is linear, residuals should be randomly scattered around zero with no discernible pattern.
        - **Partial regression plots:** Also known as added variable plots, these show the relationship between a specific predictor and the response after accounting for other predictors.
        
        **Remedies for Violation:**
        - Transform variables (e.g., log, square root, polynomial terms)
        - Add interaction terms
        - Consider non-linear models

        ### 2. Independence of Errors
        
        **Assumption:** The errors (residuals) should be independent of each other, meaning there's no correlation between consecutive residuals, especially in time-series data.
        
        **Validation Techniques:**
        - **Durbin-Watson test:** Tests for autocorrelation. Values close to 2 indicate no autocorrelation.
        - **Residual plots over time:** If residuals show patterns when plotted against time, independence may be violated.
        - **ACF/PACF plots:** In time series, autocorrelation and partial autocorrelation function plots can reveal dependencies.
        
        **Remedies for Violation:**
        - Generalized Least Squares (GLS)
        - Add lagged variables
        - Time series models (ARIMA, etc.)

        ### 3. Homoscedasticity
        
        **Assumption:** The variance of the errors should be constant across all levels of the independent variables.
        
        **Validation Techniques:**
        - **Residual plots:** Plot residuals vs. predicted values. A funnel shape indicates heteroscedasticity.
        - **Breusch-Pagan test:** Statistical test for heteroscedasticity.
        - **White test:** Another test for heteroscedasticity, more general than Breusch-Pagan.
        
        **Remedies for Violation:**
        - Transform the dependent variable (e.g., log transformation)
        - Weighted Least Squares (WLS)
        - Robust standard errors

        ### 4. Normality of Residuals
        
        **Assumption:** The residuals should follow a normal distribution.
        
        **Validation Techniques:**
        - **Histogram of residuals:** Should approximate a normal distribution.
        - **Q-Q plot:** Quantile-quantile plot comparing residuals to a normal distribution.
        - **Shapiro-Wilk test:** Statistical test for normality.
        - **Kolmogorov-Smirnov test:** Another test for normality.
        
        **Remedies for Violation:**
        - Transform the dependent variable
        - Bootstrap methods
        - For large samples, may not be a serious issue due to the Central Limit Theorem

        ### 5. No or Little Multicollinearity
        
        **Assumption:** The independent variables should not be highly correlated with each other.
        
        **Validation Techniques:**
        - **Correlation matrix:** Look for high correlation coefficients between independent variables.
        - **Variance Inflation Factor (VIF):** Values > 10 typically indicate problematic multicollinearity.
        - **Condition number:** High values indicate multicollinearity.
        
        **Remedies for Violation:**
        - Remove one of the correlated variables
        - Use dimensionality reduction techniques (e.g., PCA)
        - Ridge regression to handle multicollinearity

        ### 6. No Influential Outliers
        
        **Assumption:** The model should not be overly influenced by outliers.
        
        **Validation Techniques:**
        - **Leverage vs. Standardized Residuals plot:** Identifies high leverage points and outliers.
        - **Cook's distance:** Measures the influence of each observation.
        - **DFFITS and DFBETAS:** Measure changes in fitted values and coefficients when an observation is removed.
        
        **Remedies for Violation:**
        - Remove outliers (with justification)
        - Robust regression methods
        - Transform variables to reduce the impact of extreme values

        ## Practical Example of Validation

        Here's a structured approach to validating assumptions in practice:

        1. **Data Exploration:**
           - Check for missing values
           - Visualize distributions
           - Examine relationships between variables

        2. **Model Fitting:**
           - Fit the linear regression model
           - Collect residuals and predicted values

        3. **Assumption Validation:**
           - **Linearity:** Plot residuals vs. predicted values
           - **Independence:** Durbin-Watson test (for time series)
           - **Homoscedasticity:** Breusch-Pagan test and residual plots
           - **Normality:** Q-Q plot and Shapiro-Wilk test
           - **Multicollinearity:** VIF values
           - **Outliers:** Cook's distance and leverage plots

        4. **Remedial Measures:**
           - Transform variables if necessary
           - Consider different models if assumptions are severely violated
           - Apply appropriate statistical corrections

        ## Consequences of Violated Assumptions

        - **Biased coefficients:** Estimates may not reflect true relationships
        - **Incorrect standard errors:** Leading to invalid hypothesis tests and confidence intervals
        - **Poor prediction performance:** Model may not generalize well to new data
        - **Misleading conclusions:** Inferences drawn from the model may be invalid

        While assumption checking is important, it's worth noting that real-world data rarely satisfies all assumptions perfectly. The goal is to identify serious violations that could undermine your analysis and address them appropriately.
        """
        
        # Create visualization for assumption violations
        # Set up the figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Common Linear Regression Assumption Violations', fontsize=16)
        
        # Generate data for visualization
        np.random.seed(42)
        n_samples = 100
        
        # 1. Linearity - Compare linear and non-linear relationships
        x_lin = np.linspace(-3, 3, n_samples)
        y_lin = 2*x_lin + 1 + np.random.normal(0, 1, n_samples)  # Linear relationship
        y_nonlin = 2*x_lin**2 + x_lin + 1 + np.random.normal(0, 1, n_samples)  # Non-linear relationship
        
        # Fit linear model to both
        lin_model = LinearRegression()
        lin_model.fit(x_lin.reshape(-1, 1), y_lin)
        nonlin_pred = lin_model.predict(x_lin.reshape(-1, 1))
        
        # Plot for linearity assumption
        ax = axes[0, 0]
        ax.scatter(x_lin, y_lin, alpha=0.6, label='Linear Data')
        ax.scatter(x_lin, y_nonlin, alpha=0.6, label='Non-linear Data')
        ax.plot(x_lin, lin_model.predict(x_lin.reshape(-1, 1)), 'r-', label='Linear Model')
        ax.set_title('1. Linearity Assumption')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Independence - Show autocorrelated errors
        ax = axes[0, 1]
        t = np.arange(n_samples)
        # Generate autocorrelated errors
        errors = np.zeros(n_samples)
        errors[0] = np.random.normal(0, 1)
        for i in range(1, n_samples):
            errors[i] = 0.8 * errors[i-1] + np.random.normal(0, 0.5)
        
        # Plot autocorrelated errors
        ax.plot(t, errors, 'b-', label='Autocorrelated Errors')
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('2. Independence Assumption')
        ax.set_xlabel('Time/Sequence')
        ax.set_ylabel('Error Term')
        ax.text(10, 2, 'Pattern indicates\nautocorrelation', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8))
        ax.grid(alpha=0.3)
        
        # 3. Homoscedasticity - Compare homoscedastic and heteroscedastic errors
        ax = axes[1, 0]
        x_homo = np.linspace(-3, 3, n_samples)
        y_homo = 2*x_homo + 1 + np.random.normal(0, 1, n_samples)  # Constant variance
        
        # Heteroscedastic: variance increases with x
        y_hetero = 2*x_homo + 1 + np.random.normal(0, 0.2 + 0.8*abs(x_homo), n_samples)
        
        # Fit models and get residuals
        homo_model = LinearRegression().fit(x_homo.reshape(-1, 1), y_homo)
        homo_resid = y_homo - homo_model.predict(x_homo.reshape(-1, 1))
        
        hetero_model = LinearRegression().fit(x_homo.reshape(-1, 1), y_hetero)
        hetero_resid = y_hetero - hetero_model.predict(x_homo.reshape(-1, 1))
        
        # Plot residuals
        ax.scatter(homo_model.predict(x_homo.reshape(-1, 1)), homo_resid, alpha=0.6, 
                  label='Homoscedastic')
        ax.scatter(hetero_model.predict(x_homo.reshape(-1, 1)), hetero_resid, alpha=0.6, 
                  label='Heteroscedastic')
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('3. Homoscedasticity Assumption')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Normality of Residuals - QQ Plot
        ax = axes[1, 1]
        
        # Generate normal and non-normal residuals
        normal_residuals = np.random.normal(0, 1, n_samples)
        # Non-normal: skewed residuals (chi-square distribution)
        skewed_residuals = np.random.chisquare(3, n_samples) - 3
        
        # Calculate theoretical quantiles
        sorted_norm = np.sort(normal_residuals)
        sorted_skew = np.sort(skewed_residuals)
        quantiles = np.linspace(0, 1, n_samples + 2)[1:-1]
        theoretical = np.quantile(np.random.normal(0, 1, 10000), quantiles)
        
        # Plot QQ plots
        ax.scatter(theoretical, sorted_norm, alpha=0.6, label='Normal Residuals')
        ax.scatter(theoretical, sorted_skew, alpha=0.6, label='Non-normal Residuals')
        
        # Add reference line
        ax.plot([-3, 3], [-3, 3], 'r-', alpha=0.3)
        ax.set_title('4. Normality of Residuals (QQ Plot)')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 5. Multicollinearity - Correlation Matrix Heatmap
        ax = axes[2, 0]
        
        # Generate correlated features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        x3 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n_samples)  # Correlated with x1
        x4 = 0.9 * x1 - 0.1 * x2 + 0.1 * np.random.normal(0, 1, n_samples)  # Highly correlated with x1
        
        # Create correlation matrix
        X = np.column_stack((x1, x2, x3, x4))
        corr_matrix = np.corrcoef(X.T)
        
        # Plot correlation heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('5. Multicollinearity')
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(['X1', 'X2', 'X3', 'X4'])
        ax.set_yticklabels(['X1', 'X2', 'X3', 'X4'])
        
        # Add correlation values
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
        
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        # 6. Influential Outliers - Leverage vs. Residuals
        ax = axes[2, 1]
        
        # Generate data with outliers
        x_clean = np.random.normal(0, 1, n_samples-3)
        y_clean = 2*x_clean + 1 + np.random.normal(0, 1, n_samples-3)
        
        # Add outliers: high leverage, low residual
        x_out1, y_out1 = 4, 9
        
        # High leverage, high residual
        x_out2, y_out2 = 4, 15
        
        # Low leverage, high residual
        x_out3, y_out3 = 0, 10
        
        x_all = np.append(x_clean, [x_out1, x_out2, x_out3])
        y_all = np.append(y_clean, [y_out1, y_out2, y_out3])
        
        # Fit model
        all_model = LinearRegression().fit(x_all.reshape(-1, 1), y_all)
        all_pred = all_model.predict(x_all.reshape(-1, 1))
        all_resid = y_all - all_pred
        
        # Calculate leverage
        X_with_const = np.column_stack((np.ones(len(x_all)), x_all))
        hat_matrix = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
        leverage = np.diag(hat_matrix)
        
        # Standardized residuals
        std_resid = all_resid / np.std(all_resid)
        
        # Plot leverage vs. standardized residuals
        sc = ax.scatter(leverage, std_resid, alpha=0.6, c=np.arange(len(x_all)), cmap='viridis')
        
        # Highlight outliers
        ax.scatter([leverage[-3]], [std_resid[-3]], s=100, facecolors='none', edgecolors='r', linewidths=2, label='High Leverage, Low Residual')
        ax.scatter([leverage[-2]], [std_resid[-2]], s=100, facecolors='none', edgecolors='g', linewidths=2, label='High Leverage, High Residual')
        ax.scatter([leverage[-1]], [std_resid[-1]], s=100, facecolors='none', edgecolors='b', linewidths=2, label='Low Leverage, High Residual')
        
        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.3)
        
        # Add cook's distance contours
        p = 2  # Number of parameters (including intercept)
        
        # Function to calculate Cook's distance contour
        def cook_contour(d, p, n, x):
            # Formula for Cook's distance contours
            return np.sqrt(d * p * (1 - x) / x)
        
        x_leverage = np.linspace(0.01, 0.99, 100)
        cook_0_5 = cook_contour(0.5, p, n_samples, x_leverage)
        cook_1_0 = cook_contour(1.0, p, n_samples, x_leverage)
        
        ax.plot(x_leverage, cook_0_5, 'k--', alpha=0.3, label="Cook's distance = 0.5")
        ax.plot(x_leverage, -cook_0_5, 'k--', alpha=0.3)
        ax.plot(x_leverage, cook_1_0, 'k-', alpha=0.3, label="Cook's distance = 1.0")
        ax.plot(x_leverage, -cook_1_0, 'k-', alpha=0.3)
        
        ax.set_title('6. Influential Outliers')
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.set_ylim(-5, 5)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create a second figure for diagnostic plots example
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle('Diagnostic Plots for Linear Regression', fontsize=16)
        
        # Generate synthetic data with known issues
        np.random.seed(42)
        n = 100
        x = np.linspace(-3, 3, n)
        
        # Non-linear relationship with heteroscedastic and non-normal errors
        y = x + 0.5 * x**2 + np.random.normal(0, 0.5 + 0.3 * np.abs(x), n)
        
        # Add a few outliers
        outlier_idx = [10, 50, 80]
        y[outlier_idx] += np.array([4, -3, 5])
        
        # Fit linear model
        X = x.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # 1. Residuals vs Fitted values plot
        ax = axes2[0, 0]
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add a smoothed line to show patterns
        from scipy.stats import binned_statistic
        bins = 10
        bin_means, bin_edges, _ = binned_statistic(y_pred, residuals, statistic='mean', bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2)
        
        ax.set_title('Residuals vs Fitted')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        
        # Highlight outliers
        for i in outlier_idx:
            ax.annotate(str(i), (y_pred[i], residuals[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        # 2. Q-Q plot
        ax = axes2[0, 1]
        from scipy import stats
        
        # Calculate standardized residuals
        std_resid = residuals / np.std(residuals)
        
        # Generate Q-Q plot
        stats.probplot(std_resid, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q')
        
        # 3. Scale-Location plot (Sqrt of abs standardized residuals vs fitted values)
        ax = axes2[1, 0]
        sqrt_abs_resid = np.sqrt(np.abs(std_resid))
        ax.scatter(y_pred, sqrt_abs_resid, alpha=0.6)
        
        # Add a smoothed line
        bin_means, bin_edges, _ = binned_statistic(y_pred, sqrt_abs_resid, statistic='mean', bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2)
        
        ax.set_title('Scale-Location')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('$\\sqrt{|\\textrm{Standardized residuals}|}$')
        
        # Highlight outliers
        for i in outlier_idx:
            ax.annotate(str(i), (y_pred[i], sqrt_abs_resid[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Residuals vs Leverage plot
        ax = axes2[1, 1]
        
        # Calculate leverage
        X_with_intercept = np.column_stack([np.ones(n), X])
        hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
        leverage = np.diag(hat_matrix)
        
        # Calculate Cook's distance
        p = 2  # Number of parameters (including intercept)
        cook_distance = residuals**2 / (p * np.var(residuals)) * (leverage / (1 - leverage)**2)
        
        # Plot standardized residuals vs leverage
        ax.scatter(leverage, std_resid, alpha=0.6, c=cook_distance, cmap='viridis')
        fig2.colorbar(ax.collections[0], ax=ax, label="Cook's distance")
        
        # Draw contour lines for Cook's distance
        x_leverage = np.linspace(0.01, 0.99, 100)
        
        def cook_contour(d, p, n, x):
            return np.sqrt(d * p * (1 - x) / x)
        
        # Add Cook's distance contours
        cook_0_5 = cook_contour(0.5, p, n, x_leverage)
        cook_1_0 = cook_contour(1.0, p, n, x_leverage)
        
        # Only plot where the contour is within y-limits
        y_min, y_max = -4, 4
        valid_indices = np.logical_and(cook_0_5 < y_max, cook_0_5 > y_min)
        
        if np.any(valid_indices):
            ax.plot(x_leverage[valid_indices], cook_0_5[valid_indices], 'k--', alpha=0.3, label="Cook's distance = 0.5")
            ax.plot(x_leverage[valid_indices], -cook_0_5[valid_indices], 'k--', alpha=0.3)
        
        valid_indices = np.logical_and(cook_1_0 < y_max, cook_1_0 > y_min)
        if np.any(valid_indices):
            ax.plot(x_leverage[valid_indices], cook_1_0[valid_indices], 'k-', alpha=0.3, label="Cook's distance = 1.0") 
            ax.plot(x_leverage[valid_indices], -cook_1_0[valid_indices], 'k-', alpha=0.3)
        
        ax.set_title('Residuals vs Leverage')
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized residuals')
        ax.set_ylim(y_min, y_max)
        
        # Highlight outliers
        for i in outlier_idx:
            ax.annotate(str(i), (leverage[i], std_resid[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return {
            "question": question,
            "answer": answer,
            "figures": [fig, fig2],
            "key_points": [
                "Linear regression has 6 main assumptions: linearity, independence of errors, homoscedasticity, normality of residuals, no multicollinearity, and no influential outliers.",
                "Violation of assumptions can lead to biased coefficients, incorrect standard errors, and poor prediction performance.",
                "Diagnostic plots like residual plots, Q-Q plots, and leverage plots are essential for validating assumptions.",
                "Each assumption can be validated with specific statistical tests and visual methods.",
                "When assumptions are violated, remedies include transformations, alternative models, or robust methods."
            ],
            "references": [
                "Fox, J. (2015). Applied Regression Analysis and Generalized Linear Models.",
                "Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis.",
                "Faraway, J. J. (2016). Linear Models with R."
            ]
        }
    
    @staticmethod
    def logistic_regression_vs_linear():
        """
        Q: What are the differences between logistic regression and linear regression?
        
        Returns:
            Dict[str, Any]: Question and detailed answer with examples.
        """
        question = "What are the differences between logistic regression and linear regression?"
        
        answer = """
        # Linear Regression vs. Logistic Regression

        Linear regression and logistic regression are two fundamental supervised learning algorithms that serve different purposes. While they share some similarities in their approach, their mathematical foundations, applications, and interpretations differ significantly.

        ## Core Differences

        ### 1. Purpose and Output

        **Linear Regression:**
        - Predicts continuous numerical values
        - Output is unbounded (can be any real number)
        - Used for regression tasks
        - Example applications: predicting house prices, temperature forecasting, sales projections

        **Logistic Regression:**
        - Predicts probability of class membership
        - Output is bounded between 0 and 1 (representing probability)
        - Used primarily for binary classification tasks (can be extended to multi-class)
        - Example applications: spam detection, disease diagnosis, customer churn prediction

        ### 2. Mathematical Formula

        **Linear Regression:**
        The linear regression model follows the equation:
        
        $$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n + \\epsilon$$
        
        Where:
        - $y$ is the dependent variable (target)
        - $\\beta_0$ is the y-intercept (constant term)
        - $\\beta_1, \\beta_2, ..., \\beta_n$ are the coefficients for the independent variables
        - $x_1, x_2, ..., x_n$ are the independent variables (features)
        - $\\epsilon$ is the error term

        **Logistic Regression:**
        The logistic regression model uses the logistic (sigmoid) function:
        
        $$P(y=1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n)}}$$
        
        Or more compactly:
        
        $$P(y=1|x) = \\sigma(\\beta^T x)$$
        
        Where:
        - $P(y=1|x)$ is the probability that the dependent variable equals 1
        - $\\sigma$ is the sigmoid function
        - $\\beta^T x$ is the linear predictor function

        ### 3. Cost Function

        **Linear Regression:**
        Uses Mean Squared Error (MSE) as the cost function:
        
        $$J(\\beta) = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$
        
        Where:
        - $y_i$ is the actual value
        - $\\hat{y}_i$ is the predicted value
        - $N$ is the number of observations

        **Logistic Regression:**
        Uses Log Loss (Binary Cross-Entropy) as the cost function:
        
        $$J(\\beta) = -\\frac{1}{N} \\sum_{i=1}^{N} [y_i \\log(\\hat{p}_i) + (1-y_i) \\log(1-\\hat{p}_i)]$$
        
        Where:
        - $y_i$ is the actual class (0 or 1)
        - $\\hat{p}_i$ is the predicted probability
        - $N$ is the number of observations

        ### 4. Assumptions

        **Linear Regression:**
        - Linearity: The relationship between independent and dependent variables is linear
        - Independence: Observations are independent of each other
        - Homoscedasticity: Constant variance of errors
        - Normality: Residuals are normally distributed
        - No or little multicollinearity
