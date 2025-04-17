"""
Fundamental machine learning interview questions.

This module contains common interview questions and answers about
basic machine learning concepts, including:
- Supervised vs unsupervised learning
- Bias-variance tradeoff
- Cross-validation
- Overfitting and underfitting
- Feature selection
- Model evaluation metrics
"""

from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MLFundamentals:
    """Class containing fundamental machine learning interview questions and answers."""
    
    @staticmethod
    def supervised_vs_unsupervised():
        """
        Q: What's the difference between supervised and unsupervised learning?
        
        Returns:
            Dict[str, Any]: Question and detailed answer with examples.
        """
        question = "What's the difference between supervised and unsupervised learning?"
        
        answer = """
        # Supervised Learning vs Unsupervised Learning
        
        ## Supervised Learning
        
        Supervised learning involves training a model on a labeled dataset, where each training example consists of an input and the desired output (target). The model learns to map inputs to outputs and can then make predictions on new, unseen data.
        
        ### Key Characteristics of Supervised Learning:
        
        - **Labeled Data**: Training examples include both features and target labels.
        - **Direct Feedback**: The model receives immediate feedback on its predictions through loss functions.
        - **Goal-Oriented**: The learning process aims to predict specific outputs.
        - **Output Space**: The possible outputs are known in advance.
        
        ### Examples of Supervised Learning:
        
        - **Classification**: Predicting a categorical label (e.g., spam vs. non-spam email)
        - **Regression**: Predicting a continuous value (e.g., house prices)
        
        ### Common Supervised Learning Algorithms:
        
        - Linear/Logistic Regression
        - Decision Trees and Random Forests
        - Support Vector Machines (SVMs)
        - Neural Networks
        - k-Nearest Neighbors
        - Naive Bayes
        
        ## Unsupervised Learning
        
        Unsupervised learning involves training a model on data without labeled responses. The model learns patterns and structures in the data without specific guidance on what to predict.
        
        ### Key Characteristics of Unsupervised Learning:
        
        - **Unlabeled Data**: Training examples include only features, no target labels.
        - **No Feedback**: There are no explicit correct answers to guide the learning process.
        - **Exploratory**: The learning process aims to discover hidden patterns or structures.
        - **Output Space**: The possible outputs are not known in advance.
        
        ### Examples of Unsupervised Learning:
        
        - **Clustering**: Grouping similar data points (e.g., customer segmentation)
        - **Dimensionality Reduction**: Reducing the number of features while preserving information
        - **Association**: Discovering rules that describe associations between variables (e.g., market basket analysis)
        - **Anomaly Detection**: Identifying unusual data points
        
        ### Common Unsupervised Learning Algorithms:
        
        - K-means clustering
        - Hierarchical clustering
        - DBSCAN
        - Principal Component Analysis (PCA)
        - t-SNE
        - Autoencoders
        - Gaussian Mixture Models
        
        ## Semi-Supervised Learning
        
        It's worth mentioning that there's also semi-supervised learning, which combines elements of both approaches:
        
        - Uses both labeled and unlabeled data for training
        - Particularly useful when labeled data is scarce or expensive to obtain
        - Can achieve better performance than purely supervised learning with limited labeled data
        
        ## Reinforcement Learning
        
        Reinforcement learning is another paradigm where an agent learns to make decisions by taking actions in an environment to maximize some notion of reward:
        
        - Based on rewards and punishments
        - No labeled data provided
        - Learning occurs through trial and error
        - Examples include game playing AI and robotics control systems
        """
        
        # Create a visual explanation
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Supervised Learning visualization
        ax[0].set_title("Supervised Learning", fontsize=14)
        
        # Input data points
        X = np.random.randn(50, 2)
        # True labels (binary classification)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Plot data points
        ax[0].scatter(X[y==0, 0], X[y==0, 1], color='blue', alpha=0.6, label='Class 0')
        ax[0].scatter(X[y==1, 0], X[y==1, 1], color='red', alpha=0.6, label='Class 1')
        
        # Decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        Z = (xx + yy > 0).astype(int)
        
        ax[0].contour(xx, yy, Z, colors='black', levels=[0.5], alpha=0.5, linestyles='--')
        ax[0].set_xlim(x_min, x_max)
        ax[0].set_ylim(y_min, y_max)
        ax[0].set_xlabel('Feature 1')
        ax[0].set_ylabel('Feature 2')
        ax[0].legend()
        ax[0].text(x_min + 0.5, y_max - 0.5, "Goal: Learn decision boundary\nbased on labeled examples", 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Unsupervised Learning visualization (clustering)
        ax[1].set_title("Unsupervised Learning (Clustering)", fontsize=14)
        
        # Generate three clusters
        centers = [(-2, -2), (2, 0), (0, 3)]
        X_cluster = np.vstack([np.random.randn(20, 2) + center for center in centers])
        
        # Plot unlabeled data
        ax[1].scatter(X_cluster[:, 0], X_cluster[:, 1], color='gray', alpha=0.6)
        
        # Plot cluster centers
        ax[1].scatter([c[0] for c in centers], [c[1] for c in centers], marker='x', 
                     s=100, color='black', label='Cluster Centers')
        
        # Draw circles around clusters
        for center in centers:
            circle = plt.Circle(center, 1.2, color='black', fill=False, alpha=0.5, linestyle='--')
            ax[1].add_artist(circle)
            
        ax[1].set_xlabel('Feature 1')
        ax[1].set_ylabel('Feature 2')
        ax[1].legend()
        ax[1].text(-4, 3, "Goal: Discover structure\nin unlabeled data", 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return {
            "question": question,
            "answer": answer,
            "figure": fig,
            "key_points": [
                "Supervised learning uses labeled data to learn input-output mapping",
                "Unsupervised learning discovers patterns in unlabeled data",
                "Semi-supervised learning uses both labeled and unlabeled data",
                "Reinforcement learning learns through interaction with an environment"
            ],
            "examples": {
                "supervised": ["Classification (spam detection)", "Regression (house price prediction)"],
                "unsupervised": ["Clustering (customer segmentation)", "Dimensionality reduction (PCA)"]
            }
        }

    @staticmethod
    def bias_variance_tradeoff():
        """
        Q: Explain the bias-variance tradeoff.
        
        Returns:
            Dict[str, Any]: Question and detailed answer with examples.
        """
        question = "Explain the bias-variance tradeoff."
        
        answer = """
        # Bias-Variance Tradeoff
        
        The bias-variance tradeoff is a fundamental concept in machine learning that describes the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set.
        
        ## Decomposing Prediction Error
        
        For any machine learning model, the expected prediction error can be decomposed into three components:
        
        **Total Error = Bias² + Variance + Irreducible Error**
        
        Where:
        
        ### Bias
        
        Bias is the error introduced by approximating a real-world problem with a simplified model. High bias means the model makes strong assumptions about the target function, resulting in **underfitting**.
        
        - **High-bias models** tend to be less complex, such as linear regression.
        - They may miss relevant relations between features and target outputs.
        - They tend to have higher training error and higher test error.
        
        ### Variance
        
        Variance is the error introduced by the model's sensitivity to fluctuations in the training set. High variance means the model is too complex and fits the training data noise too closely, resulting in **overfitting**.
        
        - **High-variance models** tend to be more complex, such as high-degree polynomial regression or deep neural networks.
        - They fit the training data very well but fail to generalize to new data.
        - They tend to have low training error but high test error.
        
        ### Irreducible Error
        
        Irreducible error represents the noise in the data that cannot be reduced by any model. It sets a lower bound on the expected error.
        
        ## The Tradeoff
        
        The tradeoff emerges because:
        
        1. **Decreasing bias typically increases variance**
            - Making a model more complex to better represent the data often makes it more sensitive to the specific training data.
            
        2. **Decreasing variance typically increases bias**
            - Making a model simpler to be more robust often makes it less capable of capturing the true relationship in the data.
            
        ## Finding the Sweet Spot
        
        The goal in machine learning is to find the model complexity that minimizes the total error. This is typically done through:
        
        1. **Cross-validation**: To estimate the model's performance on unseen data.
        2. **Regularization**: To control the model complexity by penalizing large coefficients.
        3. **Model selection**: Comparing different model types and hyperparameters.
        
        ## Mathematical Formulation
        
        For a target function $f(x)$, data point $x$, and model $\hat{f}(x)$:
        
        - **Bias**: $\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$
        - **Variance**: $\text{Var}[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$
        
        Where $E[\cdot]$ represents the expected value over different training sets.
        
        The expected squared error at a point $x$ is:
        
        $E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$
        
        Where $\sigma^2$ is the irreducible error.
        """
        
        # Create a visualization of bias-variance tradeoff
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generate data
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        # True function
        true_func = lambda x: np.sin(x) * x/5 + 2
        # Noisy targets
        y = true_func(x) + np.random.normal(0, 0.5, size=len(x))
        
        # Sample points for visualization
        sample_indices = np.random.choice(len(x), 15, replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
        
        # Plot the true function and data in both plots
        for ax in [ax1, ax2]:
            ax.plot(x, true_func(x), 'k-', lw=2, label='True function')
            ax.scatter(x_sample, y_sample, c='blue', alpha=0.6, label='Sample data')
        
        # Plot models with different complexity
        
        # High bias (underfitting) - linear model
        from sklearn.linear_model import LinearRegression
        high_bias_model = LinearRegression().fit(x_sample.reshape(-1, 1), y_sample)
        y_pred_high_bias = high_bias_model.predict(x.reshape(-1, 1))
        ax1.plot(x, y_pred_high_bias, 'r-', lw=2, label='High bias model (linear)')
        ax1.set_title('High Bias (Underfitting)', fontsize=14)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        
        # High variance (overfitting) - high-degree polynomial
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        
        high_variance_model = make_pipeline(
            PolynomialFeatures(degree=15),
            LinearRegression()
        ).fit(x_sample.reshape(-1, 1), y_sample)
        
        y_pred_high_variance = high_variance_model.predict(x.reshape(-1, 1))
        ax2.plot(x, y_pred_high_variance, 'g-', lw=2, label='High variance model (polynomial)')
        ax2.set_title('High Variance (Overfitting)', fontsize=14)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.legend()
        
        plt.tight_layout()
        
        # Create another figure to show the error vs. model complexity
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        
        # Range of model complexities (polynomial degrees)
        complexities = np.arange(1, 16)
        train_errors = []
        test_errors = []
        
        # Hold out 20% of the data for testing
        from sklearn.model_selection import train_test_split
        X = x.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Calculate errors for different model complexities
        for degree in complexities:
            model = make_pipeline(
                PolynomialFeatures(degree=degree),
                LinearRegression()
            ).fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_error = np.mean((train_pred - y_train) ** 2)
            test_error = np.mean((test_pred - y_test) ** 2)
            
            train_errors.append(train_error)
            test_errors.append(test_error)
        
        # Plot training and test errors
        ax3.plot(complexities, train_errors, 'o-', color='blue', label='Training error (Variance)')
        ax3.plot(complexities, test_errors, 'o-', color='red', label='Test error')
        
        # Add bias curve (conceptual, not actual calculation)
        bias_curve = np.array([3.0, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04])
        ax3.plot(complexities, bias_curve, 'o-', color='green', label='Bias')
        
        # Mark the sweet spot
        best_complexity_idx = np.argmin(test_errors)
        best_complexity = complexities[best_complexity_idx]
        min_test_error = test_errors[best_complexity_idx]
        
        ax3.axvline(x=best_complexity, color='gray', linestyle='--', alpha=0.7)
        ax3.scatter([best_complexity], [min_test_error], s=100, c='purple', zorder=10, label='Optimal complexity')
        
        ax3.set_title('Bias-Variance Tradeoff', fontsize=14)
        ax3.set_xlabel('Model Complexity (Polynomial Degree)')
        ax3.set_ylabel('Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.annotate('Underfitting', xy=(1, train_errors[0]), xytext=(1, train_errors[0] + 1),
                    arrowprops=dict(arrowstyle='->'))
        ax3.annotate('Overfitting', xy=(15, test_errors[-1]), xytext=(13, test_errors[-1] + 1),
                    arrowprops=dict(arrowstyle='->'))
        ax3.annotate('Optimal\nBalance', xy=(best_complexity, min_test_error), 
                    xytext=(best_complexity - 3, min_test_error - 0.5),
                    arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        
        return {
            "question": question,
            "answer": answer,
            "figures": [fig, fig2],
            "key_points": [
                "Total error = Bias² + Variance + Irreducible Error",
                "Bias: Error from simplifying assumptions (underfitting)",
                "Variance: Error from sensitivity to training data (overfitting)",
                "As complexity increases, bias decreases but variance increases",
                "Optimal model finds the sweet spot that minimizes total error"
            ],
            "examples": {
                "high_bias_models": ["Linear regression", "Naive Bayes"],
                "high_variance_models": ["Decision trees (unpruned)", "High-degree polynomials"]
            }
        }
        
    @staticmethod
    def cross_validation():
        """
        Q: What is cross-validation and why is it important?
        
        Returns:
            Dict[str, Any]: Question and detailed answer with examples.
        """
        question = "What is cross-validation and why is it important?"
        
        answer = """
        # Cross-Validation
        
        Cross-validation is a resampling technique used to evaluate the performance of machine learning models and ensure they generalize well to independent datasets. It helps detect issues like overfitting and provides a more reliable estimate of a model's performance than a single train-test split.
        
        ## Why Cross-Validation is Important
        
        1. **More reliable model evaluation**: Using multiple train-test splits ensures the performance metric (e.g., accuracy, RMSE) is more robust.
        
        2. **Reduces bias and variance in performance estimation**: A single train-test split might lead to optimistic or pessimistic estimates depending on which data points happen to be in the test set.
        
        3. **Helps detect overfitting**: If a model performs well on training data but poorly on validation data across multiple folds, it's likely overfitting.
        
        4. **Maximizes use of available data**: Particularly valuable when the dataset is small, as it allows using all data for both training and validation.
        
        5. **Better hyperparameter tuning**: Cross-validation provides a more stable performance measure for comparing different hyperparameter settings.
        
        ## Common Cross-Validation Methods
        
        ### k-Fold Cross-Validation
        
        The most common form of cross-validation:
        
        1. Split the dataset into k equally sized folds
        2. For each fold i (from 1 to k):
           - Train the model on all folds except fold i
           - Validate the model on fold i
        3. Average the k validation scores to get the final performance estimate
        
        Typically k=5 or k=10 is used, balancing computational cost and estimate reliability.
        
        ### Stratified k-Fold Cross-Validation
        
        A variation of k-fold that preserves the percentage of samples for each class:
        
        - Ensures each fold has approximately the same proportion of each target class as the complete dataset
        - Particularly important for imbalanced datasets
        - Reduces bias in performance estimation
        
        ### Leave-One-Out Cross-Validation (LOOCV)
        
        An extreme case of k-fold where k equals the number of samples:
        
        - For each sample, train on all data except that sample and test on the left-out sample
        - Computationally expensive for large datasets
        - Has lower bias but can have higher variance than k-fold
        
        ### Leave-p-Out Cross-Validation
        
        A generalization of LOOCV:
        
        - For each combination of p samples, train on all data except those p samples and test on the left-out samples
        - Even more computationally expensive than LOOCV
        
        ### Time Series Cross-Validation
        
        For time series data, standard k-fold can leak future information into the model. Time series cross-validation maintains the temporal order:
        
        - Forward chaining or rolling forecasting origin
        - Each training set consists only of observations that occurred prior to the observations in the validation set
        
        ## Common Pitfalls in Cross-Validation
        
        1. **Data leakage**: Preprocessing the entire dataset before cross-validation can leak information from the test set into the model.
        
        2. **Selection bias**: Choosing the final model based on cross-validation results introduces bias.
        
        3. **Improper handling of time series data**: Using standard k-fold for time-dependent data without respecting chronological order.
        
        4. **Ignoring stratification for imbalanced data**: Can lead to folds with different class distributions.
        
        5. **Using too few folds**: Might increase bias in the performance estimate.
        """
        
        # Create visualization for k-fold cross-validation
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Parameters
        k = 5  # Number of folds
        n_samples = 20  # Number of samples for visualization
        
        # Create a grid for visualization
        fold_size = n_samples // k
        
        # Set up the plot
        ax.set_xlim(-1, n_samples)
        ax.set_ylim(-1, k+1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Plot title
        ax.set_title(f"{k}-Fold Cross-Validation", fontsize=16, pad=20)
        
        # Define colors
        train_color = 'lightblue'
        test_color = 'salmon'
        
        # Draw all samples
        for i in range(n_samples):
            ax.add_patch(plt.Rectangle((i, 0), 0.8, k, color='lightgray', ec='gray'))
            
        # Draw each fold iteration
        for fold in range(k):
            # Determine test indices for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size
            
            # Draw fold number
            ax.text(-1, fold + 0.5, f"Fold {fold+1}", verticalalignment='center', fontsize=12)
            
            # Highlight train and test sets
            for i in range(n_samples):
                if test_start <= i < test_end:
                    # Test sample
                    ax.add_patch(plt.Rectangle((i, fold), 0.8, 0.8, color=test_color, ec='black'))
                else:
                    # Train sample
                    ax.add_patch(plt.Rectangle((i, fold), 0.8, 0.8, color=train_color, ec='black'))
        
        # Add legend
        train_patch = plt.Rectangle((0, 0), 1, 1, color=train_color, ec='black')
        test_patch = plt.Rectangle((0, 0), 1, 1, color=test_color, ec='black')
        ax.legend([train_patch, test_patch], ['Training data', 'Validation data'], 
                 loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
        
        # Add explanation text
        explanation = (
            "1. Data is split into k equal folds\n"
            "2. Model is trained k times, each time using k-1 folds as training data\n"
            "3. Remaining fold is used for validation\n"
            "4. Performance is averaged across all k iterations"
        )
        fig.text(0.5, 0.02, explanation, ha='center', va='center', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        # Create visualization for time series cross-validation
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Parameters
        n_splits = 4
        n_samples = 20
        
        # Set up the plot
        ax2.set_xlim(-1, n_samples)
        ax2.set_ylim(-1, n_splits+1)
        ax2.set_aspect('equal')
        ax2.axis('off')
        
        # Plot title
        ax2.set_title("Time Series Cross-Validation", fontsize=16, pad=20)
        
        # Draw timeline
        ax2.arrow(-0.5, -0.5, n_samples+0.5, 0, head_width=0.2, head_length=0.5, fc='black', ec='black')
        ax2.text(n_samples/2, -1, "Time", ha='center', fontsize=12)
        
        # Draw all samples
        for i in range(n_samples):
            ax2.add_patch(plt.Rectangle((i, 0), 0.8, n_splits, color='lightgray', ec='gray'))
            # Add time tick
            ax2.text(i+0.4, -0.8, str(i+1), ha='center', fontsize=8)
            
        # Draw each fold iteration with expanding window
        for split in range(n_splits):
            # Determine test indices for this split
            min_train = 0
            max_train = 8 + split * 3
            test_start = max_train
            test_end = test_start + 2
            
            # Ensure we don't go beyond available data
            test_end = min(test_end, n_samples)
            
            # Draw split number
            ax2.text(-1, split + 0.5, f"Split {split+1}", verticalalignment='center', fontsize=12)
            
            # Highlight train and test sets
            for i in range(n_samples):
                if min_train <= i < max_train:
                    # Train sample
                    ax2.add_patch(plt.Rectangle((i, split), 0.8, 0.8, color=train_color, ec='black'))
                elif test_start <= i < test_end:
                    # Test sample
                    ax2.add_patch(plt.Rectangle((i, split), 0.8, 0.8, color=test_color, ec='black'))
        
        # Add legend
        train_patch = plt.Rectangle((0, 0), 1, 1, color=train_color, ec='black')
        test_patch = plt.Rectangle((0, 0), 1, 1, color=test_color, ec='black')
        ax2.legend([train_patch, test_patch], ['Training data', 'Validation data'], 
                  loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
        
        # Add explanation text
        explanation2 = (
            "Time Series Cross-Validation:\n"
            "1. Initial training window\n"
            "2. Predict next time period\n"
            "3. Expand training window and repeat\n"
            "4. Maintains temporal ordering of data"
        )
        fig2.text(0.5, 0.02, explanation2, ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return {
            "question": question,
            "answer": answer,
            "figures": [fig, fig2],
            "key_points": [
                "Cross-validation uses multiple train-test splits for robust model evaluation",
                "Reduces bias and variance in performance estimation",
                "Helps detect overfitting and optimize hyperparameters",
                "Common methods: k-fold, stratified k-fold, LOOCV, time series CV",
                "Preprocessing should be done within each fold to avoid data leakage"
            ],
            "examples": {
                "k_values": ["k=5 (common balance)", "k=10 (more thorough, higher computational cost)"],
                "special_cases": ["Stratified k-fold for imbalanced data", "Time series CV for sequential data"]
            }
        }

    @staticmethod
    def overfitting_underfitting():
        """
        Q: What are overfitting and underfitting, and how can you prevent them?
        
        Returns:
            Dict[str, Any]: Question and detailed answer with examples.
        """
        question = "What are overfitting and underfitting, and how can you prevent them?"
        
        answer = """
        # Overfitting and Underfitting
        
        Overfitting and underfitting represent two fundamental challenges in machine learning that affect a model's ability to generalize to new, unseen data.
        
        ## Underfitting
        
        Underfitting occurs when a model is too simple to capture the underlying pattern in the data. It performs poorly on both the training and test datasets.
        
        ### Characteristics of Underfitting:
        
        - High bias, low variance
        - Poor performance on training data
        - Poor performance on test data
        - Model fails to capture important patterns in the data
        
        ### Causes of Underfitting:
        
        - Model is too simple for the complexity of the data
        - Important features are missing
        - Training for too few epochs
        - Regularization is too strong
        
        ## Overfitting
        
        Overfitting occurs when a model learns the training data too well, including its noise and outliers, rather than the underlying pattern. It performs extremely well on the training data but poorly on unseen data.
        
        ### Characteristics of Overfitting:
        
        - Low bias, high variance
        - Excellent performance on training data
        - Poor performance on test data
        - Model captures noise and random fluctuations in the data
        
        ### Causes of Overfitting:
        
        - Model is too complex for the available data
        - Training for too many epochs
        - Noisy training data
        - Too many features relative to the number of observations
        
        ## Preventing Underfitting
        
        1. **
