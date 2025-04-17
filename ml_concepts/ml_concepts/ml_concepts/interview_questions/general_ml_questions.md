# Extensive Machine Learning Interview Questions

This document contains comprehensive interview questions and detailed answers on general machine learning topics, covering core concepts, algorithms, evaluation methods, and practical applications.

## Table of Contents

1. [Machine Learning Fundamentals](#machine-learning-fundamentals)
2. [Model Evaluation and Selection](#model-evaluation-and-selection)
3. [Feature Engineering and Selection](#feature-engineering-and-selection)
4. [Supervised Learning Algorithms](#supervised-learning-algorithms)
5. [Ensemble Methods](#ensemble-methods)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Clustering Algorithms](#clustering-algorithms)
8. [Imbalanced Data Handling](#imbalanced-data-handling)
9. [Time Series Analysis](#time-series-analysis)
10. [Practical ML Applications](#practical-ml-applications)

---

## Machine Learning Fundamentals

### Q: What is the difference between parametric and non-parametric models?

**Answer:**

Parametric and non-parametric models represent two fundamentally different approaches to machine learning, each with distinct characteristics, advantages, and limitations.

**Parametric Models:**

Parametric models make strong assumptions about the data's underlying structure by using a fixed number of parameters, regardless of the amount of training data. These models can be described by a known functional form with a finite number of parameters.

**Characteristics:**
- Fixed number of parameters, independent of the training dataset size
- Make strong assumptions about the data's distribution or functional form
- Parameters are learned from the training data
- Computationally efficient and require less data
- Limited model complexity based on the chosen functional form

**Examples:**
- Linear Regression: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$
- Logistic Regression: $P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$
- Linear/Quadratic Discriminant Analysis
- Naive Bayes
- Simple Neural Networks with fixed architecture

**Non-parametric Models:**

Non-parametric models make fewer assumptions about the data's underlying structure. The number of parameters can grow with the size of the training dataset, allowing the model complexity to scale with the data.

**Characteristics:**
- Number of parameters can grow with the training dataset size
- Fewer assumptions about the data's distribution
- Can fit a wider range of data patterns and distributions
- Generally require more training data
- More computationally intensive
- Risk of overfitting if not properly constrained

**Examples:**
- k-Nearest Neighbors
- Decision Trees
- Random Forests
- Support Vector Machines with non-linear kernels
- Kernel Density Estimation
- Gaussian Processes
- Spline regression models

**Comparison:**

| Aspect | Parametric Models | Non-parametric Models |
|--------|------------------|---------------------|
| **Assumptions** | Strong assumptions about data structure | Fewer assumptions, more flexible |
| **Complexity** | Fixed, independent of data size | Can grow with data size |
| **Data Efficiency** | Work with smaller datasets | Often require more data |
| **Inference Speed** | Generally faster inference | Can be slower, especially with large training sets |
| **Overfitting Risk** | Lower (due to constrained complexity) | Higher (requires careful regularization) |
| **Interpretability** | Often more interpretable (e.g., linear models) | Can be less interpretable |
| **Adaptability** | Less adaptable to complex patterns | Better at capturing complex, unknown patterns |

**Trade-offs to Consider:**

1. **Available Data Volume:**
   - With limited data, parametric models may be preferable as they make stronger assumptions
   - With abundant data, non-parametric models can better capture complex patterns

2. **Domain Knowledge:**
   - If you have strong prior knowledge about the functional form, parametric models leverage this
   - With limited domain knowledge, non-parametric models make fewer assumptions

3. **Computational Resources:**
   - Parametric models are typically more efficient
   - Non-parametric models may require more computation, especially during inference

4. **Complexity of Underlying Data Patterns:**
   - Simple, well-understood relationships: parametric models
   - Complex, unknown relationships: non-parametric models

**Hybrid Approaches:**
Some modern approaches blend parametric and non-parametric elements:
- Deep neural networks: Parametric in structure but can approximate non-parametric behavior with sufficient size
- Gaussian Mixture Models: Parametric components combined in a way that can approximate complex distributions

**Code Example - Comparing Parametric vs Non-parametric Regression:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

# Generate nonlinear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Parametric model 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Parametric model 2: Polynomial Regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Non-parametric model: KNN Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict on a fine grid for visualization
X_grid = np.linspace(0, 5, 500).reshape(-1, 1)
y_linear = linear_model.predict(X_grid)
y_poly = poly_model.predict(poly.transform(X_grid))
y_knn = knn_model.predict(X_grid)

# Plotting
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='navy', s=30, label='Data points')
plt.plot(X_grid, y_linear, color='red', label='Linear (Parametric)')
plt.plot(X_grid, y_poly, color='green', label='Polynomial (Parametric)')
plt.plot(X_grid, y_knn, color='blue', label='5-NN (Non-parametric)')
plt.legend()
plt.title('Parametric vs Non-parametric Models')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
```

### Q: Explain the bias-variance tradeoff and how it relates to model complexity.

**Answer:**

The bias-variance tradeoff represents one of the most fundamental concepts in machine learning, describing the relationship between model complexity, prediction error, and generalization performance.

**Error Decomposition:**

For a given learning algorithm, the expected prediction error can be decomposed into three components:

**Total Error = Bias² + Variance + Irreducible Error**

Where:

1. **Bias**: Error from incorrect assumptions in the learning algorithm. High bias causes underfitting.
   - Represents how far off the model's predictions are from the true values on average
   - Models with high bias oversimplify the problem

2. **Variance**: Error from sensitivity to small fluctuations in the training set. High variance causes overfitting.
   - Represents how much the model's predictions fluctuate for different training sets
   - Models with high variance capture noise in the training data

3. **Irreducible Error**: Error from the inherent noise in the problem, which can't be eliminated by any model.

**Mathematical Formulation:**

For a target function $f(x)$, data point $x$, and model $\hat{f}(x)$ trained on dataset $D$:

- **Bias**: $\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$
- **Variance**: $\text{Var}[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$
- **Expected squared error**: $E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$

Where:
- $E[\cdot]$ is the expected value over different training sets
- $\sigma^2$ is the irreducible error

**Relationship with Model Complexity:**

As model complexity increases:
1. **Bias** typically decreases (model can fit training data more closely)
2. **Variance** typically increases (model becomes more sensitive to variations in training data)

This creates a tradeoff: at some optimal complexity level, the combined error from bias and variance reaches a minimum.

**Visual Representation:**

```
    ^
    │
    │
E   │            Total Error
r   │               /\
r   │              /  \
o   │             /    \
r   │    Variance/      \
    │          /         \
    │         /           \
    │        /             \
    │       /               \
    │      /                 \
    │     /                   \
    │    /        Bias         \
    │   /                       \
    │  /                         \
    │ /                           \
    ├─────────────────────────────────►
        Model Complexity
```

**Examples of Bias-Variance in Common Models:**

1. **High Bias, Low Variance Models**:
   - Linear Regression
   - Linear Discriminant Analysis
   - Simple decision trees (shallow)
   
2. **Low Bias, High Variance Models**:
   - Deep decision trees (no pruning)
   - k-Nearest Neighbors with small k
   - High-degree polynomial regression
   
3. **Moderate Bias-Variance Models**:
   - Random Forests
   - Gradient Boosting with tuned parameters
   - Support Vector Machines with appropriate kernel and regularization

**Strategies to Find Optimal Balance:**

1. **Cross-validation**: Test model on unseen data to estimate generalization performance

2. **Regularization**: Add penalty for model complexity
   - L1/L2 regularization for linear models
   - Pruning for decision trees
   - Dropout for neural networks

3. **Ensemble methods**: Combine multiple models to reduce overall variance
   - Bagging (e.g., Random Forests): Reduces variance
   - Boosting (e.g., AdaBoost, Gradient Boosting): Reduces bias

4. **Learning curves**: Plot training and validation errors versus training set size
   - High bias: Both errors converge to a high value
   - High variance: Large gap between training and validation errors

**Code Example - Learning Curves for Bias-Variance Analysis:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 1) * 5
y = np.sin(X).ravel() + np.random.normal(0, 0.1, n_samples)

# Models with different complexity (low to high)
models = {
    'Ridge (High Bias)': make_pipeline(StandardScaler(), Ridge(alpha=10)),
    'Ridge (Balanced)': make_pipeline(StandardScaler(), Ridge(alpha=1)),
    'SVR RBF (High Variance)': make_pipeline(StandardScaler(), SVR(kernel='rbf', gamma=2))
}

# Create the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot learning curves for each model
train_sizes = np.linspace(0.1, 1.0, 10)

for i, (name, model) in enumerate(models.items()):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error')
    
    # Convert negative MSE to positive for intuitive visualization
    train_scores_mean = -np.mean(train_scores, axis=1)
    valid_scores_mean = -np.mean(valid_scores, axis=1)
    
    # Plot learning curve
    axes[i].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training error')
    axes[i].plot(train_sizes, valid_scores_mean, 'o-', color='g', label='Validation error')
    axes[i].set_title(name)
    axes[i].set_xlabel('Training set size')
    axes[i].set_ylabel('Mean Squared Error')
    axes[i].grid(True)
    axes[i].legend(loc='best')

plt.tight_layout()
```

**Applications in Practice:**

1. **Feature Selection**: Remove irrelevant features to reduce variance without increasing bias

2. **Hyperparameter Tuning**: Find optimal hyperparameter values that balance bias and variance
   - Decision tree depth
   - Neural network size and regularization strength
   - k in k-Nearest Neighbors

3. **Model Selection**: Choose an algorithm with the right complexity for your data
   - Simple models for small datasets
   - Complex models with strong regularization for large datasets

**Key Takeaways:**

1. Every machine learning model must balance bias and variance to minimize total error
2. Adding complexity reduces bias but increases variance
3. The optimal model complexity depends on the amount and quality of training data
4. Cross-validation is essential to find the sweet spot between underfitting and overfitting
5. As more training data becomes available, you can afford to use more complex models (lower bias) without excessive variance

### Q: Explain the difference between L1 and L2 regularization and their effects on models.

**Answer:**

L1 and L2 regularization are two common techniques used to prevent overfitting in machine learning models by adding a penalty term to the loss function. While they serve the same general purpose, they have different mathematical properties and effects on the resulting models.

**Basic Concepts:**

Regularization adds a penalty term to the model's loss function:

**Regularized Loss = Loss Function + λ × Regularization Term**

Where:
- λ (lambda) is the regularization strength (hyperparameter)
- Higher λ values enforce stronger regularization

**L1 Regularization (Lasso):**

Adds the sum of the absolute values of the coefficients to the loss function:

**L1 Term = λ × Σ|w_i|**

Where w_i are the model coefficients/weights.

**L2 Regularization (Ridge):**

Adds the sum of the squared values of the coefficients to the loss function:

**L2 Term = λ × Σw_i²**

**Key Differences and Effects:**

| Aspect | L1 Regularization (Lasso) | L2 Regularization (Ridge) |
|--------|------------------------|------------------------|
| **Mathematical Form** | λ × Σ\|w_i\| | λ × Σw_i² |
| **Effect on Weights** | Drives some weights exactly to zero (sparse solutions) | Shrinks weights toward zero but rarely exactly to zero |
| **Feature Selection** | Performs automatic feature selection | Does not perform feature selection |
| **Solution Uniqueness** | May not have a unique solution if features are correlated | Always has a unique solution |
| **Computational Efficiency** | Less efficient to compute (non-differentiable at zero) | More efficient to compute (differentiable everywhere) |
| **Penalty Behavior** | Penalizes small weights more heavily than L2 | Penalizes large weights more heavily than L1 |
| **Geometric Interpretation** | Constraint region is a diamond (L1 norm ball) | Constraint region is a circle (L2 norm ball) |
| **Best Used When** | Many features are expected to be irrelevant | Most or all features contribute to the outcome |

**Mathematical Perspective:**

For linear regression, the formulations are:

**Ridge (L2):**
- Minimize: ||y - Xw||² + λ||w||²
- Closed-form solution: w = (X^T X + λI)^(-1) X^T y

**Lasso (L1):**
- Minimize: ||y - Xw||² + λ||w||₁
- No closed-form solution; requires iterative optimization

**Visual Interpretation:**

The key to understanding why L1 yields sparse solutions while L2 doesn't lies in the geometry of the constraint regions:

```
L1 Constraint (Diamond)    L2 Constraint (Circle)
       w₂                       w₂
        |                        |
        |                        |
        |       /\               |      __
        |      /  \              |     /  \
        |     /    \             |    /    \
        |    /      \            |   /      \
--------+---/--------\----- w₁  -+--/--------\--- w₁
        |  /          \          |  /          \
        | /            \         | /            \
        |/              \        |/              \
        /                \       /                \
       /                  \     /                  \
      /____________________\   /____________________\
```

When the loss function contour intersects with the constraint region:
- L1: Intersection often occurs at corners (where some weights are exactly zero)
- L2: Intersection can occur anywhere on the circle (rarely at axes)

**Practical Examples:**

1. **L1 (Lasso) Example:**
   - Creates sparse models by setting some coefficients to zero
   - Useful when you suspect many features are irrelevant
   - Example: Gene expression analysis with thousands of genes but only a small subset is relevant

2. **L2 (Ridge) Example:**
   - Preserves all features but reduces their impact
   - Useful when most features contribute somewhat to the prediction
   - Example: Image recognition where all pixels might contain useful information

3. **Elastic Net:**
   - Combines L1 and L2 regularization: α||w||₁ + (1-α)||w||²
   - Gets the best of both worlds: feature selection and handling of correlated features
   - Example: Recommendation systems with many features, some of which may be correlated

**Code Example - Effects of L1 vs L2 Regularization:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data with some irrelevant features
X, y = make_regression(n_samples=200, n_features=50, n_informative=10, 
                      random_state=42, noise=30)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with different regularization
alpha_value = 0.1

# No regularization (OLS)
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)
ols_coef = ols.coef_

# L1 Regularization (Lasso)
lasso = Lasso(alpha=alpha_value)
lasso.fit(X_train_scaled, y_train)
lasso_coef = lasso.coef_

# L2 Regularization (Ridge)
ridge = Ridge(alpha=alpha_value)
ridge.fit(X_train_scaled, y_train)
ridge_coef = ridge.coef_

# Elastic Net (Combination)
elastic = ElasticNet(alpha=alpha_value, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
elastic_coef = elastic.coef_

# Plot coefficients
plt.figure(figsize=(15, 8))
plt.plot(ols_coef, 's-', label='No regularization')
plt.plot(lasso_coef, 'o-', label=f'L1 (Lasso, α={alpha_value})')
plt.plot(ridge_coef, '^-', label=f'L2 (Ridge, α={alpha_value})')
plt.plot(elastic_coef, 'd-', label=f'Elastic Net (α={alpha_value}, l1_ratio=0.5)')

plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Effect of Different Regularization Methods on Model Coefficients')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Print sparsity information
lasso_nonzero = np.sum(lasso_coef != 0)
ridge_nonzero = np.sum(ridge_coef != 0)
elastic_nonzero = np.sum(elastic_coef != 0)

print(f"Original features: {X.shape[1]}")
print(f"Non-zero coefficients in Lasso: {lasso_nonzero} ({lasso_nonzero/X.shape[1]:.1%})")
print(f"Non-zero coefficients in Ridge: {ridge_nonzero} ({ridge_nonzero/X.shape[1]:.1%})")
print(f"Non-zero coefficients in ElasticNet: {elastic_nonzero} ({elastic_nonzero/X.shape[1]:.1%})")
```

**Choosing Between L1 and L2:**

1. **Choose L1 (Lasso) when:**
   - You have many features and suspect many are irrelevant
   - You want an interpretable model with fewer features
   - Feature selection is a priority
   - You're comfortable with potentially unstable feature selection if features are correlated

2. **Choose L2 (Ridge) when:**
   - Most features are likely relevant
   - Features may be correlated with each other
   - You want stable predictions with small coefficient values
   - You don't need explicit feature selection

3. **Choose Elastic Net when:**
   - You want a balance between feature selection and handling correlated features
   - The number of features greatly exceeds the number of observations
   - You're dealing with groups of correlated features

**Hyperparameter Tuning:**

The regularization strength (λ) is a critical hyperparameter:
- Too high: Underfitting (high bias)
- Too low: Overfitting (high variance)
- Optimal: Determined through cross-validation

**Beyond Linear Models:**

The concepts extend to other models:
- Neural Networks: Weight decay (L2) and sparse activations
- Decision Trees: Pruning and minimum samples per leaf
- SVMs: Regularization parameter C (inverse of λ)

**Advanced Considerations:**

1. **Scale Sensitivity:**
   - Both L1 and L2 are sensitive to feature scaling
   - Always standardize features before applying regularization

2. **Grouped Selection:**
   - Group Lasso for selecting/excluding groups of features together

3. **Bayesian Interpretation:**
   - L1: Laplace prior on weights
   - L2: Gaussian prior on weights

### Q: What is the curse of dimensionality in machine learning and how do we address it?

**Answer:**

The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces, which do not occur in low-dimensional settings. This concept is foundational in machine learning as it explains why many algorithms struggle with high-dimensional data.

**Key Manifestations:**

1. **Exponential Growth of Space**
   - As dimensions increase, the volume of the space grows exponentially
   - Data becomes increasingly sparse in higher dimensions
   - The number of samples needed to maintain the same density grows exponentially

2. **Distance Concentration**
   - In high dimensions, the distance between any pair of points becomes almost uniform
   - The ratio of the difference between the maximum and minimum distances to the minimum distance approaches zero
   - This undermines algorithms that rely on meaningful distance metrics (like k-NN)

3. **Boundary vs. Interior Points**
   - Most points in a high-dimensional hypercube lie close to the edges rather than the interior
   - This makes sampling and interpolation more difficult

4. **Computational Complexity**
   - The computational requirements for many algorithms grow exponentially with dimensions
   - Training times become prohibitive

5. **Statistical Significance**
   - More features increase the chance of finding spurious correlations
   - Higher risk of overfitting

**Mathematical Illustration:**

Consider a unit hypercube in d dimensions:
- The volume is always 1, regardless of dimensionality
- To sample 10% of the volume in 1D: need to sample 10% of each axis
- To sample 10% of the volume in 10D: need to sample 80% of each axis (0.8^10 ≈ 0.1)
- To sample 10% of the volume in 100D: need to sample 98% of each axis (0.98^100 ≈ 0.1)

**Distance Concentration Formula:**
For i.i.d. random points in high dimensions, the variance of the distances becomes smaller relative to the mean:

$$\frac{\mathbb{E}[d(X,Y)^2] - \mathbb{E}[d(X,Y)]^2}{\mathbb{E}[d(X,Y)]^2} \to 0 \text{ as } d \to \infty$$

**Strategies to Address the Curse of Dimensionality:**

1. **Dimensionality Reduction Techniques**
   - Principal Component Analysis (PCA)
   - t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - UMAP (Uniform Manifold Approximation and Projection)
   - Autoencoders

2. **Feature Selection**
   - Filter methods: Correlation, mutual information
   - Wrapper methods: Forward selection, backward elimination
   - Embedded methods: Lasso regularization, tree-based importance

3. **Feature Engineering**
   - Create meaningful combinations of existing features
   - Apply domain knowledge to reduce dimensionality

4. **Regularization**
   - L1/L2 regularization to reduce model complexity
   - Dropout in neural networks

5. **Specialized Algorithms for High Dimensions**
   - Random forests (less affected by high dimensions)
   - Approximate nearest neighbors algorithms
   - Locality-sensitive hashing

6. **Increasing Sample Size**
   - Collect more data to combat sparsity
   - Data augmentation techniques

7. **Manifold Learning**
   - Assume data lies on a lower-dimensional manifold within the high-dimensional space
   - Manifold learning algorithms like Isomap, LLE (Locally Linear Embedding)

**Code Example - Visualizing Distance Concentration:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Function to generate random points and compute pairwise distances
def analyze_distances(n_points, dimensions, n_trials=10):
    mean_distances = []
    std_distances = []
    rel_variations = []  # Relative variation (std/mean)
    
    for dim in dimensions:
        trial_means = []
        trial_stds = []
        
        for _ in range(n_trials):
            # Generate random points in dim-dimensional unit hypercube
            points = np.random.random((n_points, dim))
            
            # Compute pairwise distances
            dists = euclidean_distances(points)
            
            # Only consider upper triangle, excluding diagonal
            distances = dists[np.triu_indices_from(dists, k=1)]
            
            trial_means.append(distances.mean())
            trial_stds.append(distances.std())
        
        # Average over trials
        mean_dist = np.mean(trial_means)
        std_dist = np.mean(trial_stds)
        rel_var = std_dist / mean_dist
        
        mean_distances.append(mean_dist)
        std_distances.append(std_dist)
        rel_variations.append(rel_var)
    
    return mean_distances, std_distances, rel_variations

# Setup
dimensions = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
n_points = 1000

# Run analysis
mean_distances, std_distances, rel_variations = analyze_distances(n_points, dimensions)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot mean and std of distances
ax1.set_xlabel('Dimensions (log scale)')
ax1.set_ylabel('Distance')
ax1.plot(dimensions, mean_distances, 'bo-', label='Mean Distance')
ax1.plot(dimensions, std_distances, 'go-', label='Standard Deviation')
ax1.set_xscale('log')
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)

# Plot relative variation on secondary axis
ax2 = ax1.twinx()
ax2.set_ylabel('Relative Variation (Std/Mean)')
ax2.plot(dimensions, rel_variations, 'ro-', label='Relative Variation')
ax2.tick_params(axis='y')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Distance Concentration in High Dimensions')
plt.xscale('log')
plt.tight_layout()
```

**Practical Example - Effect on k-NN Performance:**

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import time

# Function to evaluate kNN in different dimensions
def evaluate_knn(dims, n_samples=1000, k=5):
    accuracies = []
    training_times = []
    
    for dim in dims:
        # Create dataset with relevant and noise features
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=dim,
            n_informative=min(10, dim),  # Only 10 informative features
            n_redundant=0,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train and time kNN
        start_time = time.time()
        knn = K
