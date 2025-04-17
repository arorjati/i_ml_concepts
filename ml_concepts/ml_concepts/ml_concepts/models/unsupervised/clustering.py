"""
Clustering models for unsupervised learning.

This module implements various clustering algorithms for unsupervised learning,
including mathematical foundations and detailed explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage


class KMeansModel:
    """
    K-Means clustering implementation with detailed mathematical explanation.
    
    K-Means is a centroid-based clustering algorithm that partitions a dataset into k clusters
    by minimizing the within-cluster sum-of-squares (inertia):
        
        minimize Sum(Sum(||x_i - μ_j||^2))
        
    where:
        - x_i is a data point in cluster j
        - μ_j is the centroid of cluster j
        - ||x_i - μ_j||^2 is the squared Euclidean distance
    """
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', 
               n_init: int = 10, max_iter: int = 300, tol: float = 1e-4,
               random_state: Optional[int] = None, algorithm: str = 'auto'):
        """
        Initialize a KMeansModel instance.
        
        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 8.
            init (str, optional): Method for initialization. Defaults to 'k-means++'.
            n_init (int, optional): Number of times to run with different centroid seeds. Defaults to 10.
            max_iter (int, optional): Maximum number of iterations. Defaults to 300.
            tol (float, optional): Relative tolerance for convergence. Defaults to 1e-4.
            random_state (Optional[int], optional): Random seed for reproducible results. Defaults to None.
            algorithm (str, optional): K-means algorithm to use. Defaults to 'auto'.
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm
        )
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.is_fitted = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'KMeansModel':
        """
        Fit the K-Means clustering model.
        
        The K-Means algorithm works as follows:
        1. Initialize k centroids (e.g., randomly or using k-means++)
        2. Repeat until convergence:
           a. Assign each data point to the nearest centroid
           b. Update centroids as the mean of all data points assigned to that centroid
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            
        Returns:
            KMeansModel: Fitted model.
        """
        self.model.fit(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): New data.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.predict(X)
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the model and predict clusters for X in a single call.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        return self.model.fit_predict(X)
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform X to a cluster-distance space.
        
        In the new space, each dimension is the distance to a cluster center.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): New data.
            
        Returns:
            np.ndarray: X transformed in the cluster-distance space.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        return self.model.transform(X)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
        """
        Evaluate the clustering quality using various metrics.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data to evaluate.
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        # Get cluster predictions
        y_pred = self.predict(X_np)
        
        # Only calculate silhouette score if there is more than one cluster
        # and we have more samples than clusters
        metrics = {
            'inertia': self.inertia_,
            'n_clusters': self.n_clusters
        }
        
        if self.n_clusters > 1 and X_np.shape[0] > self.n_clusters:
            metrics['silhouette_score'] = silhouette_score(X_np, y_pred)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_np, y_pred)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_np, y_pred)
            
        return metrics
    
    def elbow_method(self, X: Union[np.ndarray, pd.DataFrame], max_clusters: int = 10,
                    random_state: Optional[int] = None) -> plt.Figure:
        """
        Run the elbow method to find the optimal number of clusters.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data to cluster.
            max_clusters (int, optional): Maximum number of clusters to try. Defaults to 10.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            
        Returns:
            plt.Figure: Matplotlib figure with the elbow plot.
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        # Calculate inertia (within-cluster sum of squares) for different k values
        inertia_values = []
        cluster_range = range(1, max_clusters + 1)
        
        for k in cluster_range:
            model = KMeans(n_clusters=k, random_state=random_state)
            model.fit(X_np)
            inertia_values.append(model.inertia_)
            
        # Create the elbow plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cluster_range, inertia_values, 'bo-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        ax.grid(True, alpha=0.3)
        
        # Add explanation text
        ax.text(max_clusters * 0.6, inertia_values[0] * 0.8,
               "Look for the 'elbow' point where\n"
               "adding more clusters doesn't\n"
               "significantly reduce inertia",
               bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def silhouette_analysis(self, X: Union[np.ndarray, pd.DataFrame], 
                           cluster_range: Optional[List[int]] = None,
                           random_state: Optional[int] = None) -> plt.Figure:
        """
        Perform silhouette analysis to find the optimal number of clusters.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data to cluster.
            cluster_range (Optional[List[int]], optional): Range of clusters to try. 
                                                         Defaults to None (2 to 10).
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            
        Returns:
            plt.Figure: Matplotlib figure with the silhouette analysis.
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        # Default cluster range if not specified
        if cluster_range is None:
            cluster_range = range(2, 11)
            
        # Calculate silhouette score for different k values
        silhouette_scores = []
        
        for k in cluster_range:
            model = KMeans(n_clusters=k, random_state=random_state)
            labels = model.fit_predict(X_np)
            
            # Only calculate silhouette score if there is more than one cluster
            # and we have more samples than clusters
            if k > 1 and X_np.shape[0] > k:
                score = silhouette_score(X_np, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
                
        # Create the silhouette score plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cluster_range, silhouette_scores, 'bo-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Analysis for Optimal k')
        ax.grid(True, alpha=0.3)
        
        # Add explanation text
        max_score_idx = np.argmax(silhouette_scores)
        best_k = list(cluster_range)[max_score_idx]
        max_score = silhouette_scores[max_score_idx]
        
        ax.text(cluster_range[-1] * 0.6, max_score * 0.9,
               f"Optimal k = {best_k}\n"
               "Higher silhouette score indicates\n"
               "better-defined clusters",
               bbox=dict(facecolor='white', alpha=0.8))
        
        # Highlight the optimal k
        ax.scatter([best_k], [max_score], s=100, c='red', zorder=10)
        ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def visualize_clusters(self, X: Union[np.ndarray, pd.DataFrame], 
                          feature_indices: Optional[Tuple[int, int]] = None,
                          title: Optional[str] = None) -> plt.Figure:
        """
        Visualize the clusters in a 2D plot.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data used for clustering.
            feature_indices (Optional[Tuple[int, int]], optional): Indices of two features to plot. 
                                                                Defaults to first two features.
            title (Optional[str], optional): Plot title. Defaults to None.
            
        Returns:
            plt.Figure: Matplotlib figure with the cluster visualization.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X_np = X.values
        else:
            X_np = X
            feature_names = [f"Feature {i}" for i in range(X_np.shape[1])]
            
        # If data has more than 2 dimensions, select 2 features to plot
        if X_np.shape[1] > 2:
            if feature_indices is None:
                feature_indices = (0, 1)
                
            X_plot = X_np[:, feature_indices]
            x_label = feature_names[feature_indices[0]]
            y_label = feature_names[feature_indices[1]]
        else:
            X_plot = X_np
            x_label = feature_names[0]
            y_label = feature_names[1] if len(feature_names) > 1 else "Feature 1"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data points colored by cluster
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=self.labels_, cmap='viridis', 
                           alpha=0.6, s=50)
        
        # Plot cluster centers
        if X_np.shape[1] > 2:
            centers = self.cluster_centers_[:, feature_indices]
        else:
            centers = self.cluster_centers_
            
        ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', alpha=0.8, 
                  marker='X', label='Centroids')
        
        # Add legend, title and labels
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.legend()
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'KMeans Clustering (k={self.n_clusters})', fontsize=14)
            
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class DBSCANModel:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation
    with detailed mathematical explanation.
    
    DBSCAN is a density-based clustering algorithm that groups together points that are 
    closely packed together, while marking points in low-density regions as outliers.
    
    The algorithm works by:
    1. Finding core points that have at least 'min_samples' points within 'eps' distance
    2. Connecting core points that are within 'eps' distance to form clusters
    3. Assigning non-core points that are within 'eps' of a core point to that cluster
    4. Labeling points not assigned to any cluster as noise (outliers)
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
               metric: str = 'euclidean', algorithm: str = 'auto',
               leaf_size: int = 30, p: Optional[float] = None,
               n_jobs: Optional[int] = None):
        """
        Initialize a DBSCANModel instance.
        
        Args:
            eps (float, optional): Maximum distance between two samples for them to be considered
                                 in the same neighborhood. Defaults to 0.5.
            min_samples (int, optional): Minimum number of samples in a neighborhood for a point
                                       to be considered a core point. Defaults to 5.
            metric (str, optional): Metric used to compute distances. Defaults to 'euclidean'.
            algorithm (str, optional): Algorithm used to compute nearest neighbors. Defaults to 'auto'.
            leaf_size (int, optional): Leaf size for BallTree or KDTree. Defaults to 30.
            p (Optional[float], optional): Power parameter for Minkowski metric. Defaults to None.
            n_jobs (Optional[int], optional): Number of parallel jobs. Defaults to None.
        """
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs
        )
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self.is_fitted = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'DBSCANModel':
        """
        Fit the DBSCAN clustering model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            
        Returns:
            DBSCANModel: Fitted model.
        """
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.core_sample_indices_ = self.model.core_sample_indices_
        
        # Calculate number of clusters (ignoring noise points with label -1)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.is_fitted = True
        return self
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the model and return cluster labels.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
            
        Returns:
            np.ndarray: Cluster labels for each point in the dataset.
        """
        self.fit(X)
        return self.labels_
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate the clustering quality using various metrics.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data to evaluate.
            
        Returns:
            Dict[str, Any]: Dictionary with evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        # Calculate metrics that don't require ground truth labels
        metrics = {
            'n_clusters': self.n_clusters_,
            'n_noise': list(self.labels_).count(-1),
            'ratio_noise': list(self.labels_).count(-1) / len(self.labels_)
        }
        
        # Only calculate additional metrics if there is more than one cluster
        if self.n_clusters_ > 1:
            # Filter out noise points for these metrics
            non_noise_mask = self.labels_ != -1
            if np.sum(non_noise_mask) > self.n_clusters_:
                X_non_noise = X_np[non_noise_mask]
                labels_non_noise = self.labels_[non_noise_mask]
                metrics['silhouette_score'] = silhouette_score(X_non_noise, labels_non_noise)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_non_noise, labels_non_noise)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_non_noise, labels_non_noise)
            
        return metrics
    
    def eps_analysis(self, X: Union[np.ndarray, pd.DataFrame], 
                    eps_range: Optional[List[float]] = None,
                    min_samples: Optional[int] = None) -> plt.Figure:
        """
        Analyze the effect of different eps values on clustering results.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data to cluster.
            eps_range (Optional[List[float]], optional): Range of eps values to try. 
                                                      Defaults to None (0.1 to 1.0).
            min_samples (Optional[int], optional): Min samples for DBSCAN. 
                                                Defaults to None (use self.min_samples).
            
        Returns:
            plt.Figure: Matplotlib figure with the eps analysis.
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        # Default eps range if not specified
        if eps_range is None:
            eps_range = np.linspace(0.1, 1.0, 10)
            
        # Use instance min_samples if not specified
        if min_samples is None:
            min_samples = self.min_samples
            
        # Calculate metrics for different eps values
        n_clusters_list = []
        n_noise_list = []
        
        for eps in eps_range:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_np)
            
            # Calculate number of clusters (ignoring noise points with label -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            n_clusters_list.append(n_clusters)
            n_noise_list.append(n_noise / len(labels))  # Ratio of noise points
            
        # Create the analysis plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot number of clusters
        color = 'tab:blue'
        ax1.set_xlabel('Epsilon (ε)')
        ax1.set_ylabel('Number of Clusters', color=color)
        ax1.plot(eps_range, n_clusters_list, 'o-', color=color, label='Number of Clusters')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for noise ratio
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Ratio of Noise Points', color=color)
        ax2.plot(eps_range, n_noise_list, 'o-', color=color, label='Noise Ratio')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title and legend
        title = f'DBSCAN: Effect of Epsilon (min_samples={min_samples})'
        fig.suptitle(title, fontsize=14)
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        
        # Add explanation text
        text = ("As ε increases:\n"
               "- Small ε: More noise, many small clusters\n"
               "- Large ε: Less noise, fewer large clusters\n"
               "Look for ε where noise stabilizes but\n"
               "before clusters start merging")
        
        fig.text(0.15, 0.02, text, fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig
    
    def visualize_clusters(self, X: Union[np.ndarray, pd.DataFrame], 
                          feature_indices: Optional[Tuple[int, int]] = None,
                          title: Optional[str] = None) -> plt.Figure:
        """
        Visualize the clusters and noise points in a 2D plot.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Data used for clustering.
            feature_indices (Optional[Tuple[int, int]], optional): Indices of two features to plot. 
                                                                Defaults to first two features.
            title (Optional[str], optional): Plot title. Defaults to None.
            
        Returns:
            plt.Figure: Matplotlib figure with the cluster visualization.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X_np = X.values
        else:
            X_np = X
            feature_names = [f"Feature {i}" for i in range(X_np.shape[1])]
            
        # If data has more than 2 dimensions, select 2 features to plot
        if X_np.shape[1] > 2:
            if feature_indices is None:
                feature_indices = (0, 1)
                
            X_plot = X_np[:, feature_indices]
            x_label = feature_names[feature_indices[0]]
            y_label = feature_names[feature_indices[1]]
        else:
            X_plot = X_np
            x_label = feature_names[0]
            y_label = feature_names[1] if len(feature_names) > 1 else "Feature 1"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a colormap with a specific color for noise points (-1)
        unique_labels = set(self.labels_)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in unique_labels else 0)))
        color_dict = {i: colors[j] for j, i in enumerate(sorted(list(unique_labels - {-1})))}
        # Add black for noise points
        if -1 in unique_labels:
            color_dict[-1] = [0, 0, 0, 1]  # Black for noise
        
        # Plot data points colored by cluster
        for label, color in color_dict.items():
            mask = self.labels_ == label
            marker = 'x' if label == -1 else 'o'
            label_text = 'Noise' if label == -1 else f'Cluster {label}'
            ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[color], marker=marker, 
                      s=50 if label == -1 else 60, alpha=0.6 if label == -1 else 0.7,
                      label=label_text)
        
        # Add title and labels
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'DBSCAN Clustering (eps={self.eps}, min_samples={self.min_samples}, clusters={self.n_clusters_})', 
                        fontsize=14)
            
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class HierarchicalClusteringModel:
    """
    Hierarchical Clustering implementation with detailed mathematical explanation.
    
    Hierarchical clustering builds nested clusters by merging or splitting them successively.
    This creates a cluster hierarchy visualized as a dendrogram.
    
    Two main approaches:
    1. Agglomerative (bottom-up): Each observation starts in its own cluster, and pairs of 
       clusters are merged as one moves up the hierarchy.
    2. Divisive (top-down): All observations start in one cluster, and splits are performed 
       recursively as one moves down the hierarchy.
    """
    
    def __init__(self, n_clusters: int = 2, affinity: str = 'euclidean', 
               linkage: str = 'ward', distance_threshold: Optional[float] = None,
               compute_full_tree: Union[str, bool] = 'auto',
               memory: Optional[str] = None):
        """
        Initialize a HierarchicalClusteringModel instance using scikit-learn's AgglomerativeClustering.
        
        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 2.
                                      If distance_threshold is not None, n_clusters is ignored.
            affinity (str, optional): Metric used to compute linkage. Defaults to 'euclidean'.
            linkage (str, optional): Linkage criterion: 'ward', 'complete', 'average', 'single'.
                                   Defaults to 'ward'.
            distance_threshold (Optional[float], optional): The linkage distance threshold
                                                          for determining clusters. Defaults to None.
            compute_full_tree (Union[str, bool], optional): When to compute the full tree.
                                                          Defaults to 'auto'.
            memory (Optional[str], optional): Memory location for caching. Defaults to None.
        """
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_full_tree=compute_full_tree,
            memory=memory
        )
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.labels_ = None
        self.n_clusters_ = None
        self
