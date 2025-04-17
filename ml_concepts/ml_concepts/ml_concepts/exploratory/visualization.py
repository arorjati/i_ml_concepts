"""
Visualization utilities for exploratory data analysis.

This module provides functions for creating various types of visualizations for data exploration, including:
- Univariate analysis (histograms, box plots, density plots)
- Bivariate analysis (scatter plots, correlation heatmaps)
- Multivariate analysis (pair plots, dimensionality reduction visualization)
- Categorical data visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


class Visualizer:
    """Class for creating visualizations for exploratory data analysis."""
    
    # Set default style
    @staticmethod
    def set_style(style: str = 'whitegrid', context: str = 'paper', 
                 palette: str = 'viridis', font_scale: float = 1.2):
        """
        Set the default visualization style.
        
        Args:
            style (str, optional): Seaborn style. Defaults to 'whitegrid'.
            context (str, optional): Seaborn context. Defaults to 'paper'.
            palette (str, optional): Color palette. Defaults to 'viridis'.
            font_scale (float, optional): Font scaling factor. Defaults to 1.2.
        """
        sns.set_theme(style=style, context=context, palette=palette, font_scale=font_scale)
    
    # Univariate Analysis
    @staticmethod
    def plot_histogram(data: Union[pd.Series, pd.DataFrame], 
                      column: Optional[str] = None,
                      bins: int = 30,
                      kde: bool = True,
                      rug: bool = False,
                      figsize: Tuple[int, int] = (10, 6),
                      title: Optional[str] = None,
                      xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None,
                      color: Optional[str] = None,
                      ax: Optional[plt.Axes] = None,
                      **kwargs) -> plt.Axes:
        """
        Plot a histogram with optional KDE and rug plot.
        
        Args:
            data (Union[pd.Series, pd.DataFrame]): The data to plot.
            column (Optional[str], optional): Column name if data is a DataFrame. Defaults to None.
            bins (int, optional): Number of bins. Defaults to 30.
            kde (bool, optional): Whether to plot KDE. Defaults to True.
            rug (bool, optional): Whether to plot rug. Defaults to False.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            xlabel (Optional[str], optional): X-axis label. Defaults to None.
            ylabel (Optional[str], optional): Y-axis label. Defaults to None.
            color (Optional[str], optional): Plot color. Defaults to None.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            **kwargs: Additional arguments to pass to seaborn's histplot.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Handle input data
        if column is not None and isinstance(data, pd.DataFrame):
            if column in data.columns:
                data = data[column]
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Create histogram
        sns.histplot(data, bins=bins, kde=kde, rug=rug, color=color, ax=ax, **kwargs)
        
        # Set labels and title
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        elif column and isinstance(data, pd.Series):
            ax.set_xlabel(column, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
            
        # Add annotations with descriptive statistics
        if isinstance(data, pd.Series):
            stats_text = (
                f"Mean: {data.mean():.2f}\n"
                f"Median: {data.median():.2f}\n"
                f"Std Dev: {data.std():.2f}\n"
                f"Min: {data.min():.2f}\n"
                f"Max: {data.max():.2f}"
            )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        return ax
    
    @staticmethod
    def plot_boxplot(data: Union[pd.Series, pd.DataFrame], 
                    column: Optional[str] = None,
                    by: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    ax: Optional[plt.Axes] = None,
                    vert: bool = True,
                    showfliers: bool = True,
                    **kwargs) -> plt.Axes:
        """
        Create a box plot.
        
        Args:
            data (Union[pd.Series, pd.DataFrame]): The data to plot.
            column (Optional[str], optional): Column name if data is a DataFrame. Defaults to None.
            by (Optional[str], optional): Column to group by for multiple boxplots. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            xlabel (Optional[str], optional): X-axis label. Defaults to None.
            ylabel (Optional[str], optional): Y-axis label. Defaults to None.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            vert (bool, optional): Whether to plot vertical boxplot. Defaults to True.
            showfliers (bool, optional): Whether to show outlier points. Defaults to True.
            **kwargs: Additional arguments to pass to seaborn's boxplot.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Handle input data
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame
            data = pd.DataFrame(data)
            column = data.columns[0]
            
        if by is not None:
            # Multiple boxplots grouped by 'by' column
            if column is None:
                # If column is not specified, use all numeric columns
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    warnings.warn("No numeric columns found for boxplot.")
                    return ax
                
                # Melt the DataFrame to long format
                melted_data = pd.melt(data, id_vars=[by], value_vars=numeric_cols)
                sns.boxplot(x='variable', y='value', hue=by, data=melted_data, ax=ax, 
                           vert=vert, showfliers=showfliers, **kwargs)
            else:
                # Create boxplot for a single column grouped by 'by'
                if vert:
                    sns.boxplot(x=by, y=column, data=data, ax=ax, showfliers=showfliers, **kwargs)
                else:
                    sns.boxplot(x=column, y=by, data=data, ax=ax, showfliers=showfliers, **kwargs)
        else:
            # Single boxplot
            if column is None:
                # If column is not specified, use all numeric columns
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    warnings.warn("No numeric columns found for boxplot.")
                    return ax
                    
                if vert:
                    sns.boxplot(data=data[numeric_cols], ax=ax, showfliers=showfliers, **kwargs)
                else:
                    # Transpose for horizontal boxplot of multiple columns
                    sns.boxplot(data=data[numeric_cols].T, ax=ax, showfliers=showfliers, **kwargs)
            else:
                # Create boxplot for a single column
                sns.boxplot(x=data[column] if vert else None, 
                           y=None if vert else data[column], 
                           ax=ax, showfliers=showfliers, **kwargs)
        
        # Set labels and title
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
            
        # Rotate x-tick labels if there are many categories
        if by is not None or (column is None and not vert):
            plt.xticks(rotation=45, ha='right')
            
        return ax
    
    @staticmethod
    def plot_density(data: Union[pd.Series, pd.DataFrame], 
                    column: Optional[str] = None,
                    by: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    ax: Optional[plt.Axes] = None,
                    shade: bool = True,
                    **kwargs) -> plt.Axes:
        """
        Create a kernel density estimation (KDE) plot.
        
        Args:
            data (Union[pd.Series, pd.DataFrame]): The data to plot.
            column (Optional[str], optional): Column name if data is a DataFrame. Defaults to None.
            by (Optional[str], optional): Column to group by for multiple densities. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            xlabel (Optional[str], optional): X-axis label. Defaults to None.
            ylabel (Optional[str], optional): Y-axis label. Defaults to None.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            shade (bool, optional): Whether to shade the density. Defaults to True.
            **kwargs: Additional arguments to pass to seaborn's kdeplot.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Handle input data
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame
            data = pd.DataFrame(data)
            column = data.columns[0]
            
        if by is not None and column is not None:
            # Multiple density plots grouped by 'by' column
            for category, group in data.groupby(by):
                sns.kdeplot(group[column].dropna(), label=f"{by}={category}", ax=ax, shade=shade, **kwargs)
            ax.legend()
        else:
            if column is None:
                # If column is not specified, use all numeric columns
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    warnings.warn("No numeric columns found for density plot.")
                    return ax
                    
                # Create multiple density plots
                for col in numeric_cols:
                    sns.kdeplot(data[col].dropna(), label=col, ax=ax, shade=shade, **kwargs)
                ax.legend()
            else:
                # Create density plot for a single column
                sns.kdeplot(data[column].dropna(), ax=ax, shade=shade, **kwargs)
        
        # Set labels and title
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        elif column:
            ax.set_xlabel(column, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel('Density', fontsize=12)
            
        return ax
    
    @staticmethod
    def plot_qq(data: Union[pd.Series, pd.DataFrame], 
               column: Optional[str] = None,
               figsize: Tuple[int, int] = (10, 6),
               title: Optional[str] = None,
               ax: Optional[plt.Axes] = None,
               **kwargs) -> plt.Axes:
        """
        Create a Q-Q plot to check for normality.
        
        Args:
            data (Union[pd.Series, pd.DataFrame]): The data to plot.
            column (Optional[str], optional): Column name if data is a DataFrame. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            **kwargs: Additional arguments to pass to scipy's probplot.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Handle input data
        if column is not None and isinstance(data, pd.DataFrame):
            if column in data.columns:
                data = data[column]
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
        # Convert Series to numpy array
        if isinstance(data, pd.Series):
            data_array = data.dropna().values
        else:
            raise ValueError("Data must be a pandas Series or a DataFrame with a specified column")
                
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Create Q-Q plot
        stats.probplot(data_array, plot=ax, **kwargs)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14)
        elif column:
            ax.set_title(f"Q-Q Plot: {column}", fontsize=14)
        else:
            ax.set_title("Q-Q Plot", fontsize=14)
            
        return ax
    
    @staticmethod
    def plot_ecdf(data: Union[pd.Series, pd.DataFrame], 
                 column: Optional[str] = None,
                 by: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 ax: Optional[plt.Axes] = None,
                 **kwargs) -> plt.Axes:
        """
        Create an empirical cumulative distribution function (ECDF) plot.
        
        Args:
            data (Union[pd.Series, pd.DataFrame]): The data to plot.
            column (Optional[str], optional): Column name if data is a DataFrame. Defaults to None.
            by (Optional[str], optional): Column to group by for multiple ECDFs. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            xlabel (Optional[str], optional): X-axis label. Defaults to None.
            ylabel (Optional[str], optional): Y-axis label. Defaults to None.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            **kwargs: Additional arguments to pass to sns.ecdfplot.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Handle input data
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame
            data = pd.DataFrame(data)
            column = data.columns[0]
            
        if by is not None and column is not None:
            # Multiple ECDF plots grouped by 'by' column
            sns.ecdfplot(data=data, x=column, hue=by, ax=ax, **kwargs)
        else:
            if column is None:
                # If column is not specified, use all numeric columns
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    warnings.warn("No numeric columns found for ECDF plot.")
                    return ax
                    
                # Melt the DataFrame to long format
                melted_data = pd.melt(data, value_vars=numeric_cols)
                sns.ecdfplot(data=melted_data, x='value', hue='variable', ax=ax, **kwargs)
            else:
                # Create ECDF plot for a single column
                sns.ecdfplot(data=data, x=column, ax=ax, **kwargs)
        
        # Set labels and title
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel('Cumulative Probability', fontsize=12)
            
        return ax
    
    # Bivariate Analysis
    @staticmethod
    def plot_scatter(data: pd.DataFrame, 
                    x: str, 
                    y: str,
                    hue: Optional[str] = None,
                    size: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    add_reg_line: bool = True,
                    add_lowess: bool = False,
                    ax: Optional[plt.Axes] = None,
                    **kwargs) -> plt.Axes:
        """
        Create a scatter plot.
        
        Args:
            data (pd.DataFrame): The data to plot.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            hue (Optional[str], optional): Column name for color encoding. Defaults to None.
            size (Optional[str], optional): Column name for size encoding. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            xlabel (Optional[str], optional): X-axis label. Defaults to None.
            ylabel (Optional[str], optional): Y-axis label. Defaults to None.
            add_reg_line (bool, optional): Whether to add regression line. Defaults to True.
            add_lowess (bool, optional): Whether to add LOWESS line. Defaults to False.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            **kwargs: Additional arguments to pass to seaborn's scatterplot.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Create scatter plot
        scatter_kws = {k: v for k, v in kwargs.items() if k != 'lowess' and k != 'line_kws'}
        sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, ax=ax, **scatter_kws)
        
        # Add regression line
        if add_reg_line:
            line_kws = kwargs.get('line_kws', {})
            sns.regplot(data=data, x=x, y=y, scatter=False, ax=ax, lowess=add_lowess, 
                       line_kws=line_kws)
            
            # Add correlation coefficient
            if hue is None:  # Only calculate correlation if not grouped
                corr_coef = data[[x, y]].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set labels and title
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        else:
            ax.set_xlabel(x, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel(y, fontsize=12)
            
        # Adjust legend if both hue and size are specified
        if hue is not None and size is not None:
            handles, labels = ax.get_legend_handles_labels()
            n_hue = len(data[hue].unique())
            plt.legend(handles[:n_hue], labels[:n_hue], title=hue)
            
        return ax
    
    @staticmethod
    def plot_correlation_heatmap(data: pd.DataFrame,
                                method: str = 'pearson',
                                annot: bool = True,
                                cmap: str = 'coolwarm',
                                vmin: float = -1.0,
                                vmax: float = 1.0,
                                figsize: Tuple[int, int] = (10, 8),
                                title: Optional[str] = None,
                                mask_upper: bool = False,
                                ax: Optional[plt.Axes] = None,
                                **kwargs) -> plt.Axes:
        """
        Create a correlation heatmap.
        
        Args:
            data (pd.DataFrame): The data to plot (numeric columns only).
            method (str, optional): Correlation method ('pearson', 'spearman', or 'kendall'). 
                                    Defaults to 'pearson'.
            annot (bool, optional): Whether to annotate heatmap with correlation values. 
                                    Defaults to True.
            cmap (str, optional): Colormap. Defaults to 'coolwarm'.
            vmin (float, optional): Minimum value for colormap. Defaults to -1.0.
            vmax (float, optional): Maximum value for colormap. Defaults to 1.0.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
            title (Optional[str], optional): Plot title. Defaults to None.
            mask_upper (bool, optional): Whether to mask the upper triangle. Defaults to False.
            ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
            **kwargs: Additional arguments to pass to seaborn's heatmap.
            
        Returns:
            plt.Axes: The matplotlib axes containing the plot.
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            warnings.warn("No numeric columns found for correlation heatmap.")
            return None
            
        # Compute correlation matrix
        corr_matrix = numeric_data.corr(method=method)
        
        # Create mask for upper triangle if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix), k=1)
            
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, 
                   ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, **kwargs)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, pad=20)
            
        # Rotate y-axis tick labels
        plt.yticks(rotation=0)
        
        return ax
    
    @staticmethod
    def plot_joint(data: pd.DataFrame,
                  x: str,
                  y: str,
                  kind: str = 'scatter',
                  hue: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 10),
                  title: Optional[str] = None,
                  **kwargs) -> sns.JointGrid:
        """
        Create a joint plot (scatter plot with marginal distributions).
        
        Args:
            data (pd.DataFrame): The data to plot.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            kind (str, optional): Kind of plot ('scatter', 'reg', 'resid', 'kde', 'hex'). 
                                  Defaults to 'scatter'.
            hue (Optional[str], optional): Column name for color encoding. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 10).
            title (Optional[str], optional): Plot title. Defaults to None.
            **kwargs: Additional arguments to pass to seaborn's jointplot.
            
        Returns:
            sns.JointGrid: The seaborn JointGrid object containing the plot.
        """
        # Set figure size
        plt.figure(figsize=figsize)
        
        # Create joint plot
        g = sns.jointplot(data=data, x=x, y=y, kind=kind, hue=hue, **kwargs)
        
        # Set title
        if title:
            g.fig.suptitle(title, fontsize=14)
            g.fig.tight_layout()
            g.fig.subplots_adjust(top=0.95)  # Adjust top spacing for title
            
        return g
    
    @staticmethod
    def plot_pairwise_correlation(data: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 method: str = 'pearson',
                                 figsize: Tuple[int, int] = (12, 10),
                                 title: Optional[str] = None,
                                 **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a matrix of pairwise correlation plots.
        
        Args:
            data (pd.DataFrame): The data to plot.
            columns (Optional[List[str]], optional): List of columns to include. 
                                                    If None, use all numeric columns. Defaults to None.
            method (str, optional): Correlation method ('pearson', 'spearman', or 'kendall'). 
                                    Defaults to 'pearson'.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 10).
            title (Optional[str], optional): Plot title. Defaults to None.
            **kwargs: Additional arguments to pass to matplotlib's scatter.
            
        Returns:
            Tuple[plt.Figure, np.ndarray]: Figure and axes array.
        """
        # Select columns
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
        else:
            # Verify all columns exist and are numeric
            for col in columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                if not np.issubdtype(data[col].dtype, np.number):
                    warnings.warn(f"Column '{col}' is not numeric. Correlation may not be meaningful.")
                    
        n_cols = len(columns)
        if n_cols == 0:
            warnings.warn("No numeric columns found for pairwise correlation.")
            return None, None
            
        # Create figure and axes
        fig, axes = plt.subplots(n_cols, n_cols, figsize=figsize)
        
        # Make sure axes is a 2D array even with a single column
        if n_cols == 1:
            axes = np.array([[axes]])
            
        # Compute correlation matrix
        corr_matrix = data[columns].corr(method=method)
        
        # Create the plots
        for i in range(n_cols):
            for j in range(n_cols):
                ax = axes[i, j]
                
                # Clear axis if it's not on the diagonal and not in lower triangle
                if i < j:
                    ax.axis('off')
                    continue
                    
                # Diagonal: Show column name and histogram
                if i == j:
                    ax.text(0.5, 0.5, columns[i], horizontalalignment='center',
                           verticalalignment='center', fontsize=12, fontweight='bold',
                           transform=ax.transAxes)
                    ax.axis('off')
                else:
                    # Lower triangle: scatter plot
                    ax.scatter(data[columns[j]], data[columns[i]], alpha=0.6, **kwargs)
                    
                    # Add correlation coefficient
                    corr_val = corr_matrix.iloc[i, j]
                    ax.text(0.05, 0.95, f'{corr_val:.2f}', transform=ax.transAxes,
                           horizontalalignment='left', verticalalignment='top',
                           fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Only add labels on the edge axes
                    if i == n_cols - 1:
                        ax.set_xlabel(columns[j])
                    else:
                        ax.set_xticklabels([])
                        
                    if j == 0:
                        ax.set_ylabel(columns[i])
                    else:
                        ax.set_yticklabels([])
                        
        # Adjust layout
        plt.tight_layout()
        
        # Add title
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
            
        return fig, axes
    
    # Multivariate Analysis
    @staticmethod
    def plot_pairplot(data: pd.DataFrame,
                     columns: Optional[List[str]] = None,
                     hue: Optional[str] = None,
                     kind: str = 'scatter',
                     diag_kind: str = 'auto',
                     corner: bool = False,
                     markers: Optional[List[str]] = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     **kwargs) -> sns.PairGrid:
