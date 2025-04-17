"""
Statistical analysis utilities for exploratory data analysis.

This module provides functions for statistical analysis of datasets, including:
- Summary statistics
- Distribution analysis
- Correlation analysis
- Hypothesis testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Any
import warnings


class StatisticalAnalyzer:
    """Class for performing statistical analysis on datasets."""
    
    @staticmethod
    def generate_summary_statistics(data: pd.DataFrame, 
                                   include_non_numeric: bool = False) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for a DataFrame.
        
        Args:
            data (pd.DataFrame): The dataset to analyze.
            include_non_numeric (bool, optional): Whether to include statistics for non-numeric columns. 
                                                  Defaults to False.
                
        Returns:
            pd.DataFrame: DataFrame containing summary statistics.
        """
        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            warnings.warn("No numeric columns found in the dataset.")
            
        # Generate summary statistics for numeric columns
        if numeric_cols:
            numeric_stats = pd.DataFrame({
                'count': data[numeric_cols].count(),
                'missing': data[numeric_cols].isna().sum(),
                'missing_pct': (data[numeric_cols].isna().sum() / len(data) * 100).round(2),
                'unique': data[numeric_cols].nunique(),
                'mean': data[numeric_cols].mean(),
                'median': data[numeric_cols].median(),
                'std': data[numeric_cols].std(),
                'min': data[numeric_cols].min(),
                '25%': data[numeric_cols].quantile(0.25),
                '50%': data[numeric_cols].quantile(0.5),
                '75%': data[numeric_cols].quantile(0.75),
                'max': data[numeric_cols].max(),
                'skew': data[numeric_cols].skew(),
                'kurtosis': data[numeric_cols].kurtosis(),
                'iqr': data[numeric_cols].quantile(0.75) - data[numeric_cols].quantile(0.25),
                'range': data[numeric_cols].max() - data[numeric_cols].min(),
                'cv': (data[numeric_cols].std() / data[numeric_cols].mean()).abs().replace([np.inf, -np.inf], np.nan)
            }).T
        else:
            numeric_stats = pd.DataFrame()
            
        # Process non-numeric columns if requested
        if include_non_numeric:
            non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
            
            if non_numeric_cols:
                # Generate summary statistics for non-numeric columns
                non_numeric_stats = pd.DataFrame({
                    'count': data[non_numeric_cols].count(),
                    'missing': data[non_numeric_cols].isna().sum(),
                    'missing_pct': (data[non_numeric_cols].isna().sum() / len(data) * 100).round(2),
                    'unique': data[non_numeric_cols].nunique(),
                    'mode': [data[col].mode()[0] if not data[col].mode().empty else None for col in non_numeric_cols],
                    'mode_freq': [data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0 for col in non_numeric_cols],
                    'mode_pct': [(data[col].value_counts().iloc[0] / data[col].count() * 100).round(2) 
                                if len(data[col].value_counts()) > 0 else 0 for col in non_numeric_cols],
                    'unique_ratio': (data[non_numeric_cols].nunique() / data[non_numeric_cols].count()).round(4)
                }).T
                
                # Combine numeric and non-numeric statistics
                if not numeric_stats.empty:
                    # Fill in missing values for columns that don't apply to both types
                    combined_stats = pd.concat([numeric_stats, non_numeric_stats], axis=1)
                else:
                    combined_stats = non_numeric_stats
            else:
                combined_stats = numeric_stats
        else:
            combined_stats = numeric_stats
            
        return combined_stats
    
    @staticmethod
    def test_normality(data: Union[pd.Series, pd.DataFrame],
                      test_method: str = 'shapiro',
                      alpha: float = 0.05) -> pd.DataFrame:
        """
        Test the normality of data using various statistical tests.
        
        Args:
            data (Union[pd.Series, pd.DataFrame]): The data to test.
            test_method (str, optional): The test method to use. Options are 'shapiro', 'ks' (Kolmogorov-Smirnov),
                                         'anderson', or 'all'. Defaults to 'shapiro'.
            alpha (float, optional): The significance level. Defaults to 0.05.
            
        Returns:
            pd.DataFrame: DataFrame with test results.
            
        Raises:
            ValueError: If test_method is not one of the supported options.
        """
        valid_methods = ['shapiro', 'ks', 'anderson', 'all']
        if test_method not in valid_methods:
            raise ValueError(f"test_method must be one of {valid_methods}")
            
        # Convert pd.Series to pd.DataFrame for uniform processing
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
            
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            warnings.warn("No numeric columns found for normality testing.")
            return pd.DataFrame()
            
        results = []
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna().values
            
            # Skip if too few data points
            if len(col_data) < 3:
                continue
                
            result = {'column': col}
            
            # Perform requested test(s)
            if test_method in ['shapiro', 'all']:
                # Shapiro-Wilk test (best for n < 5000)
                if len(col_data) < 5000:
                    stat, p_value = stats.shapiro(col_data)
                    result['shapiro_stat'] = stat
                    result['shapiro_p_value'] = p_value
                    result['shapiro_normal'] = p_value > alpha
                else:
                    result['shapiro_stat'] = None
                    result['shapiro_p_value'] = None
                    result['shapiro_normal'] = None
                    
            if test_method in ['ks', 'all']:
                # Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(col_data, 'norm', args=(np.mean(col_data), np.std(col_data, ddof=1)))
                result['ks_stat'] = stat
                result['ks_p_value'] = p_value
                result['ks_normal'] = p_value > alpha
                
            if test_method in ['anderson', 'all']:
                # Anderson-Darling test
                ad_result = stats.anderson(col_data, dist='norm')
                result['anderson_stat'] = ad_result.statistic
                
                # Compare statistic against critical values
                critical_values = ad_result.critical_values
                significance_levels = [15, 10, 5, 2.5, 1]
                
                # Find the highest significance level where we can reject the null hypothesis
                for sig_level, critical_value in zip(significance_levels, critical_values):
                    if ad_result.statistic < critical_value:
                        result['anderson_sig_level'] = sig_level / 100
                        result['anderson_normal'] = True
                        break
                else:
                    result['anderson_sig_level'] = significance_levels[-1] / 100
                    result['anderson_normal'] = False
                    
            # Add descriptive statistics
            result['skew'] = stats.skew(col_data)
            result['kurtosis'] = stats.kurtosis(col_data)
            
            # Overall normality assessment (if 'all' tests were performed)
            if test_method == 'all':
                tests = []
                if 'shapiro_normal' in result and result['shapiro_normal'] is not None:
                    tests.append(result['shapiro_normal'])
                if 'ks_normal' in result:
                    tests.append(result['ks_normal'])
                if 'anderson_normal' in result:
                    tests.append(result['anderson_normal'])
                    
                # If majority of tests suggest normality, consider it normal
                if tests:
                    result['is_normal'] = sum(tests) > len(tests) / 2
                else:
                    result['is_normal'] = None
            else:
                # For single test, use that test's result
                normal_key = f"{test_method}_normal"
                if normal_key in result:
                    result['is_normal'] = result[normal_key]
                    
            results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def correlation_analysis(data: pd.DataFrame,
                           method: str = 'pearson',
                           threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Perform correlation analysis on a DataFrame.
        
        Args:
            data (pd.DataFrame): The dataset to analyze.
            method (str, optional): Correlation method to use ('pearson', 'spearman', or 'kendall'). 
                                    Defaults to 'pearson'.
            threshold (Optional[float], optional): If provided, only return correlations with absolute 
                                                  value >= threshold. Defaults to None.
            
        Returns:
            pd.DataFrame: Correlation matrix.
            
        Raises:
            ValueError: If method is not one of 'pearson', 'spearman', or 'kendall'.
        """
        valid_methods = ['pearson', 'spearman', 'kendall']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
            
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            warnings.warn("No numeric columns found for correlation analysis.")
            return pd.DataFrame()
            
        # Compute correlation matrix
        corr_matrix = numeric_data.corr(method=method)
        
        # Apply threshold if specified
        if threshold is not None:
            # Create a mask for correlations below threshold (in absolute value)
            mask = np.abs(corr_matrix) < threshold
            # Replace values below threshold with NaN
            corr_matrix = corr_matrix.mask(mask)
            
        return corr_matrix
    
    @staticmethod
    def categorical_correlation(data: pd.DataFrame, 
                               categorical_cols: Optional[List[str]] = None,
                               numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze correlations between categorical and numeric variables using ANOVA.
        
        Args:
            data (pd.DataFrame): The dataset to analyze.
            categorical_cols (Optional[List[str]], optional): List of categorical column names. 
                                                             If None, all non-numeric columns will be used.
                                                             Defaults to None.
            numeric_cols (Optional[List[str]], optional): List of numeric column names.
                                                         If None, all numeric columns will be used.
                                                         Defaults to None.
            
        Returns:
            pd.DataFrame: DataFrame with correlation statistics (F-value, p-value, and eta-squared).
        """
        # Identify categorical and numeric columns if not provided
        if categorical_cols is None:
            categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
        if numeric_cols is None:
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
        if not categorical_cols or not numeric_cols:
            warnings.warn("No categorical or numeric columns found for analysis.")
            return pd.DataFrame()
            
        results = []
        
        for cat_col in categorical_cols:
            cat_groups = data.groupby(cat_col)
            
            for num_col in numeric_cols:
                # Skip if the numeric column has no variance
                if data[num_col].nunique() <= 1:
                    continue
                    
                # Collect data for each category
                anova_data = []
                for category, group in cat_groups:
                    values = group[num_col].dropna().values
                    if len(values) > 0:
                        anova_data.append(values)
                        
                # Skip if less than 2 groups have data
                if len(anova_data) < 2:
                    continue
                    
                # Perform one-way ANOVA
                try:
                    f_stat, p_value = stats.f_oneway(*anova_data)
                    
                    # Calculate eta-squared (effect size)
                    # Formula: SSB / SST where SSB is between-group sum of squares and SST is total sum of squares
                    grand_mean = data[num_col].mean()
                    ssb = sum(len(group) * ((group[num_col].mean() - grand_mean) ** 2) 
                             for _, group in cat_groups if len(group) > 0)
                    sst = sum((data[num_col] - grand_mean) ** 2)
                    eta_squared = ssb / sst if sst != 0 else np.nan
                    
                    results.append({
                        'categorical': cat_col,
                        'numeric': num_col,
                        'f_value': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'eta_squared': eta_squared
                    })
                except Exception as e:
                    # Skip in case of errors (e.g., constant values in a group)
                    continue
                    
        return pd.DataFrame(results)
    
    @staticmethod
    def chi_square_test(data: pd.DataFrame,
                       categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform chi-square test of independence between categorical variables.
        
        Args:
            data (pd.DataFrame): The dataset to analyze.
            categorical_cols (Optional[List[str]], optional): List of categorical column names.
                                                             If None, all non-numeric columns will be used.
                                                             Defaults to None.
            
        Returns:
            pd.DataFrame: DataFrame with chi-square test results.
        """
        # Identify categorical columns if not provided
        if categorical_cols is None:
            categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
            
        if len(categorical_cols) < 2:
            warnings.warn("At least two categorical columns are needed for chi-square test.")
            return pd.DataFrame()
            
        results = []
        
        # Analyze each pair of categorical variables
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                # Create contingency table
                contingency = pd.crosstab(data[col1], data[col2])
                
                # Skip if contingency table has any dimension of size 1 (no variation)
                if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
                    continue
                    
                # Perform chi-square test
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    
                    # Calculate Cramer's V (effect size)
                    # Formula: sqrt(chi2 / (n * min(r-1, c-1)))
                    n = contingency.sum().sum()
                    r, c = contingency.shape
                    cramers_v = np.sqrt(chi2 / (n * min(r-1, c-1))) if n > 0 and min(r-1, c-1) > 0 else np.nan
                    
                    results.append({
                        'variable1': col1,
                        'variable2': col2,
                        'chi2': chi2,
                        'p_value': p_value,
                        'dof': dof,
                        'significant': p_value < 0.05,
                        'cramers_v': cramers_v
                    })
                except Exception as e:
                    # Skip in case of errors (e.g., expected frequencies less than 5)
                    continue
                    
        return pd.DataFrame(results)
    
    @staticmethod
    def outlier_detection(data: pd.DataFrame,
                         method: str = 'iqr',
                         threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """
        Detect outliers in numeric data using various methods.
        
        Args:
            data (pd.DataFrame): The dataset to analyze.
            method (str, optional): Method to use for outlier detection.
                                   Options are 'iqr' (Interquartile Range), 'zscore', or 'modified_zscore'.
                                   Defaults to 'iqr'.
            threshold (float, optional): Threshold for outlier detection.
                                        For IQR: multiplier for IQR, default 1.5.
                                        For zscore: number of standard deviations, default 3.
                                        For modified_zscore: threshold for modified z-score, default 3.5.
                                        Defaults to 1.5.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping column names to boolean arrays indicating outliers.
            
        Raises:
            ValueError: If method is not one of the supported options.
        """
        valid_methods = ['iqr', 'zscore', 'modified_zscore']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
            
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            warnings.warn("No numeric columns found for outlier detection.")
            return {}
            
        outliers = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            
            if method == 'iqr':
                # IQR method
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers[col] = ((col_data < lower_bound) | (col_data > upper_bound)).values
                
            elif method == 'zscore':
                # Z-score method
                mean = col_data.mean()
                std = col_data.std()
                
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((col_data - mean) / std)
                    outliers[col] = (z_scores > threshold).values
                else:
                    outliers[col] = np.zeros(len(col_data), dtype=bool)
                    
            elif method == 'modified_zscore':
                # Modified Z-score method (more robust to outliers)
                median = col_data.median()
                mad = stats.median_abs_deviation(col_data, scale=1)
                
                if mad > 0:  # Avoid division by zero
                    modified_z_scores = 0.6745 * np.abs(col_data - median) / mad
                    outliers[col] = (modified_z_scores > threshold).values
                else:
                    outliers[col] = np.zeros(len(col_data), dtype=bool)
                    
        return outliers
    
    @staticmethod
    def distribution_analysis(data: pd.Series) -> Dict[str, Any]:
        """
        Analyze the distribution of a numeric variable and fit potential distributions.
        
        Args:
            data (pd.Series): The data to analyze (must be numeric).
            
        Returns:
            Dict[str, Any]: Dictionary with distribution analysis results.
        """
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError("Data must be numeric for distribution analysis.")
            
        # Clean data for analysis
        clean_data = data.dropna()
        
        if len(clean_data) < 3:
            warnings.warn("Too few data points for distribution analysis.")
            return {}
            
        results = {
            'n': len(clean_data),
            'mean': clean_data.mean(),
            'median': clean_data.median(),
            'std': clean_data.std(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'skew': stats.skew(clean_data),
            'kurtosis': stats.kurtosis(clean_data)
        }
        
        # Distribution fitting
        distributions_to_try = [
            stats.norm,         # Normal (Gaussian)
            stats.lognorm,      # Log-normal
            stats.expon,        # Exponential
            stats.gamma,        # Gamma
            stats.beta,         # Beta
            stats.uniform,      # Uniform
            stats.weibull_min   # Weibull
        ]
        
        # Initialize fit results
        fit_results = []
        
        for dist in distributions_to_try:
            try:
                # Fit distribution to data
                params = dist.fit(clean_data)
                
                # Calculate goodness of fit using Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(clean_data, dist.name, args=params)
                
                # Store results
                fit_results.append({
                    'distribution': dist.name,
                    'params': params,
                    'ks_statistic': stat,
                    'p_value': p_value
                })
            except Exception:
                # Skip if fit fails
                continue
        
        # Sort results by p-value (higher is better fit)
        fit_results = sorted(fit_results, key=lambda x: x['p_value'], reverse=True)
        
        # Add fit results to output
        results['fitted_distributions'] = fit_results
        
        return results


def generate_summary_statistics(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function for generating summary statistics.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        **kwargs: Additional arguments for StatisticalAnalyzer.generate_summary_statistics().
        
    Returns:
        pd.DataFrame: DataFrame containing summary statistics.
    """
    return StatisticalAnalyzer.generate_summary_statistics(data, **kwargs)


def test_normality(data: Union[pd.Series, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """
    Convenience function for testing normality.
    
    Args:
        data (Union[pd.Series, pd.DataFrame]): The data to test.
        **kwargs: Additional arguments for StatisticalAnalyzer.test_normality().
        
    Returns:
        pd.DataFrame: DataFrame with test results.
    """
    return StatisticalAnalyzer.test_normality(data, **kwargs)


def correlation_analysis(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function for correlation analysis.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        **kwargs: Additional arguments for StatisticalAnalyzer.correlation_analysis().
        
    Returns:
        pd.DataFrame: Correlation matrix.
    """
    return StatisticalAnalyzer.correlation_analysis(data, **kwargs)


def categorical_correlation(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function for categorical correlation analysis.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        **kwargs: Additional arguments for StatisticalAnalyzer.categorical_correlation().
        
    Returns:
        pd.DataFrame: DataFrame with correlation statistics.
    """
    return StatisticalAnalyzer.categorical_correlation(data, **kwargs)


def chi_square_test(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function for chi-square test of independence.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        **kwargs: Additional arguments for StatisticalAnalyzer.chi_square_test().
        
    Returns:
        pd.DataFrame: DataFrame with chi-square test results.
    """
    return StatisticalAnalyzer.chi_square_test(data, **kwargs)


def outlier_detection(data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
    """
    Convenience function for outlier detection.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        **kwargs: Additional arguments for StatisticalAnalyzer.outlier_detection().
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping column names to boolean arrays indicating outliers.
    """
    return StatisticalAnalyzer.outlier_detection(data, **kwargs)


def distribution_analysis(data: pd.Series) -> Dict[str, Any]:
    """
    Convenience function for distribution analysis.
    
    Args:
        data (pd.Series): The data to analyze (must be numeric).
        
    Returns:
        Dict[str, Any]: Dictionary with distribution analysis results.
    """
    return StatisticalAnalyzer.distribution_analysis(data)
