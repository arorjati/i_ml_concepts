"""
Feature engineering utilities.

This module provides functions and classes for feature engineering, including:
- Numerical feature transformations (scaling, normalization)
- Categorical feature encoding
- Feature creation and derivation
- Feature selection and dimensionality reduction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, 
    QuantileTransformer, PowerTransformer, OneHotEncoder, LabelEncoder,
    OrdinalEncoder, KBinsDiscretizer
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    VarianceThreshold, RFE, RFECV, f_classif, mutual_info_classif,
    f_regression, mutual_info_regression
)
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.decomposition import PCA, TruncatedSVD, NMF
import warnings


class NumericalFeatureTransformer:
    """Class for transforming numerical features."""
    
    @staticmethod
    def scale_standard(data: Union[np.ndarray, pd.DataFrame],
                      with_mean: bool = True,
                      with_std: bool = True,
                      copy: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], StandardScaler]:
        """
        Standardize features by removing the mean and scaling to unit variance.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): The data to transform.
            with_mean (bool, optional): If True, center the data before scaling. Defaults to True.
            with_std (bool, optional): If True, scale the data to unit variance. Defaults to True.
            copy (bool, optional): If True, create a copy of data. Defaults to True.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], StandardScaler]: 
                Scaled data and the fitted scaler.
        """
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std, copy=copy)
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save column names and index
            columns = data.columns
            index = data.index
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data)
            
            # Convert back to DataFrame with original metadata
            return pd.DataFrame(scaled_data, columns=columns, index=index), scaler
        else:
            # Handle numpy arrays
            return scaler.fit_transform(data), scaler
    
    @staticmethod
    def scale_minmax(data: Union[np.ndarray, pd.DataFrame],
                    feature_range: Tuple[float, float] = (0, 1),
                    copy: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], MinMaxScaler]:
        """
        Transform features by scaling each feature to a given range.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): The data to transform.
            feature_range (Tuple[float, float], optional): Range of transformed data. Defaults to (0, 1).
            copy (bool, optional): If True, create a copy of data. Defaults to True.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], MinMaxScaler]: 
                Scaled data and the fitted scaler.
        """
        scaler = MinMaxScaler(feature_range=feature_range, copy=copy)
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save column names and index
            columns = data.columns
            index = data.index
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data)
            
            # Convert back to DataFrame with original metadata
            return pd.DataFrame(scaled_data, columns=columns, index=index), scaler
        else:
            # Handle numpy arrays
            return scaler.fit_transform(data), scaler
    
    @staticmethod
    def scale_robust(data: Union[np.ndarray, pd.DataFrame],
                    with_centering: bool = True,
                    with_scaling: bool = True,
                    quantile_range: Tuple[float, float] = (25.0, 75.0),
                    copy: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], RobustScaler]:
        """
        Scale features using statistics that are robust to outliers.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): The data to transform.
            with_centering (bool, optional): If True, center the data before scaling. Defaults to True.
            with_scaling (bool, optional): If True, scale the data. Defaults to True.
            quantile_range (Tuple[float, float], optional): Quantile range for calculating IQR. 
                                                          Defaults to (25.0, 75.0).
            copy (bool, optional): If True, create a copy of data. Defaults to True.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], RobustScaler]: 
                Scaled data and the fitted scaler.
        """
        scaler = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy
        )
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save column names and index
            columns = data.columns
            index = data.index
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data)
            
            # Convert back to DataFrame with original metadata
            return pd.DataFrame(scaled_data, columns=columns, index=index), scaler
        else:
            # Handle numpy arrays
            return scaler.fit_transform(data), scaler
    
    @staticmethod
    def transform_power(data: Union[np.ndarray, pd.DataFrame],
                       method: str = 'yeo-johnson',
                       standardize: bool = True,
                       copy: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], PowerTransformer]:
        """
        Apply a power transform to make data more Gaussian-like.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): The data to transform.
            method (str, optional): The power transform method. 
                                    'yeo-johnson' works with positive and negative values,
                                    'box-cox' requires positive values.
                                    Defaults to 'yeo-johnson'.
            standardize (bool, optional): If True, scale the data to zero mean and unit variance.
                                         Defaults to True.
            copy (bool, optional): If True, create a copy of data. Defaults to True.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], PowerTransformer]: 
                Transformed data and the fitted transformer.
        """
        transformer = PowerTransformer(method=method, standardize=standardize, copy=copy)
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save column names and index
            columns = data.columns
            index = data.index
            
            # Fit and transform
            transformed_data = transformer.fit_transform(data)
            
            # Convert back to DataFrame with original metadata
            return pd.DataFrame(transformed_data, columns=columns, index=index), transformer
        else:
            # Handle numpy arrays
            return transformer.fit_transform(data), transformer
    
    @staticmethod
    def transform_quantile(data: Union[np.ndarray, pd.DataFrame],
                          n_quantiles: int = 1000,
                          output_distribution: str = 'normal',
                          random_state: Optional[int] = None,
                          copy: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], QuantileTransformer]:
        """
        Transform features to follow a specified distribution.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): The data to transform.
            n_quantiles (int, optional): Number of quantiles to use. Defaults to 1000.
            output_distribution (str, optional): Marginal distribution for the transformed data.
                                              Options: 'normal' or 'uniform'. Defaults to 'normal'.
            random_state (Optional[int], optional): Random seed. Defaults to None.
            copy (bool, optional): If True, create a copy of data. Defaults to True.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], QuantileTransformer]: 
                Transformed data and the fitted transformer.
        """
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=random_state,
            copy=copy
        )
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save column names and index
            columns = data.columns
            index = data.index
            
            # Fit and transform
            transformed_data = transformer.fit_transform(data)
            
            # Convert back to DataFrame with original metadata
            return pd.DataFrame(transformed_data, columns=columns, index=index), transformer
        else:
            # Handle numpy arrays
            return transformer.fit_transform(data), transformer
    
    @staticmethod
    def log_transform(data: Union[np.ndarray, pd.DataFrame],
                     constant: float = 1.0) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply a logarithmic transformation to the data.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): The data to transform.
            constant (float, optional): Constant added to data before taking logarithm.
                                       Use to handle zero or negative values. Defaults to 1.0.
            
        Returns:
            Union[np.ndarray, pd.DataFrame]: Log-transformed data.
        """
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Check for negative or zero values if constant is zero
            if constant == 0 and (data <= 0).any().any():
                warnings.warn(
                    "Data contains zero or negative values. Consider using a positive constant."
                )
            
            return np.log(data + constant)
        else:
            # Check for negative or zero values if constant is zero
            if constant == 0 and np.any(data <= 0):
                warnings.warn(
                    "Data contains zero or negative values. Consider using a positive constant."
                )
                
            return np.log(data + constant)
    
    @staticmethod
    def binning(data: Union[np.ndarray, pd.DataFrame, pd.Series],
               n_bins: int = 5,
               strategy: str = 'quantile',
               encode: str = 'onehot',
               labels: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Bin continuous data into discrete intervals.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data to transform.
            n_bins (int, optional): Number of bins. Defaults to 5.
            strategy (str, optional): Binning strategy. Options: 'uniform', 'quantile', 'kmeans'.
                                     Defaults to 'quantile'.
            encode (str, optional): Method to encode the bins. Options: 'onehot', 'ordinal'.
                                   Defaults to 'onehot'.
            labels (Optional[List[str]], optional): Custom labels for the bins. Defaults to None.
            
        Returns:
            Union[np.ndarray, pd.DataFrame]: Binned data.
        """
        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save column names and index
            columns = data.columns
            index = data.index
            
            # Initialize discretizer
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode=encode,
                strategy=strategy
            )
            
            # Fit and transform
            binned_data = discretizer.fit_transform(data)
            
            # Process output based on encoding method
            if encode == 'onehot':
                # For one-hot encoding, get the feature names from the transformer
                feature_names = []
                if hasattr(discretizer, 'get_feature_names_out'):
                    feature_names = discretizer.get_feature_names_out(columns)
                else:
                    # Fallback for older scikit-learn versions
                    for i, col in enumerate(columns):
                        for j in range(n_bins):
                            feature_names.append(f"{col}_{j}")
                
                # Convert sparse matrix to dense if needed
                if hasattr(binned_data, "toarray"):
                    binned_data = binned_data.toarray()
                    
                return pd.DataFrame(binned_data, columns=feature_names, index=index)
            else:
                # For ordinal encoding, use original column names
                return pd.DataFrame(binned_data, columns=columns, index=index)
        else:
            # Initialize discretizer
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode=encode,
                strategy=strategy
            )
            
            # Handle 1D arrays
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            # Fit and transform
            binned_data = discretizer.fit_transform(data)
            
            # Convert sparse matrix to dense if needed
            if hasattr(binned_data, "toarray"):
                binned_data = binned_data.toarray()
                
            return binned_data


class CategoricalFeatureEncoder:
    """Class for encoding categorical features."""
    
    @staticmethod
    def one_hot_encode(data: Union[np.ndarray, pd.DataFrame, pd.Series],
                      columns: Optional[List[str]] = None,
                      drop: str = 'first',
                      sparse: bool = False,
                      handle_unknown: str = 'error') -> Tuple[Union[np.ndarray, pd.DataFrame], OneHotEncoder]:
        """
        Encode categorical features as a one-hot numeric array.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data to encode.
            columns (Optional[List[str]], optional): Column names to encode. If None and data is DataFrame,
                                                   all object and category columns will be encoded.
                                                   Defaults to None.
            drop (str, optional): Whether to drop one of the categories. Options: None, 'first', 'if_binary'.
                                 Defaults to 'first'.
            sparse (bool, optional): Whether to return sparse matrix. Defaults to False.
            handle_unknown (str, optional): How to handle unknown categories. Options: 'error', 'ignore'.
                                           Defaults to 'error'.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], OneHotEncoder]: 
                Encoded data and the fitted encoder.
        """
        encoder = OneHotEncoder(drop=drop, sparse=sparse, handle_unknown=handle_unknown)
        
        # Handle pandas Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
            columns = [data.columns[0]]
            
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # If columns not specified, use all object and category dtypes
            if columns is None:
                columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not columns:
                    warnings.warn("No object or category columns found for one-hot encoding.")
                    return data, encoder
                    
            # Get data subset with only the columns to encode
            categorical_data = data[columns]
            
            # Fit encoder
            encoder.fit(categorical_data)
            
            # Transform data
            encoded_data = encoder.transform(categorical_data)
            
            # Convert sparse matrix to dense if requested
            if not sparse and hasattr(encoded_data, "toarray"):
                encoded_data = encoded_data.toarray()
                
            # Get feature names
            feature_names = []
            if hasattr(encoder, 'get_feature_names_out'):
                feature_names = encoder.get_feature_names_out(columns)
            else:
                # Fallback for older scikit-learn versions
                categories = encoder.categories_
                for i, col in enumerate(columns):
                    for j, cat in enumerate(categories[i]):
                        if drop == 'first' and j == 0:
                            continue
                        feature_names.append(f"{col}_{cat}")
                        
            # Create DataFrame with encoded data
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data.index)
            
            # Drop original categorical columns and join encoded columns
            result = data.drop(columns=columns).join(encoded_df)
            
            return result, encoder
        else:
            # Handle numpy arrays
            # Reshape if 1D
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            # Fit and transform
            encoded_data = encoder.fit_transform(data)
            
            # Convert sparse matrix to dense if requested
            if not sparse and hasattr(encoded_data, "toarray"):
                encoded_data = encoded_data.toarray()
                
            return encoded_data, encoder
    
    @staticmethod
    def label_encode(data: Union[np.ndarray, pd.DataFrame, pd.Series],
                    columns: Optional[List[str]] = None) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, LabelEncoder]]:
        """
        Encode target labels with value between 0 and n_classes-1.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data to encode.
            columns (Optional[List[str]], optional): Column names to encode. If None and data is DataFrame,
                                                   all object and category columns will be encoded.
                                                   Defaults to None.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, LabelEncoder]]: 
                Encoded data and a dictionary of fitted encoders.
        """
        # Handle pandas Series
        if isinstance(data, pd.Series):
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data)
            return encoded_data, {data.name if data.name else '0': encoder}
            
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # If columns not specified, use all object and category dtypes
            if columns is None:
                columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not columns:
                    warnings.warn("No object or category columns found for label encoding.")
                    return data, {}
                    
            # Create a copy of data to avoid modifying the original
            result = data.copy()
            encoders = {}
            
            # Encode each column
            for col in columns:
                encoder = LabelEncoder()
                result[col] = encoder.fit_transform(result[col])
                encoders[col] = encoder
                
            return result, encoders
        else:
            # Handle numpy arrays
            # For 1D arrays
            if data.ndim == 1:
                encoder = LabelEncoder()
                encoded_data = encoder.fit_transform(data)
                return encoded_data, {'0': encoder}
            else:
                # For 2D arrays, encode each column separately
                encoders = {}
                encoded_data = np.zeros_like(data)
                
                for i in range(data.shape[1]):
                    encoder = LabelEncoder()
                    encoded_data[:, i] = encoder.fit_transform(data[:, i])
                    encoders[str(i)] = encoder
                    
                return encoded_data, encoders
    
    @staticmethod
    def ordinal_encode(data: Union[np.ndarray, pd.DataFrame, pd.Series],
                      columns: Optional[List[str]] = None,
                      categories: Optional[List[List]] = None,
                      handle_unknown: str = 'error') -> Tuple[Union[np.ndarray, pd.DataFrame], OrdinalEncoder]:
        """
        Encode categorical features as an integer array.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data to encode.
            columns (Optional[List[str]], optional): Column names to encode. If None and data is DataFrame,
                                                   all object and category columns will be encoded.
                                                   Defaults to None.
            categories (Optional[List[List]], optional): List of categories for each feature. Defaults to None.
            handle_unknown (str, optional): How to handle unknown categories. Options: 'error', 'use_encoded_value'.
                                           Defaults to 'error'.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], OrdinalEncoder]: 
                Encoded data and the fitted encoder.
        """
        # Create encoder
        kwargs = {'categories': categories} if categories else {}
        if handle_unknown == 'use_encoded_value':
            kwargs['handle_unknown'] = handle_unknown
            kwargs['unknown_value'] = -1
        else:
            kwargs['handle_unknown'] = handle_unknown
        
        encoder = OrdinalEncoder(**kwargs)
        
        # Handle pandas Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
            columns = [data.columns[0]]
            
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # If columns not specified, use all object and category dtypes
            if columns is None:
                columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not columns:
                    warnings.warn("No object or category columns found for ordinal encoding.")
                    return data, encoder
                    
            # Get data subset with only the columns to encode
            categorical_data = data[columns]
            
            # Fit encoder
            encoder.fit(categorical_data)
            
            # Transform data
            encoded_data = encoder.transform(categorical_data)
            
            # Create a copy of data to avoid modifying the original
            result = data.copy()
            
            # Replace categorical columns with encoded values
            for i, col in enumerate(columns):
                result[col] = encoded_data[:, i]
                
            return result, encoder
        else:
            # Handle numpy arrays
            # Reshape if 1D
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            # Fit and transform
            encoded_data = encoder.fit_transform(data)
            
            return encoded_data, encoder
    
    @staticmethod
    def target_encode(data: Union[pd.DataFrame, pd.Series],
                     target: pd.Series,
                     columns: Optional[List[str]] = None,
                     min_samples_leaf: int = 1,
                     smoothing: int = 1) -> pd.DataFrame:
        """
        Replace categorical feature with the average target value for each category.
        
        Args:
            data (Union[pd.DataFrame, pd.Series]): The categorical data to encode.
            target (pd.Series): The target variable.
            columns (Optional[List[str]], optional): Column names to encode. If None and data is DataFrame,
                                                   all object and category columns will be encoded.
                                                   Defaults to None.
            min_samples_leaf (int, optional): Minimum samples to take category average. Defaults to 1.
            smoothing (int, optional): Smoothing effect to balance categorical average. Defaults to 1.
            
        Returns:
            pd.DataFrame: Data with target encoding.
        """
        # Handle pandas Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
            columns = [data.columns[0]]
            
        # If columns not specified, use all object and category dtypes
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not columns:
                warnings.warn("No object or category columns found for target encoding.")
                return data
                
        # Create a copy of data to avoid modifying the original
        result = data.copy()
        
        # Ensure target has the same index as data
        if not target.index.equals(data.index):
            warnings.warn("Target index does not match data index. Aligning indices.")
            target = target.reindex(data.index)
            
        # Calculate the global mean for fallback
        global_mean = target.mean()
        
        # Encode each column
        for col in columns:
            # Group by the column and calculate mean target value
            means = target.groupby(data[col]).mean()
            counts = target.groupby(data[col]).count()
            
            # Calculate smoothed mean
            smoothed_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
            
            # Map the smoothed means back to the data
            result[col] = result[col].map(smoothed_means)
            
            # Fill any missing values (new categories) with global mean
            result[col] = result[col].fillna(global_mean)
            
        return result


class FeatureSelector:
    """Class for feature selection."""
    
    @staticmethod
    def select_k_best(X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series],
                     k: int = 10,
                     score_func: Callable = f_classif) -> Tuple[Union[np.ndarray, pd.DataFrame], SelectKBest]:
        """
        Select features based on the k highest scores.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input samples.
            y (Union[np.ndarray, pd.Series]): The target values.
            k (int, optional): Number of top features to select. Defaults to 10.
            score_func (Callable, optional): Function taking two arrays X and y, and returning
                                            a pair of arrays (scores, pvalues). Defaults to f_classif.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], SelectKBest]: 
                Selected features and the fitted selector.
        """
        selector = SelectKBest(score_func=score_func, k=k)
        
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Save column names and index
            columns = X.columns
            index = X.index
            
            # Fit and transform
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support()
            selected_features = columns[selected_indices]
            
            # Convert back to DataFrame with selected columns
            return pd.DataFrame(X_selected, columns=selected_features, index=index), selector
        else:
            # Handle numpy arrays
            return selector.fit_transform(X, y), selector
    
    @staticmethod
    def select_percentile(X: Union[np.ndarray, pd.DataFrame], 
                         y: Union[np.ndarray, pd.Series],
                         percentile: int = 10,
                         score_func: Callable = f_classif) -> Tuple[Union[np.ndarray, pd.DataFrame], SelectPercentile]:
        """
        Select features based on percentile of the highest scores.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input samples.
            y (Union[np.ndarray, pd.Series]): The target values.
            percentile (int, optional): Percent of features to select. Defaults to 10.
            score_func (Callable, optional): Function taking two arrays X and y, and returning
                                            a pair of arrays (scores, pvalues). Defaults to f_classif.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], SelectPercentile]: 
                Selected features and the fitted selector.
        """
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Save column names and index
            columns = X.columns
            index = X.index
            
            # Fit and transform
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support()
            selected_features = columns[selected_indices]
            
            # Convert back to DataFrame with selected columns
            return pd.DataFrame(X_selected, columns=selected_features, index=index), selector
        else:
            # Handle numpy arrays
            return selector.fit_transform(X, y), selector
    
    @staticmethod
    def select_from_model(X: Union[np.ndarray, pd.DataFrame], 
                         y: Union[np.ndarray, pd.Series],
                         estimator: Any,
                         threshold: Optional[Union[str, float, None]] = None,
                         prefit: bool = False) -> Tuple[Union[np.ndarray, pd.DataFrame], SelectFromModel]:
        """
        Select features based on importance weights from an estimator.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input samples.
            y (Union[np.ndarray, pd.Series]): The target values.
            estimator (Any): The base estimator from which the transformer is built.
            threshold (Optional[Union[str, float, None]], optional): Feature selection threshold.
                                                                   Defaults to None.
            prefit (bool, optional): Whether estimator has already been fit. Defaults to False.
            
        Returns:
            Tuple[Union[np.ndarray, pd.DataFrame], SelectFromModel]: 
                Selected features and the fitted selector.
        """
        selector = SelectFromModel(estimator=estimator, threshold=threshold, prefit=prefit)
        
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Save column names and index
            columns = X.columns
            index = X.index
            
            # Fit and transform
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support()
            selected_features = columns[selected_indices]
            
            # Convert back to DataFrame with selected columns
            return pd.DataFrame(X_selected, columns=selected_features, index=index), selector
        else:
            # Handle numpy arrays
            return selector.fit_transform(X, y), selector
    
    @staticmethod
    def variance_threshold(X: Union[np.ndarray, pd.DataFrame],
                          threshold: float = 0.0) -> Tuple[Union[np.ndarray, pd.DataFrame], VarianceThreshold]:
        """
        Remove features with low variance.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input samples.
