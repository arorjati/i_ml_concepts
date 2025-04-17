"""
Dataset loading utilities.

This module provides functions to load datasets from scikit-learn and other sources.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from typing import Dict, Tuple, Optional, Union, List, Any


class DatasetLoader:
    """Class for loading and managing datasets from various sources."""
    
    # Dictionary mapping dataset names to their loading functions in sklearn
    SKLEARN_DATASETS = {
        # Classification datasets
        'iris': datasets.load_iris,
        'breast_cancer': datasets.load_breast_cancer,
        'wine': datasets.load_wine,
        'digits': datasets.load_digits,
        
        # Regression datasets
        'boston': datasets.fetch_california_housing,  # Boston dataset is deprecated, using California housing
        'diabetes': datasets.load_diabetes,
        
        # Clustering datasets
        'blobs': lambda **kwargs: datasets.make_blobs(**kwargs)[0],
        
        # Other datasets
        'olivetti_faces': datasets.fetch_olivetti_faces,
        '20newsgroups': datasets.fetch_20newsgroups,
        'lfw_people': datasets.fetch_lfw_people,
        'covtype': datasets.fetch_covtype,
    }
    
    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """
        Return a list of available dataset names.
        
        Returns:
            List[str]: Names of available datasets.
        """
        return list(cls.SKLEARN_DATASETS.keys())
    
    @classmethod
    def load_dataset(cls, name: str, as_pandas: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Load a dataset by name.
        
        Args:
            name (str): Name of the dataset to load.
            as_pandas (bool, optional): If True, return data as pandas DataFrame. Defaults to True.
            **kwargs: Additional arguments to pass to the dataset loader function.
            
        Returns:
            Dict[str, Any]: Dictionary containing dataset information.
            
        Raises:
            ValueError: If the dataset name is not recognized.
        """
        if name not in cls.SKLEARN_DATASETS:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {cls.get_available_datasets()}")
        
        # Load the dataset using the appropriate function
        dataset = cls.SKLEARN_DATASETS[name](**kwargs)
        
        # Process the dataset
        result = {}
        
        # Handle different dataset formats
        if hasattr(dataset, 'data') and hasattr(dataset, 'target'):
            # Most sklearn datasets
            X = dataset.data
            y = dataset.target
            
            if as_pandas:
                # Create DataFrame with feature names if available
                feature_names = getattr(dataset, 'feature_names', None)
                if feature_names is None:
                    # Create default feature names if none exist
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
                X = pd.DataFrame(X, columns=feature_names)
                
                # Create target Series with target names if available
                target_names = getattr(dataset, 'target_names', None)
                if target_names is not None and len(target_names) == len(np.unique(y)):
                    y = pd.Series(y).map(lambda i: target_names[i] if isinstance(i, (int, np.integer)) and i < len(target_names) else i)
                else:
                    y = pd.Series(y)
            
            result['X'] = X
            result['y'] = y
            
            # Add additional information if available
            for attr in ['feature_names', 'target_names', 'DESCR', 'filename', 'frame']:
                if hasattr(dataset, attr):
                    result[attr] = getattr(dataset, attr)
                    
        elif isinstance(dataset, dict):
            # Some sklearn datasets return dictionaries
            result = dataset
            
            # Convert to pandas if requested
            if as_pandas and 'data' in dataset and 'target' in dataset:
                feature_names = dataset.get('feature_names', [f'feature_{i}' for i in range(dataset['data'].shape[1])])
                result['X'] = pd.DataFrame(dataset['data'], columns=feature_names)
                result['y'] = pd.Series(dataset['target'])
        
        # Add metadata about the dataset
        result['name'] = name
        
        return result
    
    @classmethod
    def generate_synthetic_dataset(cls, type_name: str, n_samples: int = 100, 
                                  n_features: int = 2, as_pandas: bool = True, 
                                  random_state: Optional[int] = None, 
                                  **kwargs) -> Dict[str, Any]:
        """
        Generate a synthetic dataset using scikit-learn's generators.
        
        Args:
            type_name (str): Type of dataset to generate ('classification', 'regression', 'clusters', etc.).
            n_samples (int, optional): Number of samples. Defaults to 100.
            n_features (int, optional): Number of features. Defaults to 2.
            as_pandas (bool, optional): If True, return data as pandas DataFrame. Defaults to True.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional arguments for the dataset generator.
            
        Returns:
            Dict[str, Any]: Dictionary containing generated dataset.
            
        Raises:
            ValueError: If the dataset type is not recognized.
        """
        generator_map = {
            'classification': datasets.make_classification,
            'regression': datasets.make_regression,
            'clusters': datasets.make_blobs,
            'circles': datasets.make_circles,
            'moons': datasets.make_moons,
            'swiss_roll': datasets.make_swiss_roll,
            'sparse': datasets.make_sparse_uncorrelated,
            'friedman': datasets.make_friedman1,
            'gaussian_quantiles': datasets.make_gaussian_quantiles
        }
        
        if type_name not in generator_map:
            raise ValueError(f"Unknown dataset type: {type_name}. Available types: {list(generator_map.keys())}")
        
        # Select generator function
        generator = generator_map[type_name]
        
        # Build common parameters
        common_params = {
            'n_samples': n_samples,
            'random_state': random_state
        }
        
        # Add n_features parameter for supported generators
        if type_name not in ['swiss_roll', 'circles', 'moons']:
            common_params['n_features'] = n_features
            
        # Merge parameters
        params = {**common_params, **kwargs}
        
        # Generate dataset
        if type_name == 'swiss_roll':
            X, t = generator(**params)
            y = t  # Use the parametric coordinate as target
        else:
            X, y = generator(**params)
        
        # Create result dictionary
        result = {}
        
        if as_pandas:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            result['X'] = pd.DataFrame(X, columns=feature_names)
            result['y'] = pd.Series(y, name='target')
        else:
            result['X'] = X
            result['y'] = y
            
        # Add metadata
        result['name'] = f"synthetic_{type_name}"
        result['type'] = type_name
        result['params'] = params
        
        return result
    
    @staticmethod
    def split_dataset(X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      test_size: float = 0.2, 
                      validation_size: Optional[float] = None,
                      random_state: Optional[int] = None,
                      stratify: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Split dataset into training, validation, and test sets.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            validation_size (Optional[float], optional): Proportion of the dataset to include in the validation split.
                If None, no validation set is created. Defaults to None.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            stratify (Optional[Union[np.ndarray, pd.Series]], optional): If not None, data is split in a stratified fashion. Defaults to None.
            
        Returns:
            Dict[str, Any]: Dictionary containing split datasets.
        """
        from sklearn.model_selection import train_test_split
        
        result = {}
        
        # First split: training + validation vs test
        strat = y if stratify is True else stratify
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
        
        result['X_test'] = X_test
        result['y_test'] = y_test
        
        # Second split: training vs validation (if needed)
        if validation_size is not None:
            # Adjust validation_size to be relative to train_val size
            adjusted_val_size = validation_size / (1 - test_size)
            strat_val = y_train_val if stratify is True else stratify
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=adjusted_val_size, 
                random_state=random_state, stratify=strat_val
            )
            
            result['X_train'] = X_train
            result['y_train'] = y_train
            result['X_val'] = X_val
            result['y_val'] = y_val
        else:
            result['X_train'] = X_train_val
            result['y_train'] = y_train_val
            
        # Calculate split sizes
        total_samples = len(y)
        result['split_sizes'] = {
            'total': total_samples,
            'train': len(result['y_train']),
            'test': len(y_test)
        }
        
        if validation_size is not None:
            result['split_sizes']['val'] = len(result['y_val'])
            
        # Store split parameters
        result['split_params'] = {
            'test_size': test_size,
            'validation_size': validation_size,
            'random_state': random_state,
            'stratify': stratify is not None
        }
        
        return result


def load_dataset(name: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to load a dataset by name.
    
    Args:
        name (str): Name of the dataset to load.
        **kwargs: Additional arguments to pass to DatasetLoader.load_dataset().
        
    Returns:
        Dict[str, Any]: Dictionary containing dataset information.
    """
    return DatasetLoader.load_dataset(name, **kwargs)


def generate_synthetic_dataset(type_name: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to generate a synthetic dataset.
    
    Args:
        type_name (str): Type of dataset to generate.
        **kwargs: Additional arguments to pass to DatasetLoader.generate_synthetic_dataset().
        
    Returns:
        Dict[str, Any]: Dictionary containing generated dataset.
    """
    return DatasetLoader.generate_synthetic_dataset(type_name, **kwargs)


def get_available_datasets() -> List[str]:
    """
    Return a list of available dataset names.
    
    Returns:
        List[str]: Names of available datasets.
    """
    return DatasetLoader.get_available_datasets()


def split_dataset(X, y, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to split a dataset.
    
    Args:
        X: Features.
        y: Target.
        **kwargs: Additional arguments to pass to DatasetLoader.split_dataset().
        
    Returns:
        Dict[str, Any]: Dictionary containing split datasets.
    """
    return DatasetLoader.split_dataset(X, y, **kwargs)
