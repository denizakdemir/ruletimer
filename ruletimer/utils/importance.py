"""
Importance analysis utilities for rule-based time-to-event models.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable

class ImportanceAnalyzer:
    """Analyze variable importance and dependencies in rule ensemble models."""
    
    @staticmethod
    def calculate_permutation_importance(
        model,
        X: np.ndarray,
        y,
        prediction_func: Optional[Callable] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ) -> Dict:
        """Calculate permutation-based feature importance.
        
        Parameters
        ----------
        model : BaseRuleEnsemble
            Fitted model
        X : array-like
            Feature matrix
        y : MultiState, Survival, or CompetingRisks
            Time-to-event data
        prediction_func : callable, optional
            Function to generate predictions for evaluation
            Must accept X as input and return array-like
            If None, uses predict_risk method
        n_repeats : int, default=10
            Number of permutation repeats
        random_state : int, optional
            Random state for reproducibility
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'importances': Feature importances
            - 'importances_std': Standard deviation of importances
            - 'feature_names': Feature names if available
        """
        # Default to predict_risk for prediction function
        if prediction_func is None:
            prediction_func = model.predict_risk
        
        # Get baseline prediction
        baseline_pred = prediction_func(X)
        
        # Handle different return types
        if isinstance(baseline_pred, dict):
            # For multi-state or competing risks predictions
            baseline_score = np.mean([np.mean(p) for p in baseline_pred.values()])
            
            def score_func(X):
                pred = prediction_func(X)
                return np.mean([np.mean(p) for p in pred.values()])
        else:
            # For standard predictions
            baseline_score = np.mean(baseline_pred)
            
            def score_func(X):
                return np.mean(prediction_func(X))
        
        # Initialize importance arrays
        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))
        
        # Set random state
        rng = np.random.RandomState(random_state)
        
        # Permute each feature and calculate importance
        for i in range(n_features):
            for j in range(n_repeats):
                # Create permuted data
                X_permuted = X.copy()
                perm_idx = rng.permutation(X.shape[0])
                X_permuted[:, i] = X[perm_idx, i]
                
                # Calculate score with permuted feature
                permuted_score = score_func(X_permuted)
                
                # Importance is decrease in score (or increase for errors)
                importances[j, i] = baseline_score - permuted_score
        
        # Calculate mean and std of importances
        imp_mean = np.mean(importances, axis=0)
        imp_std = np.std(importances, axis=0)
        
        # Package results
        result = {
            'importances': imp_mean,
            'importances_std': imp_std
        }
        
        # Add feature names if available
        if hasattr(model, '_feature_names'):
            result['feature_names'] = model._feature_names
        
        return result
    
    @staticmethod
    def calculate_dependence_matrix(
        model,
        X: np.ndarray,
        prediction_func: Optional[Callable] = None,
        threshold: float = 0.05,
        n_points: int = 10
    ) -> np.ndarray:
        """Calculate feature dependency matrix using partial dependence.
        
        Parameters
        ----------
        model : BaseRuleEnsemble
            Fitted model
        X : array-like
            Feature matrix
        prediction_func : callable, optional
            Function to generate predictions for evaluation
            Must accept X as input and return array-like
            If None, uses predict_risk method
        threshold : float, default=0.05
            Threshold for significance of dependency
        n_points : int, default=10
            Number of points for evaluating each feature
            
        Returns
        -------
        numpy.ndarray
            Dependency matrix of shape (n_features, n_features)
            with values between 0 and 1 indicating strength of dependency
        """
        # Default to predict_risk for prediction function
        if prediction_func is None:
            prediction_func = model.predict_risk
        
        # Get number of features
        n_features = X.shape[1]
        
        # Initialize dependency matrix
        dependency = np.zeros((n_features, n_features))
        
        # For each pair of features
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Generate grid of values for features i and j
                x_i = np.linspace(np.min(X[:, i]), np.max(X[:, i]), n_points)
                x_j = np.linspace(np.min(X[:, j]), np.max(X[:, j]), n_points)
                
                # Create 2D grid
                X_grid = np.zeros((n_points * n_points, n_features))
                for k in range(n_features):
                    if k != i and k != j:
                        X_grid[:, k] = np.median(X[:, k])
                
                # Fill in feature i and j values
                idx = 0
                for vi in x_i:
                    for vj in x_j:
                        X_grid[idx, i] = vi
                        X_grid[idx, j] = vj
                        idx += 1
                
                # Get predictions for grid
                pred_grid = prediction_func(X_grid)
                
                # Reshape predictions to 2D grid
                if isinstance(pred_grid, dict):
                    # For multi-state or competing risks
                    pred_values = np.mean([p for p in pred_grid.values()], axis=0)
                else:
                    pred_values = pred_grid
                
                pred_2d = pred_values.reshape(n_points, n_points)
                
                # Calculate dependency using correlation of differences
                row_diffs = np.diff(pred_2d, axis=0)
                col_diffs = np.diff(pred_2d, axis=1)
                
                # Flatten and calculate correlation
                corr = np.corrcoef(row_diffs.flatten(), col_diffs.flatten())[0, 1]
                
                # Set dependency value (absolute correlation)
                dependency[i, j] = dependency[j, i] = abs(corr)
        
        # Set diagonal to 1
        np.fill_diagonal(dependency, 1.0)
        
        # Apply threshold
        dependency[dependency < threshold] = 0
        
        return dependency 