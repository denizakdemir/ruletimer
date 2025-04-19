"""
Base class for rule ensemble models
"""

import numpy as np
from sklearn.base import BaseEstimator
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import warnings

class BaseRuleEnsemble(BaseEstimator):
    """Base class for rule ensemble models"""
    
    def __init__(self, max_rules: int = 100,
                 alpha: float = 0.05,
                 l1_ratio: float = 0.5,
                 random_state: Optional[int] = None,
                 tree_type: str = 'regression',
                 max_depth: int = 3,
                 min_samples_leaf: int = 10,
                 n_estimators: int = 100,
                 prune_rules: bool = True,
                 prune_threshold: float = 0.01,
                 support_time_dependent: bool = True):
        """
        Initialize base rule ensemble model
        
        Parameters
        ----------
        max_rules : int, default=100
            Maximum number of rules to include in the ensemble
        alpha : float, default=0.05
            Regularization strength
        l1_ratio : float, default=0.5
            Ratio of L1 to L2 regularization (0 = L2, 1 = L1)
        random_state : int, optional
            Random seed for reproducibility
        tree_type : str, default='regression'
            Type of decision tree to use ('regression' or 'classification')
        max_depth : int, default=3
            Maximum depth of the decision trees
        min_samples_leaf : int, default=10
            Minimum number of samples required to be at a leaf node
        n_estimators : int, default=100
            Number of trees in the forest (if using random forest)
        prune_rules : bool, default=True
            Whether to prune redundant rules
        prune_threshold : float, default=0.01
            Threshold for rule pruning based on similarity
        support_time_dependent : bool, default=True
            Whether to support time-dependent covariates
        """
        self.max_rules = max_rules
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.tree_type = tree_type
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.prune_rules = prune_rules
        self.prune_threshold = prune_threshold
        self.support_time_dependent = support_time_dependent
        
        self.rules_ = None
        self.rule_weights_ = None
        self._feature_importances_ = None
        self.time_dependent_features_ = None
        self.scaler_ = StandardScaler()
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances"""
        if not hasattr(self, '_feature_importances_'):
            self._feature_importances_ = None
        if self._feature_importances_ is None and hasattr(self, '_X'):
            self._compute_feature_importances()
        return self._feature_importances_

    @feature_importances_.setter
    def feature_importances_(self, value: np.ndarray) -> None:
        """Set feature importances"""
        self._feature_importances_ = value
    
    def _get_tree_model(self) -> Union[DecisionTreeRegressor, DecisionTreeClassifier,
                                      RandomForestRegressor, RandomForestClassifier]:
        """Get the appropriate tree model based on configuration"""
        if self.tree_type == 'regression':
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        elif self.tree_type == 'classification':
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        elif self.tree_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported tree_type: {self.tree_type}")
    
    def _extract_rules(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """
        Extract rules from decision trees with enhanced capabilities
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        rules : list of str
            List of extracted rules

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def _prune_rules(self, rules: List[List[Tuple]]) -> List[List[Tuple]]:
        """
        Prune redundant rules based on similarity
        
        Parameters
        ----------
        rules : list of list of tuples
            List of rules to prune
            
        Returns
        -------
        pruned_rules : list of list of tuples
            Pruned list of rules
        """
        if len(rules) <= 1:
            return rules
        
        # Convert rules to binary vectors
        if isinstance(self._X, pd.DataFrame):
            n_features = len(self._X.columns)
        else:
            n_features = self._X.shape[1]
        
        rule_vectors = np.zeros((len(rules), n_features))
        for i, rule in enumerate(rules):
            for feature, _, _ in rule:
                rule_vectors[i, feature] = 1
        
        # Calculate similarity matrix
        similarity = np.abs(np.corrcoef(rule_vectors))
        
        # Find similar rules
        to_remove = set()
        for i in range(len(rules)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(rules)):
                if similarity[i, j] > self.prune_threshold:
                    to_remove.add(j)
        
        # Return pruned rules
        return [rule for i, rule in enumerate(rules) if i not in to_remove]
    
    def _handle_time_dependent_covariates(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Handle time-dependent covariates
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        X_processed : array-like of shape (n_samples, n_features)
            Processed data with time-dependent features
        """
        if not self.support_time_dependent:
            return X
        
        if isinstance(X, pd.DataFrame):
            # Identify time-dependent features
            time_dependent_features = []
            for col in X.columns:
                if isinstance(col, tuple) and len(col) == 2:
                    time_dependent_features.append(col)
            
            if time_dependent_features:
                self.time_dependent_features_ = time_dependent_features
                # Create interaction terms with time
                X_processed = X.copy()
                for feature in time_dependent_features:
                    X_processed[f"{feature}_time"] = X[feature] * self._y.time
                return X_processed
        
        return X
    
    def _evaluate_rules(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Evaluate rules on data with support for time-dependent covariates
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to evaluate rules on
            
        Returns
        -------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations for each sample
        """
        # Handle time-dependent covariates
        X_processed = self._handle_time_dependent_covariates(X)
        
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.values
        
        rule_values = np.zeros((X_processed.shape[0], len(self.rules_)))
        for i, conditions in enumerate(self.rules_):
            mask = np.ones(X_processed.shape[0], dtype=bool)
            for feature, op, threshold in conditions:
                if op == "<=":
                    mask &= (X_processed[:, feature] <= threshold)
                else:  # op == ">"
                    mask &= (X_processed[:, feature] > threshold)
            rule_values[:, i] = mask
        
        return rule_values
    
    def _compute_feature_importances(self) -> None:
        """
        Compute feature importances based on rule weights and information gain
        
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def _fit_weights(self, rule_values: np.ndarray, y: np.ndarray) -> None:
        """
        Fit weights for the rules using regularized regression
        
        Parameters
        ----------
        rule_values : array-like of shape (n_samples, n_rules)
            Binary matrix indicating which rules apply to each sample
        y : array-like of shape (n_samples,)
            Target values
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'BaseRuleEnsemble':
        """
        Fit the rule ensemble model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : BaseRuleEnsemble
            Fitted model

        Raises
        ------
        NotImplementedError
            If _extract_rules or _fit_weights is not implemented by subclass
        """
        # Store target variable and input data
        self._y = y
        self._X = X
        
        # Extract rules
        self.rules_ = self._extract_rules(X)
        
        # Evaluate rules
        rule_values = self._evaluate_rules(X)
        
        # Fit weights
        self._fit_weights(rule_values, y)
        
        # Compute feature importances
        self._compute_feature_importances()
        
        return self
    
    def get_rules(self) -> List[str]:
        """
        Get the list of rules in the ensemble
        
        Returns
        -------
        rules : list of str
            List of rules
        """
        return self.rules_
    
    def get_rule_weights(self) -> np.ndarray:
        """
        Get the weights of rules in the ensemble
        
        Returns
        -------
        weights : array-like of shape (n_rules,)
            Rule weights
        """
        return self.rule_weights_ 