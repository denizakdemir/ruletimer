"""
Rule ensemble model for standard survival analysis
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from typing import Union, List, Optional, Dict, Tuple
import pandas as pd
from scipy.stats import weibull_min, expon
from scipy.optimize import minimize
from ..data import Survival
from .base import BaseRuleEnsemble

class RuleSurvival(BaseRuleEnsemble):
    """Rule ensemble model for standard survival analysis"""
    
    def __init__(self, max_rules: int = 100,
                 alpha: float = 0.05,
                 l1_ratio: float = 0.5,
                 random_state: Optional[int] = None,
                 model_type: str = 'cox',
                 tree_type: str = 'regression',
                 max_depth: int = 3,
                 min_samples_leaf: int = 10,
                 n_estimators: int = 100,
                 prune_rules: bool = True,
                 prune_threshold: float = 0.01,
                 support_time_dependent: bool = True,
                 tree_growing_strategy: str = 'single',
                 interaction_depth: int = 2):
        """
        Initialize rule ensemble survival model
        
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
        model_type : str, default='cox'
            Type of survival model ('cox', 'weibull', 'exponential')
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
        tree_growing_strategy : str, default='single'
            Strategy for growing trees ('single', 'forest', 'interaction')
        interaction_depth : int, default=2
            Maximum depth for interaction terms when using 'interaction' strategy
        """
        super().__init__(max_rules=max_rules, alpha=alpha,
                        l1_ratio=l1_ratio, random_state=random_state,
                        tree_type=tree_type, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        n_estimators=n_estimators,
                        prune_rules=prune_rules,
                        prune_threshold=prune_threshold,
                        support_time_dependent=support_time_dependent)
        
        self.model_type = model_type
        self.tree_growing_strategy = tree_growing_strategy
        self.interaction_depth = interaction_depth
        self.baseline_hazard_ = None
        self.baseline_survival_ = None
        self.parametric_params_ = None
    
    def _extract_rules(self, X: np.ndarray) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from the data"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Store data for later use
        self._X = X
        
        # Initialize rules list
        rules = []
        rule_features = []
        
        # For each feature
        for feature_idx in range(X.shape[1]):
            # Get feature values
            feature_values = X[:, feature_idx]
            
            # Get quantiles for continuous features or unique values for binary features
            if len(np.unique(feature_values)) > 2:  # Continuous feature
                thresholds = np.percentile(feature_values, [25, 50, 75])
                for threshold in thresholds:
                    # Add rules for this threshold
                    rules.append([(feature_idx, "<=", threshold)])
                    rules.append([(feature_idx, ">", threshold)])
                    rule_features.extend([[feature_idx], [feature_idx]])
            else:  # Binary feature
                threshold = 0.5
                rules.append([(feature_idx, "<=", threshold)])
                rules.append([(feature_idx, ">", threshold)])
                rule_features.extend([[feature_idx], [feature_idx]])
        
        # Add interaction rules
        if self.max_depth > 1:
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    # Get feature values
                    feature_i = X[:, i]
                    feature_j = X[:, j]
                    
                    # Get thresholds
                    if len(np.unique(feature_i)) > 2:
                        thresh_i = np.median(feature_i)
                    else:
                        thresh_i = 0.5
                    
                    if len(np.unique(feature_j)) > 2:
                        thresh_j = np.median(feature_j)
                    else:
                        thresh_j = 0.5
                    
                    # Add interaction rules
                    rules.append([(i, "<=", thresh_i), (j, "<=", thresh_j)])
                    rules.append([(i, "<=", thresh_i), (j, ">", thresh_j)])
                    rules.append([(i, ">", thresh_i), (j, "<=", thresh_j)])
                    rules.append([(i, ">", thresh_i), (j, ">", thresh_j)])
                    
                    rule_features.extend([[i, j]] * 4)
        
        # Store rule features
        self.rule_features_ = rule_features
        
        # Limit number of rules
        if len(rules) > self.max_rules:
            rules = rules[:self.max_rules]
            self.rule_features_ = rule_features[:self.max_rules]
        
        return rules
    
    def _extract_rules_single_tree(self, X: np.ndarray) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from a single decision tree"""
        if self.tree_type == 'regression':
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf,
                                       random_state=self.random_state)
        else:
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.random_state)
        
        # Fit tree to get rules
        tree.fit(X, self._y.time)
        
        # Extract rules from tree
        rules = []
        n_nodes = tree.tree_.node_count
        left_child = tree.tree_.children_left
        right_child = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        def extract_rules(node_id, conditions):
            if left_child[node_id] == right_child[node_id]:  # leaf node
                if conditions:  # Only add non-empty rules
                    rule = []
                    unique_features = set()
                    for feat, op, thresh in conditions:
                        rule.append((feat, op, thresh))
                        unique_features.add(feat)
                    rules.append((rule, list(unique_features)))
                return
            
            # Add left child rule
            left_conditions = conditions + [(feature[node_id], "<=", threshold[node_id])]
            extract_rules(left_child[node_id], left_conditions)
            
            # Add right child rule
            right_conditions = conditions + [(feature[node_id], ">", threshold[node_id])]
            extract_rules(right_child[node_id], right_conditions)
        
        extract_rules(0, [])
        
        # If no rules were extracted, create a simple threshold rule
        if not rules:
            feature_idx = 0
            threshold = np.median(X[:, feature_idx])
            rules = [
                ([(feature_idx, "<=", threshold)], [feature_idx]),
                ([(feature_idx, ">", threshold)], [feature_idx])
            ]
        
        # Store both rules and their feature indices
        self.rule_features_ = [features for _, features in rules]
        return [rule for rule, _ in rules]
    
    def _extract_rules_forest(self, X: np.ndarray) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from a random forest"""
        if self.tree_type == 'regression':
            forest = RandomForestRegressor(n_estimators=self.n_estimators,
                                         max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=self.random_state)
        else:
            forest = RandomForestClassifier(n_estimators=self.n_estimators,
                                          max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          random_state=self.random_state)
        
        forest.fit(X, self._y.time)
        rules = []
        for tree in forest.estimators_:
            tree_rules = self._extract_rules_from_tree(tree)
            rules.extend(tree_rules)
            
            if len(rules) >= self.max_rules:
                return rules[:self.max_rules]
        
        # If no rules were extracted, create a simple threshold rule
        if not rules:
            feature_idx = 0
            threshold = np.median(X[:, feature_idx])
            rules = [[(feature_idx, "<=", threshold)]]
        
        return rules
    
    def _extract_rules_interaction(self, X: np.ndarray) -> List[List[Tuple[int, str, float]]]:
        """Extract rules focusing on feature interactions"""
        rules = []
        n_features = X.shape[1]
        
        # Extract rules from individual features
        for i in range(n_features):
            tree = DecisionTreeRegressor(max_depth=1,
                                       min_samples_leaf=self.min_samples_leaf,
                                       random_state=self.random_state)
            tree.fit(X[:, [i]], self._y.time)
            tree_rules = self._extract_rules_from_tree(tree)
            rules.extend(tree_rules)
            
            if len(rules) >= self.max_rules:
                return rules[:self.max_rules]
        
        # Extract rules from feature interactions
        for depth in range(2, self.interaction_depth + 1):
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    tree = DecisionTreeRegressor(max_depth=depth,
                                               min_samples_leaf=self.min_samples_leaf,
                                               random_state=self.random_state)
                    tree.fit(X[:, [i, j]], self._y.time)
                    tree_rules = self._extract_rules_from_tree(tree)
                    rules.extend(tree_rules)
                    
                    if len(rules) >= self.max_rules:
                        return rules[:self.max_rules]
        
        # If no rules were extracted, create a simple threshold rule
        if not rules:
            feature_idx = 0
            threshold = np.median(X[:, feature_idx])
            rules = [[(feature_idx, "<=", threshold)]]
        
        return rules
    
    def _extract_rules_from_tree(self, tree: Union[DecisionTreeRegressor, DecisionTreeClassifier]) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from a single decision tree"""
        rules = []
        n_nodes = tree.tree_.node_count
        left_child = tree.tree_.children_left
        right_child = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        def extract_rules(node_id, conditions):
            if left_child[node_id] == right_child[node_id]:  # leaf node
                if conditions:  # Only add non-empty rules
                    rule = []
                    unique_features = set()
                    for feat, op, thresh in conditions:
                        rule.append((feat, op, thresh))
                        unique_features.add(feat)
                    rules.append((rule, list(unique_features)))
                return
            
            # Add left child rule
            left_conditions = conditions + [(feature[node_id], "<=", threshold[node_id])]
            extract_rules(left_child[node_id], left_conditions)
            
            # Add right child rule
            right_conditions = conditions + [(feature[node_id], ">", threshold[node_id])]
            extract_rules(right_child[node_id], right_conditions)
        
        extract_rules(0, [])
        # Store both rules and their feature indices
        self.rule_features_ = [features for _, features in rules]
        return [rule for rule, _ in rules]
    
    def _prune_rules(self, rules: List[List[Tuple[int, str, float]]]) -> List[List[Tuple[int, str, float]]]:
        """
        Prune redundant rules based on similarity
        
        Parameters
        ----------
        rules : list of rules
            List of rules to prune
            
        Returns
        -------
        pruned_rules : list of rules
            List of pruned rules
        """
        if not self.prune_rules or len(rules) <= 1:
            return rules
        
        # Convert rules to boolean matrices
        rule_matrices = []
        for rule in rules:
            # Create mask for current rule
            mask = np.ones(self._X.shape[0], dtype=bool)
            
            # Apply each condition in the rule
            for feature_idx, operator, threshold in rule:
                if operator == "<=":
                    mask &= (self._X[:, feature_idx] <= threshold)
                else:  # operator == ">"
                    mask &= (self._X[:, feature_idx] > threshold)
            
            rule_matrices.append(mask)
        
        # Compute similarity matrix
        n_rules = len(rules)
        similarity = np.zeros((n_rules, n_rules))
        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                # Compute Jaccard similarity
                intersection = np.sum(rule_matrices[i] & rule_matrices[j])
                union = np.sum(rule_matrices[i] | rule_matrices[j])
                similarity[i, j] = intersection / union if union > 0 else 0
                similarity[j, i] = similarity[i, j]
        
        # Cluster rules based on similarity
        clusters = []
        used = set()
        for i in range(n_rules):
            if i in used:
                continue
            
            # Find similar rules
            similar = np.where(similarity[i] >= self.prune_threshold)[0]
            cluster = [j for j in similar if j not in used]
            
            # Add cluster
            if cluster:
                clusters.append(cluster)
                used.update(cluster)
        
        # Select representative rule from each cluster
        pruned_rules = []
        for cluster in clusters:
            # Use the shortest rule as representative
            rep_idx = min(cluster, key=lambda i: len(rules[i]))
            pruned_rules.append(rules[rep_idx])
        
        return pruned_rules
    
    def _evaluate_rules(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Evaluate rules on data
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to evaluate rules on
            
        Returns
        -------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations for each sample
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        rule_values = np.zeros((X.shape[0], len(self.rules_)))
        for i, rule in enumerate(self.rules_):
            # Start with all True
            mask = np.ones(X.shape[0], dtype=bool)
            
            # Apply each condition in the rule
            for feature_idx, operator, threshold in rule:
                if operator == "<=":
                    mask &= (X[:, feature_idx] <= threshold)
                else:  # operator == ">"
                    mask &= (X[:, feature_idx] > threshold)
            
            rule_values[:, i] = mask
        
        return rule_values
    
    def _fit_weights(self, rule_values: np.ndarray, y: Survival) -> None:
        """
        Fit rule weights using Cox regression
        
        Parameters
        ----------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations
        y : Survival
            Survival data
        """
        self._y = y
        
        if self.model_type == 'cox':
            # Sort by time
            sort_idx = np.argsort(y.time)
            times = y.time[sort_idx]
            events = y.event[sort_idx]
            rule_values = rule_values[sort_idx]
            
            # Initialize weights
            n_rules = rule_values.shape[1]
            weights = np.zeros(n_rules)
            
            # Newton-Raphson optimization
            max_iter = 100
            tol = 1e-6
            
            for iter in range(max_iter):
                # Compute risk scores
                risk_scores = np.exp(np.dot(rule_values, weights))
                
                # Compute gradient
                gradient = np.zeros(n_rules)
                hessian = np.zeros((n_rules, n_rules))
                
                # For each event time
                for i in range(len(times)):
                    if events[i]:
                        # Get risk set
                        at_risk = times >= times[i]
                        risk_set_scores = risk_scores[at_risk]
                        risk_set_rules = rule_values[at_risk]
                        
                        # Compute gradient
                        weighted_mean = np.average(risk_set_rules, weights=risk_set_scores, axis=0)
                        gradient += rule_values[i] - weighted_mean
                        
                        # Compute hessian
                        z = risk_set_rules * risk_set_scores[:, np.newaxis]
                        weighted_cov = np.dot(z.T, risk_set_rules) / np.sum(risk_set_scores)
                        weighted_mean_outer = np.outer(weighted_mean, weighted_mean)
                        hessian -= (weighted_cov - weighted_mean_outer)
                
                # Add L1 and L2 regularization
                gradient -= self.alpha * (self.l1_ratio * np.sign(weights) + 
                                        (1 - self.l1_ratio) * weights)
                hessian -= self.alpha * (1 - self.l1_ratio) * np.eye(n_rules)
                
                # Update weights
                try:
                    update = np.linalg.solve(hessian, gradient)
                    weights_new = weights - update
                    
                    # Check convergence
                    if np.max(np.abs(weights_new - weights)) < tol:
                        weights = weights_new
                        break
                        
                    weights = weights_new
                except np.linalg.LinAlgError:
                    # If matrix is singular, use gradient descent
                    weights += 0.01 * gradient
            
            self.rule_weights_ = weights
            self._compute_baseline()
        
        elif self.model_type in ['weibull', 'exponential']:
            self._fit_parametric(rule_values, y)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _fit_parametric(self, rule_values: np.ndarray, y: Survival) -> None:
        """
        Fit parametric survival model
        
        Parameters
        ----------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations
        y : Survival
            Survival data
        """
        def negative_log_likelihood(params):
            # Split parameters into rule weights and distribution parameters
            rule_weights = params[:-2]
            if self.model_type == 'weibull':
                scale = np.exp(params[-2])
                shape = np.exp(params[-1])
            else:  # exponential
                scale = np.exp(params[-1])
                shape = 1.0
            
            # Compute risk scores
            risk_scores = np.exp(np.dot(rule_values, rule_weights))
            
            # Compute log-likelihood
            if self.model_type == 'weibull':
                log_hazard = np.log(shape) - np.log(scale) + \
                            (shape - 1) * np.log(y.time/scale) + \
                            np.log(risk_scores)
                log_survival = -((y.time/scale) ** shape) * risk_scores
            else:  # exponential
                log_hazard = -np.log(scale) + np.log(risk_scores)
                log_survival = -(y.time/scale) * risk_scores
            
            # Compute negative log-likelihood
            nll = -np.sum(y.event * log_hazard + log_survival)
            return nll
        
        # Initialize parameters
        n_rules = rule_values.shape[1]
        initial_params = np.zeros(n_rules + 2)  # rule weights + distribution parameters
        
        # Optimize parameters
        result = minimize(negative_log_likelihood, initial_params,
                        method='L-BFGS-B')
        
        # Store results
        self.rule_weights_ = result.x[:-2]
        if self.model_type == 'weibull':
            self.parametric_params_ = {
                'scale': np.exp(result.x[-2]),
                'shape': np.exp(result.x[-1])
            }
        else:  # exponential
            self.parametric_params_ = {
                'scale': np.exp(result.x[-1])
            }
    
    def _compute_baseline(self) -> None:
        """Compute baseline hazard and survival functions"""
        if self.model_type == 'cox':
            # Compute risk scores
            risk_scores = np.exp(np.dot(self._evaluate_rules(self._X),
                                      self.rule_weights_))
            
            # Sort by time
            sort_idx = np.argsort(self._y.time)
            times = self._y.time[sort_idx]
            events = self._y.event[sort_idx]
            risk_scores = risk_scores[sort_idx]
            
            # Compute baseline hazard
            unique_times = np.unique(times[events == 1])
            baseline_hazard = np.zeros_like(unique_times)
            
            for i, t in enumerate(unique_times):
                at_risk = times >= t
                events_at_t = (times == t) & (events == 1)
                baseline_hazard[i] = np.sum(events_at_t) / np.sum(risk_scores[at_risk])
            
            self.baseline_hazard_ = (unique_times, baseline_hazard)
            
            # Compute baseline survival
            cumulative_hazard = np.cumsum(baseline_hazard)
            self.baseline_survival_ = np.exp(-cumulative_hazard)
        else:
            # For parametric models, baseline is determined by distribution parameters
            self.baseline_hazard_ = None
            self.baseline_survival_ = None
    
    def predict_survival(self, X: Union[np.ndarray, pd.DataFrame],
                        times: np.ndarray) -> np.ndarray:
        """
        Predict survival probabilities
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict survival
            
        Returns
        -------
        survival : array-like of shape (n_samples, n_times)
            Predicted survival probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(self._evaluate_rules(X),
                                  self.rule_weights_))
        
        # Compute survival probabilities
        survival = np.zeros((X.shape[0], len(times)))
        
        if self.model_type == 'cox':
            for i, t in enumerate(times):
                hazard_at_t = np.interp(t, self.baseline_hazard_[0],
                                      self.baseline_hazard_[1],
                                      left=0, right=0)
                survival[:, i] = np.exp(-hazard_at_t * risk_scores)
        elif self.model_type == 'weibull':
            scale = self.parametric_params_['scale']
            shape = self.parametric_params_['shape']
            for i, t in enumerate(times):
                survival[:, i] = np.exp(-((t/scale) ** shape) * risk_scores)
        else:  # exponential
            scale = self.parametric_params_['scale']
            for i, t in enumerate(times):
                survival[:, i] = np.exp(-(t/scale) * risk_scores)
        
        return survival
    
    def predict_hazard(self, X: Union[np.ndarray, pd.DataFrame],
                      times: np.ndarray) -> np.ndarray:
        """
        Predict hazard rates
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict hazard
            
        Returns
        -------
        hazard : array-like of shape (n_samples, n_times)
            Predicted hazard rates
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(self._evaluate_rules(X),
                                  self.rule_weights_))
        
        # Compute hazard rates
        hazard = np.zeros((X.shape[0], len(times)))
        
        if self.model_type == 'cox':
            for i, t in enumerate(times):
                hazard[:, i] = np.interp(t, self.baseline_hazard_[0],
                                       self.baseline_hazard_[1],
                                       left=0, right=0) * risk_scores
        elif self.model_type == 'weibull':
            scale = self.parametric_params_['scale']
            shape = self.parametric_params_['shape']
            for i, t in enumerate(times):
                hazard[:, i] = (shape/scale) * ((t/scale) ** (shape-1)) * risk_scores
        else:  # exponential
            scale = self.parametric_params_['scale']
            for i, t in enumerate(times):
                hazard[:, i] = (1/scale) * risk_scores
        
        return hazard
    
    def _compute_feature_importances(self) -> None:
        """Compute feature importances based on rule weights"""
        if isinstance(self._X, pd.DataFrame):
            feature_names = self._X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(self._X.shape[1])]
        
        feature_importances = np.zeros(len(feature_names))
        for i, feature_indices in enumerate(self.rule_features_):
            # Filter out invalid feature indices
            valid_indices = [idx for idx in feature_indices if idx < len(feature_names)]
            for idx in valid_indices:
                feature_importances[idx] += abs(self.rule_weights_[i])
        
        # Normalize feature importances
        total_importance = np.sum(feature_importances)
        if total_importance > 0:
            feature_importances /= total_importance
        else:
            # If all weights are zero, assign equal importance
            feature_importances = np.ones_like(feature_importances) / len(feature_importances)
        
        self.feature_importances_ = feature_importances 

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores for each sample
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
            
        Returns
        -------
        risk_scores : array-like of shape (n_samples,)
            Predicted risk scores
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(self._evaluate_rules(X),
                                  self.rule_weights_))
        
        return risk_scores

    def predict_cumulative_hazard(self, X: Union[np.ndarray, pd.DataFrame],
                                times: np.ndarray) -> np.ndarray:
        """
        Predict cumulative hazard function
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict cumulative hazard
            
        Returns
        -------
        cumulative_hazard : array-like of shape (n_samples, n_times)
            Predicted cumulative hazard
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(self._evaluate_rules(X),
                                  self.rule_weights_))
        
        # Compute cumulative hazard
        cumulative_hazard = np.zeros((X.shape[0], len(times)))
        
        if self.model_type == 'cox':
            for i, t in enumerate(times):
                hazard_at_t = np.interp(t, self.baseline_hazard_[0],
                                      self.baseline_hazard_[1],
                                      left=0, right=0)
                cumulative_hazard[:, i] = hazard_at_t * risk_scores
        elif self.model_type == 'weibull':
            scale = self.parametric_params_['scale']
            shape = self.parametric_params_['shape']
            for i, t in enumerate(times):
                cumulative_hazard[:, i] = ((t/scale) ** shape) * risk_scores
        else:  # exponential
            scale = self.parametric_params_['scale']
            for i, t in enumerate(times):
                cumulative_hazard[:, i] = (t/scale) * risk_scores
        
        return cumulative_hazard 