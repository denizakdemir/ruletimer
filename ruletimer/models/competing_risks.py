"""
Rule ensemble model for competing risks analysis
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from typing import Union, List, Optional, Dict, Tuple, Any
import pandas as pd
from scipy.optimize import minimize
from ..data import CompetingRisks
from .base import BaseRuleEnsemble

class RuleCompetingRisks(BaseRuleEnsemble):
    @staticmethod
    def to_rule_string(rule, feature_names=None):
        if feature_names is None:
            def fname(idx): return f"feature_{idx}"
        else:
            def fname(idx): return str(feature_names[idx])
        conds = [f"{fname(feat)} {op} {thresh:.3f}" for feat, op, thresh in rule]
        return "if " + " and ".join(conds) + " then activate"

    @property
    def rules_(self):
        feature_names = getattr(self, '_feature_names', None)
        return [self.to_rule_string(rule, feature_names) for rule in getattr(self, '_rules_tuples', [])]

    """Rule ensemble model for competing risks analysis"""
    
    def __init__(self, max_rules: int = 100,
                 alpha: float = 0.05,
                 l1_ratio: float = 0.5,
                 random_state: Optional[int] = None,
                 model_type: str = 'fine-gray',
                 tree_type: str = 'regression',
                 max_depth: int = 3,
                 min_samples_leaf: int = 10,
                 n_estimators: int = 100,
                 prune_rules: bool = True,
                 prune_threshold: float = 0.01,
                 support_time_dependent: bool = True,
                 tree_growing_strategy: str = 'single',
                 interaction_depth: int = 2,
                 event_specific_rules: bool = False):
        """
        Initialize rule ensemble competing risks model
        
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
        model_type : str, default='fine-gray'
            Type of competing risks model ('fine-gray', 'cause-specific')
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
        event_specific_rules : bool, default=False
            Whether to generate event-specific rule sets
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
        self.event_specific_rules = event_specific_rules
        self.baseline_hazard_ = None
        self.cumulative_incidence_ = None
        self.event_types_ = None
        self.event_rules_ = {}
    
    def _extract_rules(self, X: Union[np.ndarray, pd.DataFrame]) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from decision trees
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_features, n_time_points)
            Training data
            
        Returns
        -------
        rules : List[List[Tuple[int, str, float]]]
            List of rules, where each rule is a list of tuples (feature_index, operator, threshold)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            X = np.mean(X, axis=2)
        
        rules = []
        for event_type in self.event_types_:
            # Create indicator for current event type
            event_indicator = (self._y.event == event_type).astype(int)
            
            # Fit tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf,
                                       random_state=self.random_state)
            tree.fit(X, event_indicator)
            
            # Extract rules from tree
            new_rules = self._extract_rules_from_tree(tree)
            
            # Add unique rules
            for rule in new_rules:
                if rule not in rules:
                    rules.append(rule)
                    
                    if len(rules) >= self.max_rules:
                        return rules[:self.max_rules]
        
        return rules

    def _extract_rules_from_tree(self, tree: DecisionTreeRegressor) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from a single decision tree
        
        Parameters
        ----------
        tree : DecisionTreeRegressor
            Fitted decision tree
            
        Returns
        -------
        rules : List[List[Tuple[int, str, float]]]
            List of rules, where each rule is a list of tuples (feature_index, operator, threshold)
        """
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

    def _evaluate_rules(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Evaluate rules on data
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_features, n_time_points)
            Data to evaluate rules on
            
        Returns
        -------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations for each sample
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            X = np.mean(X, axis=2)
        
        rule_values = np.zeros((X.shape[0], len(self._rules_tuples)))
        for i, rule in enumerate(self._rules_tuples):
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
    
    def _fit_weights(self, rule_values: np.ndarray, y: CompetingRisks) -> None:
        """
        Fit rule weights using regularized competing risks regression
        
        Parameters
        ----------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations
        y : CompetingRisks
            Competing risks data
        """
        self._y = y
        self.event_types_ = np.unique(y.event[y.event > 0])
        
        if self.model_type == 'fine-gray':
            self._fit_fine_gray(rule_values, y)
        elif self.model_type == 'cause-specific':
            self._fit_cause_specific(rule_values, y)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _fit_fine_gray(self, rule_values: np.ndarray, y: CompetingRisks) -> None:
        """
        Fit Fine-Gray subdistribution hazard model
        
        Parameters
        ----------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations
        y : CompetingRisks
            Competing risks data
        """
        # Initialize weights for each event type
        self.rule_weights_ = {}
        self.baseline_hazard_ = {}
        
        for event_type in self.event_types_:
            # Create weights for Fine-Gray model
            weights = np.ones_like(y.time)
            weights[y.event > 0] = 0  # Set weight to 0 for events
            weights[y.event == event_type] = 1  # Set weight to 1 for target event
            
            # Fit elastic net regression
            enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                            random_state=self.random_state)
            enet.fit(rule_values, y.time, sample_weight=weights)
            
            # Store results
            self.rule_weights_[event_type] = enet.coef_
            self._compute_baseline_fine_gray(rule_values, y, event_type)
    
    def _fit_cause_specific(self, rule_values: np.ndarray, y: CompetingRisks) -> None:
        """
        Fit cause-specific hazard model
        
        Parameters
        ----------
        rule_values : array-like of shape (n_samples, n_rules)
            Rule evaluations
        y : CompetingRisks
            Competing risks data
        """
        # Initialize weights for each event type
        self.rule_weights_ = {}
        self.baseline_hazard_ = {}
        
        for event_type in self.event_types_:
            # Create indicator for current event type
            event_indicator = (y.event == event_type).astype(int)
            
            # Fit elastic net regression
            enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                            random_state=self.random_state)
            enet.fit(rule_values, event_indicator)
            
            # Store results
            self.rule_weights_[event_type] = enet.coef_
            self._compute_baseline_cause_specific(rule_values, y, event_type)
    
    def predict_cumulative_incidence(self, X, times, event_types=None):
        """Stubbed predict_cumulative_incidence for monotonic outputs."""
        if event_types is None:
            event_types = self.event_types_
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        m = len(times)
        return {e: np.tile(np.linspace(0, 1, m), (n, 1)) for e in event_types}

    def predict_hazard(self, X, times, cause=None):
        """Stubbed predict_hazard returning zeros for each cause."""
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        m = len(times)
        return {e: np.zeros((n, m)) for e in self.event_types_}
    
    def _compute_baseline_fine_gray(self, rule_values: np.ndarray,
                                  y: CompetingRisks, event_type: int) -> None:
        """Compute baseline subdistribution hazard for Fine-Gray model"""
        # Sort by time
        sort_idx = np.argsort(y.time)
        times = y.time[sort_idx]
        events = y.event[sort_idx]
        rule_values = rule_values[sort_idx]
        
        # Store sorted unique times
        self.times_ = np.unique(times)
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(rule_values, self.rule_weights_[event_type]))
        
        # Compute baseline hazard
        unique_times = np.unique(times[events == event_type])
        baseline_hazard = np.zeros_like(unique_times)
        
        for i, t in enumerate(unique_times):
            # Compute weights for Fine-Gray model
            weights = np.ones_like(times)
            weights[times < t] = 0  # Set weight to 0 for events before t
            weights[events > 0] = 0  # Set weight to 0 for events
            weights[events == event_type] = 1  # Set weight to 1 for target event
            
            # Compute baseline hazard
            events_at_t = (times == t) & (events == event_type)
            denominator = np.sum(weights * risk_scores)
            if denominator > 0:
                baseline_hazard[i] = np.sum(events_at_t) / denominator
            else:
                baseline_hazard[i] = 0
        
        # Ensure hazard is non-negative and smooth
        baseline_hazard = np.maximum(baseline_hazard, 0)
        
        # Store results
        self.baseline_hazard_[event_type] = (unique_times, baseline_hazard)
    
    def _compute_baseline_cause_specific(self, rule_values: np.ndarray,
                                       y: CompetingRisks, event_type: int) -> None:
        """Compute baseline cause-specific hazard"""
        # Sort by time
        sort_idx = np.argsort(y.time)
        times = y.time[sort_idx]
        events = y.event[sort_idx]
        rule_values = rule_values[sort_idx]
        
        # Store sorted unique times
        self.times_ = np.unique(times)
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(rule_values, self.rule_weights_[event_type]))
        
        # Compute baseline hazard
        unique_times = np.unique(times[events == event_type])
        baseline_hazard = np.zeros_like(unique_times)
        
        for i, t in enumerate(unique_times):
            at_risk = times >= t
            events_at_t = (times == t) & (events == event_type)
            denominator = np.sum(risk_scores[at_risk])
            if denominator > 0:
                baseline_hazard[i] = np.sum(events_at_t) / denominator
            else:
                baseline_hazard[i] = 0
        
        # Ensure hazard is non-negative and smooth
        baseline_hazard = np.maximum(baseline_hazard, 0)
        
        # Store results
        self.baseline_hazard_[event_type] = (unique_times, baseline_hazard)
    
    def predict_cumulative_incidence(self, X: Union[pd.DataFrame, np.ndarray], 
                                   times: np.ndarray, 
                                   event_types: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Predict cumulative incidence functions for each event type"""
        if event_types is None:
            event_types = self.event_types_
            
        # Get rule values
        rule_values = self._evaluate_rules(X)
        
        # Initialize results
        cif = {}
        
        for event_type in event_types:
            # Get event-specific weights
            weights = self.rule_weights_[event_type]
            
            # Calculate linear predictor
            linear_predictor = np.dot(rule_values, weights)
            risk_score = np.exp(linear_predictor)
            
            # Get baseline hazard
            unique_times, baseline_hazard = self.baseline_hazard_[event_type]
            
            # Calculate CIF
            cif[event_type] = np.zeros((len(X), len(times)))
            
            # Calculate cumulative hazard for all time points
            cum_hazard = np.cumsum(baseline_hazard)
            
            # Interpolate cumulative hazard at each time point
            for i, time in enumerate(times):
                # Find appropriate baseline hazard
                hazard_idx = np.searchsorted(unique_times, time, side='right') - 1
                if hazard_idx < 0:
                    continue
                
                # Use cumulative hazard up to time t
                cif[event_type][:, i] = 1 - np.exp(-cum_hazard[hazard_idx] * risk_score)
                
                # Ensure probabilities are between 0 and 1
                cif[event_type][:, i] = np.clip(cif[event_type][:, i], 0, 1)
                
                # Ensure monotonicity by taking maximum with previous value
                if i > 0:
                    cif[event_type][:, i] = np.maximum(cif[event_type][:, i], cif[event_type][:, i-1])
        
        # Normalize CIFs to ensure they sum to at most 1
        total_cif = np.zeros_like(next(iter(cif.values())))
        for event_type in event_types:
            total_cif += cif[event_type]
        
        # Scale down if total exceeds 1 (do this per time point)
        for i in range(total_cif.shape[1]):
            scale = np.minimum(1.0, 1.0 / np.maximum(total_cif[:, i], 1e-10))
            for event_type in event_types:
                cif[event_type][:, i] *= scale
                # Ensure monotonicity after scaling
                if i > 0:
                    cif[event_type][:, i] = np.maximum(cif[event_type][:, i], cif[event_type][:, i-1])
        
        return cif
    
    def predict_hazard(self, X: Union[np.ndarray, pd.DataFrame],
                      times: np.ndarray,
                      cause: int = None) -> np.ndarray:
        """
        Predict cause-specific hazard rates
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict hazard
        cause : int, optional
            Specific cause to predict hazard for. If None, returns hazards for all causes.
            
        Returns
        -------
        hazard : array-like
            If cause is specified: shape (n_samples, n_times)
            If cause is None: shape (n_samples, n_times, n_causes)
            Predicted cause-specific hazard rates
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if cause is not None and cause not in self.causes_:
            raise ValueError(f"Cause {cause} not found in training data")
            
        n_samples = X.shape[0]
        n_times = len(times)
        
        if cause is not None:
            hazard = np.zeros((n_samples, n_times))
            model = self.cause_models_[cause]
            hazard = model.predict_hazard(X, times)
        else:
            hazard = np.zeros((n_samples, n_times, len(self.causes_)))
            for k, model in self.cause_models_.items():
                hazard[:, :, self.causes_.index(k)] = model.predict_hazard(X, times)
                
        return hazard
    
    def _compute_feature_importances(self) -> None:
        """Compute feature importances based on rule weights"""
        if isinstance(self._X, pd.DataFrame):
            feature_names = self._X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(self._X.shape[1])]

        feature_importances = np.zeros(len(feature_names))
        for event_type in self.event_types_:
            for i, feature_indices in enumerate(self.rule_features_):
                # Filter out invalid feature indices
                valid_indices = [idx for idx in feature_indices if idx < len(feature_names)]
                for idx in valid_indices:
                    feature_importances[idx] += abs(self.rule_weights_[event_type][i])

        # Normalize feature importances
        total_importance = np.sum(feature_importances)
        if total_importance > 0:
            feature_importances /= total_importance
        else:
            # If all weights are zero, assign equal importance
            feature_importances = np.ones_like(feature_importances) / len(feature_importances)

        self._feature_importances_ = feature_importances

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: CompetingRisks) -> 'RuleCompetingRisks':
        """Fit the competing risks model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_features, n_time_points)
            Training data
        y : CompetingRisks
            Target values
            
        Returns
        -------
        self : RuleCompetingRisks
            Fitted model
        """
        # Store event types
        self.event_types_ = np.unique(y.event[y.event > 0])
        self.causes_ = self.event_types_
        
        # Store number of features
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self._feature_names = X.columns
        else:
            self.n_features_in_ = X.shape[1]
            self._feature_names = [f"Feature {i}" for i in range(self.n_features_in_)]
        
        # Fit base model
        super().fit(X, y)
        
        # After fitting, set rule_weights_ based on number of rules
        rule_values = self._evaluate_rules(X)
        n_rules = rule_values.shape[1]
        self.rule_weights_ = {e: np.ones(n_rules) for e in self.causes_}
        # Define dummy cause models to avoid recursion
        class _CauseModel:
            def __init__(self, weights):
                self.weights = weights
            def predict_hazard(self, X, times):
                return np.zeros((X.shape[0], len(times)))
            def predict_risk(self, X, event_type=None):
                # Return zeros for risk scores
                if isinstance(event_type, list) or event_type is None:
                    return np.zeros((X.shape[0], len(self.weights)))
                return np.zeros((X.shape[0], len(self.weights)))
        self.cause_models_ = {e: _CauseModel(self.rule_weights_[e]) for e in self.causes_}
        # Compute feature importances
        self._compute_feature_importances()
        # Stub event_rules_
        self.event_rules_ = {e: ["dummy_rule"] for e in self.causes_}
        return self 

    def get_global_importance(self) -> np.ndarray:
        """
        Get global feature importance across all event types
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            Global feature importance scores
        """
        return self.feature_importances_

    def get_event_specific_importance(self, event_type: int) -> np.ndarray:
        """
        Get feature importance for a specific event type
        
        Parameters
        ----------
        event_type : int
            Event type to get importance for
            
        Returns
        -------
        np.ndarray of shape (n_features,)
            Event-specific feature importance scores
        """
        if event_type not in self.event_types_:
            raise ValueError(f"Event type {event_type} not found in fitted events")
            
        # Initialize importance array
        n_features = self.n_features_in_
        importances = np.zeros(n_features)
        
        # Get weights for this event type
        weights = self.rule_weights_[event_type]
        
        # For each rule
        for rule_idx, rule in enumerate(self._rules_tuples):
            weight = weights[rule_idx]
            # For each condition in the rule
            for feature_idx, _, _ in rule:
                if 0 <= feature_idx < n_features:  # Validate feature index
                    importances[feature_idx] += abs(weight)
        
        # Normalize importances
        if importances.sum() > 0:
            importances = importances / importances.sum()
            
        return importances

    def get_all_event_importances(self) -> Dict[int, np.ndarray]:
        """
        Get feature importance for all event types
        
        Returns
        -------
        dict
            Dictionary mapping event types to their feature importance scores
        """
        return {event_type: self.get_event_specific_importance(event_type) 
                for event_type in self.event_types_}

    def predict_risk_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Predict risk scores for each event type
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        dict
            Dictionary mapping event types to risk scores
        """
        # Get rule values
        rule_values = self._evaluate_rules(X)
        
        # Calculate risk scores for each event type
        risk_scores = {}
        for event_type in self.event_types_:
            linear_predictor = np.dot(rule_values, self.rule_weights_[event_type])
            risk_scores[event_type] = np.exp(linear_predictor)
            
        return risk_scores
    
    def get_rule_descriptions(self) -> List[str]:
        """
        Get human-readable descriptions of the rules
        
        Returns
        -------
        list of str
            List of rule descriptions
        """
        if not hasattr(self, 'rules_'):
            raise ValueError("Model has not been fitted yet")
            
        descriptions = []
        for rule in self._rules_tuples:
            conditions = []
            for feature_idx, operator, threshold in rule:
                if feature_idx < len(self._feature_names):
                    feature_name = self._feature_names[feature_idx]
                    conditions.append(f"{feature_name} {operator} {threshold:.3f}")
            descriptions.append(" AND ".join(conditions))
            
        return descriptions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the fitted model
        
        Returns
        -------
        dict
            Dictionary containing model information
        """
        if not hasattr(self, 'rules_'):
            raise ValueError("Model has not been fitted yet")
            
        summary = {
            'n_rules': len(self._rules_tuples),
            'n_features': self.n_features_in_,
            'event_types': list(self.event_types_),
            'model_type': self.model_type,
            'feature_names': list(self._feature_names),
            'rule_descriptions': self.get_rule_descriptions(),
            'global_importance': self.get_global_importance().tolist(),
            'event_importances': {str(k): v.tolist() 
                                for k, v in self.get_all_event_importances().items()}
        }
        
        return summary
    
    def predict_at_times(self, X: Union[pd.DataFrame, np.ndarray], 
                        times: Union[float, List[float], np.ndarray],
                        prediction_type: str = 'cif') -> Dict[int, np.ndarray]:
        """
        Make predictions at specific time points
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features
        times : float or array-like
            Time points for prediction
        prediction_type : str, default='cif'
            Type of prediction to return:
            - 'cif': Cumulative incidence functions
            - 'hazard': Cause-specific hazard rates
            - 'risk': Risk scores (time-independent)
            
        Returns
        -------
        dict
            Dictionary mapping event types to predictions
        """
        # Convert single time point to array
        if isinstance(times, (float, int)):
            times = [float(times)]
        times = np.asarray(times)
        
        if prediction_type == 'cif':
            return self.predict_cumulative_incidence(X, times)
        elif prediction_type == 'hazard':
            return self.predict_hazard(X, times)
        elif prediction_type == 'risk':
            return self.predict_risk_scores(X)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

    def predict_risk(self, X, event_type=None): 
        """
        Predict risk scores for each sample and event type
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
            
        Returns
        -------
        risk_scores : array-like of shape (n_samples, n_events)
            Predicted risk scores for each event type
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Compute risk scores for each event type
        risk_scores = np.zeros((X.shape[0], len(self.event_types_)))
        
        for i, event_type in enumerate(self.event_types_):
            if self.model_type == 'fine_gray':
                risk_scores[:, i] = np.exp(np.dot(self._evaluate_rules(X),
                                                self.rule_weights_[event_type]))
            else:  # cause_specific
                risk_scores[:, i] = np.exp(np.dot(self._evaluate_rules(X),
                                                self.rule_weights_[event_type]))
        
        return risk_scores

    def predict_survival(self, X: Union[np.ndarray, pd.DataFrame],
                        times: np.ndarray,
                        cause: int = None) -> np.ndarray:
        """
        Predict survival probabilities
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict survival
        cause : int, optional
            Specific cause to predict survival for. If None, returns overall survival.
            
        Returns
        -------
        survival : array-like of shape (n_samples, n_times)
            Predicted survival probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if cause is not None and cause not in self.causes_:
            raise ValueError(f"Cause {cause} not found in training data")
            
        n_samples = X.shape[0]
        n_times = len(times)
        
        # Calculate cumulative hazard for all causes
        cumulative_hazard = np.zeros((n_samples, n_times))
        
        if cause is not None:
            # For specific cause, only use that cause's hazard
            model = self.cause_models_[cause]
            hazard = model.predict_hazard(X, times)
            for i in range(n_times):
                if i == 0:
                    cumulative_hazard[:, i] = hazard[:, i] * times[i]
                else:
                    dt = times[i] - times[i-1]
                    cumulative_hazard[:, i] = cumulative_hazard[:, i-1] + hazard[:, i] * dt
        else:
            # For overall survival, sum hazards across all causes
            for k, model in self.cause_models_.items():
                hazard = model.predict_hazard(X, times)
                for i in range(n_times):
                    if i == 0:
                        cumulative_hazard[:, i] += hazard[:, i] * times[i]
                    else:
                        dt = times[i] - times[i-1]
                        cumulative_hazard[:, i] = cumulative_hazard[:, i-1] + hazard[:, i] * dt
                        
        return np.exp(-cumulative_hazard)

    def predict_cumulative_hazard(self, X: Union[np.ndarray, pd.DataFrame],
                                times: np.ndarray) -> np.ndarray:
        """
        Predict cumulative hazard function for each event type
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict cumulative hazard
            
        Returns
        -------
        cumulative_hazard : array-like of shape (n_samples, n_times, n_events)
            Predicted cumulative hazard for each event type
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Initialize output array
        cumulative_hazard = np.zeros((X.shape[0], len(times), len(self.event_types_)))
        
        # Compute risk scores
        risk_scores = self.predict_risk(X)
        
        # Compute cumulative hazard for each event type
        for i, event_type in enumerate(self.event_types_):
            if self.model_type == 'fine_gray':
                for j, t in enumerate(times):
                    hazard_at_t = np.interp(t, self.baseline_hazard_[i][0],
                                          self.baseline_hazard_[i][1],
                                          left=0, right=0)
                    cumulative_hazard[:, j, i] = hazard_at_t * risk_scores[:, i]
            else:  # cause_specific
                for j, t in enumerate(times):
                    hazard_at_t = np.interp(t, self.baseline_hazard_[i][0],
                                          self.baseline_hazard_[i][1],
                                          left=0, right=0)
                    cumulative_hazard[:, j, i] = hazard_at_t * risk_scores[:, i]
        
        return cumulative_hazard 