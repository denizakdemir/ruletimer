"""
Specialized competing risks model implemented as a multi-state model.
"""
from typing import Dict, List, Optional, Union
import numpy as np
from ruletimer.models.base_multi_state import BaseMultiStateModel
from ruletimer.data import CompetingRisks
from sklearn.linear_model import ElasticNet
from scipy.interpolate import interp1d

class RuleCompetingRisks(BaseMultiStateModel):
    """
    Competing risks model implemented as a special case of multi-state model.
    
    This model represents competing risks as a multi-state model with:
    - One initial state (0)
    - Multiple absorbing states (1, 2, ..., K) for each event type
    """
    
    def __init__(
        self,
        max_rules: int = 32,
        max_depth: int = 4,
        n_estimators: int = 200,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        hazard_method: str = "nelson-aalen",
        random_state: Optional[int] = None
    ):
        """
        Initialize competing risks model.
        
        Parameters
        ----------
        max_rules : int
            Maximum number of rules to generate
        max_depth : int
            Maximum depth of trees
        n_estimators : int
            Number of trees in random forest
        alpha : float
            L1 + L2 regularization strength
        l1_ratio : float
            L1 ratio for elastic net (1 = lasso, 0 = ridge)
        hazard_method : str
            Method for hazard estimation
        random_state : int, optional
            Random state for reproducibility
        """
        # Create states list with initial state and event states
        event_types = ["Event1", "Event2"]  # Default event types
        states = ["Initial"] + event_types
        
        # Create transitions from initial state to each event state
        transitions = [(0, i+1) for i in range(len(event_types))]
        
        # Initialize base class
        super().__init__(
            states=states,
            transitions=transitions,
            hazard_method=hazard_method
        )
        
        self.event_types = event_types
        self.event_type_to_state = {
            event: i+1 for i, event in enumerate(event_types)
        }
        self.max_rules = max_rules
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.support_time_dependent = False  # Currently not supported

        # Initialize dictionaries to store rules and weights for each transition
        self.transition_rules_ = {}
        self.rule_weights_ = {}
        self.rules_ = None  # Temporary storage for current rules
    
    def fit(self, X: np.ndarray, y: CompetingRisks) -> 'RuleCompetingRisks':
        """
        Fit the competing risks model.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : CompetingRisks
            Target competing risks data
            
        Returns
        -------
        self : RuleCompetingRisks
            Fitted model
        """
        if not isinstance(y, CompetingRisks):
            raise ValueError("y must be a CompetingRisks object")
            
        # Prepare transition-specific data
        transition_times = {}
        transition_events = {}
        
        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)
            
            # Event indicator for this transition
            is_event = y.event == state
            
            transition_times[transition] = y.time
            transition_events[transition] = is_event
        
        # Estimate baseline hazards
        self._estimate_baseline_hazards(transition_times, transition_events)
        
        # Fit transition-specific models
        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)
            self._fit_transition_model(X, y, transition)
        
        self.is_fitted_ = True
        return self
    
    def predict_transition_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict transition-specific hazard.
        
        Parameters
        ----------
        X : array-like
            Covariate values
        times : array-like
            Times at which to predict hazard
        from_state : str or int
            Starting state
        to_state : str or int
            Target state
            
        Returns
        -------
        np.ndarray
            Predicted hazard values of shape (n_samples, n_times)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Convert states to internal indices
        from_idx = self.state_manager.to_internal_index(from_state)
        to_idx = self.state_manager.to_internal_index(to_state)
        transition = (from_idx, to_idx)
        
        if transition not in self.transition_models_:
            raise ValueError(f"No model for transition {transition}")
            
        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            # For time-dependent covariates, we'll use the values at each time point
            # This is a simple approach - more sophisticated methods could be implemented
            X = X[:, :, 0]  # Use features at first time point for now
            # TODO: Implement proper handling of time-dependent covariates
        
        # Evaluate rules on input data
        rule_matrix = self._evaluate_rules(X)
        
        # Get baseline hazard for these times
        baseline_times, baseline_values = self.baseline_hazards_[transition]
        
        # If no events were observed for this transition, return zeros
        if len(baseline_times) == 0:
            return np.zeros((X.shape[0], len(times)))
        
        # Create interpolation function for baseline hazard
        baseline_interp = interp1d(
            baseline_times,
            baseline_values,
            kind='linear',
            bounds_error=False,
            fill_value=(baseline_values[0], baseline_values[-1])
        )
        
        # Get interpolated baseline hazard values
        interpolated_baseline = baseline_interp(times)
        
        # Get relative hazard from elastic net model
        relative_hazard = np.exp(self.transition_models_[transition].predict(rule_matrix))
        
        # Compute final hazard by multiplying baseline and relative hazard
        # Expand relative hazard to match shape of times
        relative_hazard = relative_hazard.reshape(-1, 1)  # Shape: (n_samples, 1)
        interpolated_baseline = interpolated_baseline.reshape(1, -1)  # Shape: (1, n_times)
        
        return relative_hazard * interpolated_baseline
    
    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        event_type: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict cumulative incidence functions.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict CIF
        event_type : str, optional
            Specific event type to predict for. If None, returns all event types.
            
        Returns
        -------
        Union[np.ndarray, Dict[str, np.ndarray]]
            If event_type is specified: array of CIF values
            If event_type is None: dict mapping event types to CIF values
        """
        if event_type is not None:
            if event_type not in self.event_types:
                raise ValueError(f"Unknown event type: {event_type}")
            state = self.event_type_to_state[event_type]
            return self.predict_transition_probability(X, times, 0, state)
        
        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            # For time-dependent covariates, we'll use the values at each time point
            # This is a simple approach - more sophisticated methods could be implemented
            X = X[:, :, 0]  # Use features at first time point for now
            # TODO: Implement proper handling of time-dependent covariates
        
        return {
            event_type: self.predict_cumulative_incidence(X, times, event_type)
            for event_type in self.event_types
        }
    
    def predict_cause_specific_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        event_type: Union[str, int]
    ) -> np.ndarray:
        """
        Predict cause-specific hazard for a specific event type.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict hazard
        event_type : str or int
            Event type to predict for
            
        Returns
        -------
        np.ndarray
            Predicted cause-specific hazard values
        """
        # Convert event type to state if it's a string
        if isinstance(event_type, str):
            if event_type not in self.event_types:
                raise ValueError(f"Unknown event type: {event_type}")
            state = self.event_type_to_state[event_type]
        else:
            state = event_type
            
        return self.predict_transition_hazard(X, times, 0, state)
    
    def predict_hazard(self, X: np.ndarray, times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict cause-specific hazard functions.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict hazard
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping event types to hazard arrays of shape (n_samples, n_times)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        hazard = {}
        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)
            
            if transition not in self.transition_models_:
                raise ValueError(f"No model for transition {transition}")
            
            # Handle time-dependent covariates
            if self.support_time_dependent and X.ndim == 3:
                # For time-dependent covariates, we'll use the values at each time point
                # This is a simple approach - more sophisticated methods could be implemented
                X = X[:, :, 0]  # Use features at first time point for now
                # TODO: Implement proper handling of time-dependent covariates
            
            # Evaluate rules on input data
            rule_matrix = self._evaluate_rules(X)
            
            # Get baseline hazard for these times using interpolation
            baseline_times, baseline_values = self.baseline_hazards_[transition]
            if len(baseline_times) == 0:
                # No events observed for this transition
                hazard[event_type] = np.zeros((len(X), len(times)))
                continue
            
            # Create interpolation function
            f = interp1d(baseline_times, baseline_values, bounds_error=False, fill_value=(baseline_values[0], baseline_values[-1]))
            baseline_hazard = f(times)
            
            # Get relative hazard from elastic net model
            relative_hazard = self.transition_models_[transition].predict(rule_matrix)
            
            # Compute hazard for each time point
            hazard[event_type] = np.zeros((len(X), len(times)))
            for i in range(len(X)):
                hazard[event_type][i] = baseline_hazard * np.exp(relative_hazard[i])
        
        return hazard
    
    def _fit_transition_model(
        self,
        X: np.ndarray,
        y: CompetingRisks,
        transition: tuple
    ) -> None:
        """
        Fit transition-specific model using random forest for rule generation
        and elastic net for fitting.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : CompetingRisks
            Target competing risks data
        transition : tuple
            Transition to fit model for
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Initialize random forest
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Get event indicator for this transition
        is_event = y.event == transition[1]
        
        # Fit forest on survival times
        forest.fit(X, y.time)
        
        # Extract rules from forest
        self.rules_ = self._extract_rules_from_forest(forest)
        
        # Evaluate rules on data
        rule_matrix = self._evaluate_rules(X)
        
        # Fit elastic net on rule matrix
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )
        model.fit(rule_matrix, y.time)
        
        # Store rules and weights for this transition
        self.transition_rules_[transition] = self.rules_
        self.rule_weights_[transition] = model.coef_
        self.transition_models_[transition] = model
        
        # Compute feature importances
        self.feature_importances_ = self._compute_feature_importances()
        
    def _extract_rules_from_forest(self, forest):
        """Extract rules from random forest."""
        rules = []
        for tree in forest.estimators_:
            rules.extend(self._extract_rules_from_tree(tree.tree_))
        return rules[:self.max_rules]
        
    def _extract_rules_from_tree(self, tree):
        """Extract rules from a single tree."""
        rules = []
        
        def recurse(node, rule):
            if tree.feature[node] != -2:  # Not a leaf
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                
                # Left branch: feature <= threshold
                left_rule = rule + [(feature, "<=", threshold)]
                recurse(tree.children_left[node], left_rule)
                
                # Right branch: feature > threshold
                right_rule = rule + [(feature, ">", threshold)]
                recurse(tree.children_right[node], right_rule)
            else:
                rules.append(rule)
        
        recurse(0, [])
        return rules
        
    def _evaluate_rules(self, X):
        """Evaluate rules on data."""
        rule_matrix = np.zeros((X.shape[0], len(self.rules_)))
        for i, rule in enumerate(self.rules_):
            mask = np.ones(X.shape[0], dtype=bool)
            for feature, op, threshold in rule:
                if op == "<=":
                    mask &= (X[:, feature] <= threshold)
                else:  # op == ">"
                    mask &= (X[:, feature] > threshold)
            rule_matrix[:, i] = mask
        return rule_matrix
        
    def _compute_feature_importances(self):
        """Compute feature importances from rules."""
        # Get number of features from all rules
        n_features = 0
        for rules in self.transition_rules_.values():
            for rule in rules:
                for feature, _, _ in rule:
                    n_features = max(n_features, feature + 1)
                    
        importances = np.zeros(n_features)
        
        # Compute importances for each transition
        for transition in self.transition_rules_:
            rules = self.transition_rules_[transition]
            weights = self.rule_weights_[transition]
            
            for rule, weight in zip(rules, weights):
                for feature, _, _ in rule:
                    importances[feature] += abs(weight)
                    
        # Normalize importances
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
            
        return importances

    def get_feature_importances(self, event_type: Union[str, int]) -> np.ndarray:
        """
        Get feature importances for a specific event type.
        
        Parameters
        ----------
        event_type : str or int
            Event type to get feature importances for
            
        Returns
        -------
        np.ndarray
            Feature importance values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importances")
            
        # Convert event type to state if it's a string
        if isinstance(event_type, str):
            if event_type not in self.event_types:
                raise ValueError(f"Unknown event type: {event_type}")
            state = self.event_type_to_state[event_type]
        else:
            state = event_type
            
        transition = (0, state)
        if transition not in self.transition_rules_:
            raise ValueError(f"No rules found for event type {event_type}")
            
        # Get rules and weights for this transition
        rules = self.transition_rules_[transition]
        weights = self.rule_weights_[transition]
        
        # Get number of features from the first rule's first condition's feature index
        n_features = max(feature for rule in rules for feature, _, _ in rule) + 1 if rules else 0
        importances = np.zeros(n_features)
        
        # Compute feature importances
        for rule, weight in zip(rules, weights):
            for feature, _, _ in rule:
                importances[feature] += abs(weight)
                
        # Normalize importances
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
            
        return importances