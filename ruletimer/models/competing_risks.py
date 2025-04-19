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
        event_types: List[str],
        hazard_method: str = "nelson-aalen",
        model_type='cause-specific',
        tree_growing_strategy='single',
        max_rules=10,
        support_time_dependent=False,
        alpha=0.1,
        l1_ratio=0.5,
        random_state=None
    ):
        """
        Initialize competing risks model.
        
        Parameters
        ----------
        event_types : list of str
            List of event type names
        hazard_method : str
            Method for hazard estimation: "nelson-aalen" or "parametric"
        model_type : str, default='cause-specific'
            Type of competing risks model. Options:
            - 'cause-specific': Cause-specific hazard model
            - 'fine-gray': Fine-Gray subdistribution hazard model
        tree_growing_strategy : str, default='single'
            Strategy for generating rules. Options:
            - 'single': Single decision tree
            - 'forest': Random forest
            - 'interaction': Interaction trees
        max_rules : int, default=10
            Maximum number of rules to generate
        support_time_dependent : bool, default=False
            Whether to support time-dependent covariates
        alpha : float, default=0.1
            Constant that multiplies the penalty terms in elastic net
        l1_ratio : float, default=0.5
            The elastic net mixing parameter (0 <= l1_ratio <= 1)
        random_state : int, default=None
            Random state for reproducibility
        """
        # Create states list with initial state and event states
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
        self.model_type = model_type
        self.tree_growing_strategy = tree_growing_strategy
        self.max_rules = max_rules
        self.support_time_dependent = support_time_dependent
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state

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
        event_type: str
    ) -> np.ndarray:
        """
        Predict cause-specific hazard function.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict hazard
        event_type : str
            Event type to predict hazard for
            
        Returns
        -------
        np.ndarray
            Predicted cause-specific hazard values
        """
        if event_type not in self.event_types:
            raise ValueError(f"Unknown event type: {event_type}")
            
        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            # For time-dependent covariates, we'll use the values at each time point
            # This is a simple approach - more sophisticated methods could be implemented
            X = X[:, :, 0]  # Use features at first time point for now
            # TODO: Implement proper handling of time-dependent covariates
            
        state = self.event_type_to_state[event_type]
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
        Fit transition-specific model by generating rules and fitting weights.

        Parameters
        ----------
        X : array-like
            Training data
        y : CompetingRisks
            Target competing risks data
        transition : tuple
            Transition to fit model for (from_state, to_state)
        """
        # Get event type from transition
        _, to_state = transition
        event_type = to_state  # In competing risks, state index = event type

        # Create binary target for this event type
        y_binary = (y.event == event_type).astype(int)

        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            # For time-dependent covariates, we'll use the values at event times
            # This is a simple approach - more sophisticated methods could be implemented
            X = X[:, :, 0]  # Use features at first time point for now
            # TODO: Implement proper handling of time-dependent covariates

        # Generate rules based on tree growing strategy
        if self.tree_growing_strategy == 'single':
            self._generate_rules_single_tree(X, y_binary)
        elif self.tree_growing_strategy == 'forest':
            self._generate_rules_forest(X, y_binary)
        elif self.tree_growing_strategy == 'interaction':
            self._generate_rules_interaction(X, y_binary)
        else:
            raise ValueError(f"Unknown tree growing strategy: {self.tree_growing_strategy}")

        # Store rules for this transition
        self.transition_rules_[transition] = self.rules_

        # Evaluate rules on training data
        rule_matrix = self._evaluate_rules(X)

        # Fit elastic net to get rule weights
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )
        model.fit(rule_matrix, y_binary)

        # Store weights and model for this transition
        self.rule_weights_[transition] = model.coef_
        self.transition_models_[transition] = model
        
        # Update feature importances
        if not hasattr(self, 'feature_importances_'):
            self.feature_importances_ = np.zeros(X.shape[1])
        
        # Compute feature importances from rules
        for rule, weight in zip(self.rules_, model.coef_):
            for feature, _, _ in rule['conditions']:
                self.feature_importances_[feature] += abs(weight * rule['importance'])
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
    
    def _generate_rules_single_tree(self, X, y):
        """
        Generate rules using a single decision tree.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Binary target variable
        """
        from sklearn.tree import DecisionTreeClassifier
        
        # Fit a decision tree
        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_rules,
            random_state=self.random_state
        )
        tree.fit(X, y)
        
        # Extract rules from tree paths
        self.rules_ = self._extract_rules_from_tree(tree)
    
    def _generate_rules_forest(self, X, y):
        """
        Generate rules using a random forest.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Binary target variable
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Fit a random forest
        forest = RandomForestClassifier(
            n_estimators=min(10, self.max_rules),  # Number of trees
            max_leaf_nodes=max(2, self.max_rules // 10),  # Rules per tree
            random_state=self.random_state
        )
        forest.fit(X, y)
        
        # Extract rules from all trees
        rules = []
        for tree in forest.estimators_:
            rules.extend(self._extract_rules_from_tree(tree))
            
        # Keep only the top rules based on importance
        importances = forest.feature_importances_
        rules = sorted(rules, key=lambda x: x['importance'], reverse=True)[:self.max_rules]
        self.rules_ = rules
    
    def _generate_rules_interaction(self, X, y):
        """
        Generate rules using interaction trees.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Binary target variable
        """
        from sklearn.tree import DecisionTreeClassifier
        
        # First level tree
        tree1 = DecisionTreeClassifier(
            max_leaf_nodes=max(2, self.max_rules // 2),
            random_state=self.random_state
        )
        tree1.fit(X, y)
        rules1 = self._extract_rules_from_tree(tree1)
        
        # Second level trees (one for each leaf in first tree)
        rules = rules1.copy()
        for rule in rules1:
            # Get samples that satisfy this rule
            mask = self._evaluate_rule(X, rule)
            if mask.sum() > 10:  # Only if enough samples
                tree2 = DecisionTreeClassifier(
                    max_leaf_nodes=2,
                    random_state=self.random_state
                )
                tree2.fit(X[mask], y[mask])
                
                # Get rules from second tree and combine with first rule
                rules2 = self._extract_rules_from_tree(tree2)
                for rule2 in rules2:
                    combined_rule = {
                        'conditions': rule['conditions'] + rule2['conditions'],
                        'importance': (rule['importance'] + rule2['importance']) / 2
                    }
                    rules.append(combined_rule)
        
        # Keep only the top rules
        rules = sorted(rules, key=lambda x: x['importance'], reverse=True)[:self.max_rules]
        self.rules_ = rules
    
    def _extract_rules_from_tree(self, tree):
        """
        Extract rules from a decision tree.
        
        Parameters
        ----------
        tree : DecisionTreeClassifier
            Fitted decision tree
        
        Returns
        -------
        list
            List of rules, where each rule is a dict with 'conditions' and 'importance'
        """
        from sklearn.tree import _tree
        
        rules = []
        
        def recurse(node, conditions):
            if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                
                # Left branch (<=)
                left_conditions = conditions + [(feature, '<=', threshold)]
                recurse(tree.tree_.children_left[node], left_conditions)
                
                # Right branch (>)
                right_conditions = conditions + [(feature, '>', threshold)]
                recurse(tree.tree_.children_right[node], right_conditions)
            else:
                # Leaf node
                if len(conditions) > 0:  # Skip empty rules
                    importance = tree.tree_.impurity[node]
                    rules.append({
                        'conditions': conditions,
                        'importance': importance
                    })
        
        recurse(0, [])
        return rules
    
    def _evaluate_rules(self, X):
        """
        Evaluate rules on data.
        
        Parameters
        ----------
        X : array-like
            Data to evaluate rules on
        
        Returns
        -------
        np.ndarray
            Binary matrix where each column represents a rule
            and each row represents a sample
        """
        if self.rules_ is None or len(self.rules_) == 0:
            return np.zeros((X.shape[0], 1))
            
        rule_matrix = np.zeros((X.shape[0], len(self.rules_)))
        for i, rule in enumerate(self.rules_):
            rule_matrix[:, i] = self._evaluate_rule(X, rule)
        return rule_matrix
    
    def _evaluate_rule(self, X, rule):
        """
        Evaluate a single rule on data.
        
        Parameters
        ----------
        X : array-like
            Data to evaluate rule on
        rule : dict
            Rule to evaluate with 'conditions' key
        
        Returns
        -------
        np.ndarray
            Binary array indicating which samples satisfy the rule
        """
        mask = np.ones(X.shape[0], dtype=bool)
        for feature, op, threshold in rule['conditions']:
            if op == '<=':
                mask &= X[:, feature] <= threshold
            else:  # op == '>'
                mask &= X[:, feature] > threshold
        return mask

    def _compute_feature_importances(self):
        """Compute feature importances"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing feature importances")
        
        # Initialize feature importances
        n_features = len(self._rules_tuples)
        self._feature_importances = np.zeros(n_features)
        
        # Compute average absolute weights across all transitions
        for weights in self.rule_weights_.values():
            self._feature_importances += np.abs(weights)
        
        # Normalize to sum to 1
        total_importance = np.sum(self._feature_importances)
        if total_importance > 0:
            self._feature_importances /= total_importance
        
        return self._feature_importances