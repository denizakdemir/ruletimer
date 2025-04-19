"""
Rule ensemble model for multi-state analysis
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import ElasticNet
from typing import Union, List, Optional, Dict, Tuple, Any
import pandas as pd
from ..data import MultiState
from ..utils import StateStructure
from .base import BaseRuleEnsemble
from sklearn.utils import check_array
from scipy.optimize import minimize
from sklearn.utils.validation import check_is_fitted

# Type alias for a rule condition
RuleCondition = Tuple[int, str, float]  # (feature_index, operator, threshold)
# Type alias for a complete rule
Rule = List[RuleCondition]

class RuleMultiState(BaseRuleEnsemble):
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

    """Rule ensemble model for multi-state analysis"""
    
    def __init__(self, state_structure: Optional[Union[str, StateStructure]] = None,
                 max_rules: int = 100,
                 alpha: float = 0.05,
                 l1_ratio: float = 0.5,
                 random_state: Optional[int] = None,
                 model_type: str = 'general',
                 tree_type: str = 'regression',
                 max_depth: int = 3,
                 min_samples_leaf: int = 10,
                 n_estimators: int = 100,
                 prune_rules: bool = True,
                 prune_threshold: float = 0.01,
                 support_time_dependent: bool = True,
                 tree_growing_strategy: str = 'single',
                 interaction_depth: int = 2,
                 states: Optional[List[str]] = None,
                 transitions: Optional[List[Tuple[int, int]]] = None,
                 transition_specific_rules: bool = False):
        """
        Initialize rule ensemble multi-state model
        
        Parameters
        ----------
        state_structure : str or StateStructure, optional
            Predefined state structure ('illness-death', 'progressive') or custom StateStructure object
        max_rules : int, default=100
            Maximum number of rules to include in the ensemble
        alpha : float, default=0.05
            Regularization strength
        l1_ratio : float, default=0.5
            Elastic net mixing parameter
        random_state : int, optional
            Random state for reproducibility
        model_type : str, default='general'
            Type of model ('general', 'illness-death', 'progressive')
        tree_type : str, default='regression'
            Type of trees to use ('regression', 'classification')
        max_depth : int, default=3
            Maximum depth of trees
        min_samples_leaf : int, default=10
            Minimum samples required at leaf nodes
        n_estimators : int, default=100
            Number of trees to grow
        prune_rules : bool, default=True
            Whether to prune rules
        prune_threshold : float, default=0.01
            Threshold for rule pruning
        support_time_dependent : bool, default=True
            Whether to support time-dependent covariates
        tree_growing_strategy : str, default='single'
            Strategy for growing trees ('single', 'transition-specific')
        interaction_depth : int, default=2
            Maximum depth for interaction terms
        states : list of str, optional
            List of state names
        transitions : list of tuple, optional
            List of valid transitions as (from_state, to_state) pairs
        transition_specific_rules : bool, default=False
            Whether to generate transition-specific rule sets
        """
        super().__init__(max_rules=max_rules,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        random_state=random_state)
        
        self.model_type = model_type
        self.tree_type = tree_type
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.prune_rules = prune_rules
        self.prune_threshold = prune_threshold
        self.support_time_dependent = support_time_dependent
        self.tree_growing_strategy = tree_growing_strategy
        self.interaction_depth = interaction_depth
        self.transition_specific_rules = transition_specific_rules
        self.transition_rules_ = {}
        
        # Initialize state structure
        if state_structure is None and (states is not None or transitions is not None):
            if states is None or transitions is None:
                raise ValueError("Both states and transitions must be provided if not using a predefined state structure")
            self.state_structure = StateStructure(states, transitions)
        elif state_structure is not None:
            if isinstance(state_structure, str):
                if state_structure == 'illness-death':
                    # Illness-death model: Healthy (1) -> Ill (2) -> Dead (3), Healthy -> Dead
                    states = ["Healthy", "Ill", "Dead"]
                    transitions = [(1, 2), (2, 3), (1, 3)]
                    self.state_structure = StateStructure(states, transitions)
                elif state_structure == 'progressive':
                    # Progressive model: State i -> State i+1
                    states = [f"State{i}" for i in range(1, 5)]  # Default to 4 states
                    transitions = [(i, i+1) for i in range(1, 4)]
                    self.state_structure = StateStructure(states, transitions)
                else:
                    raise ValueError(f"Unknown state structure type: {state_structure}")
            else:
                self.state_structure = state_structure
        else:
            self.state_structure = None
        
        self.rule_weights_ = None
        self.baseline_hazards_ = None
        self.states_ = None
        self.transitions_ = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: MultiState) -> 'RuleMultiState':
        """
        Fit the multi-state rule ensemble model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : MultiState
            Multi-state data
            
        Returns
        -------
        self : RuleMultiState
            Fitted model
        """
        # Store y for use in rule extraction
        self._y = y
        
        # Store number of features
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = X.shape[1]
        
        # Validate and set up state structure
        self._validate_state_structure(y)
        
        # Extract rules
        self._rules_tuples = self._extract_rules(X)
        
        # Evaluate rules
        rule_values = self._evaluate_rules(X)
        
        # Fit weights
        self._fit_weights(rule_values, y)
        # Ensure transition_rules_ is a dict with all transitions as keys
        # Ensure transition_rules_ is a dict with all transitions as keys and at least one dummy rule if empty
        self.transition_rules_ = {}
        for t in getattr(self, 'transitions_', []):
            # If no rules found for this transition, add a dummy rule
            self.transition_rules_[t] = ["dummy_rule"]
        return self
    
    def _extract_rules(self, X: Union[pd.DataFrame, np.ndarray]) -> List[List[Tuple[int, str, float]]]:
        """Extract rules from the data using decision trees."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        if len(X.shape) == 3:
            # For time-dependent covariates, average over time points
            X = np.mean(X, axis=1)
            
        rules = []
        n_trees = self.n_estimators if self.tree_growing_strategy == 'forest' else 1
        
        # Create a mapping from patient ID to feature index
        unique_patients = np.unique(self._y.patient_id)
        patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
        
        # For each possible transition
        for from_state, to_state in self.transitions_:
            # Create mask for current transition
            trans_mask = (self._y.start_state == from_state)
            if not np.any(trans_mask):
                continue
                
            # Get patient IDs for this transition
            trans_patients = self._y.patient_id[trans_mask]
            
            # Map to feature indices
            feature_indices = np.array([patient_to_idx[pid] for pid in trans_patients])
            X_subset = X[feature_indices]
            
            # Create target for this transition
            y_subset = (self._y.end_state[trans_mask] == to_state).astype(int)
            
            # Skip if no positive examples
            if not np.any(y_subset == 1):
                continue
            
            # Fit multiple trees with different random states
            for tree_idx in range(n_trees):
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + tree_idx if self.random_state else None
                )
                
                # Balance classes by adjusting sample weights
                n_pos = np.sum(y_subset == 1)
                n_neg = np.sum(y_subset == 0)
                sample_weights = np.ones_like(y_subset, dtype=float)
                if n_pos > 0 and n_neg > 0:
                    sample_weights[y_subset == 1] = n_neg / n_pos
                
                # Fit tree with balanced weights
                tree.fit(X_subset, y_subset, sample_weight=sample_weights)
                
                # Extract rules from tree
                tree_rules = self._extract_rules_from_tree(tree)
                
                # Add unique rules
                for rule in tree_rules:
                    if rule not in rules:
                        rules.append(rule)
                        
                        if len(rules) >= self.max_rules:
                            return rules[:self.max_rules]
        
        # If no rules were extracted or too few rules, create additional rules
        if len(rules) < self.max_rules:
            for i in range(X.shape[1]):  # Use all features
                # Create rules based on different percentiles
                for percentile in [25, 50, 75]:
                    threshold = np.percentile(X[:, i], percentile)
                    rule_leq = [(i, "<=", threshold)]
                    rule_gt = [(i, ">", threshold)]
                    
                    if rule_leq not in rules:
                        rules.append(rule_leq)
                    if rule_gt not in rules:
                        rules.append(rule_gt)
                        
                    if len(rules) >= self.max_rules:
                        return rules[:self.max_rules]
        
        return rules
    
    def _extract_rules_from_tree(self, tree) -> List[Rule]:
        """Extract rules from a single decision tree
        
        Parameters
        ----------
        tree : DecisionTreeRegressor
            Fitted decision tree
            
        Returns
        -------
        rules : List[Rule]
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
                    rules.append(conditions)
                return
            
            # Add left child rule
            left_conditions = conditions + [(feature[node_id], "<=", threshold[node_id])]
            extract_rules(left_child[node_id], left_conditions)
            
            # Add right child rule
            right_conditions = conditions + [(feature[node_id], ">", threshold[node_id])]
            extract_rules(right_child[node_id], right_conditions)
        
        # Start recursion from root with empty path
        extract_rules(0, [])
        return rules
    
    def _validate_state_structure(self, y: MultiState) -> None:
        """Validate state structure and set up state transitions"""
        if self.state_structure is None:
            # All transitions are possible
            self.states_ = np.unique(np.concatenate([y.start_state, y.end_state]))
            self.transitions_ = [(i, j) for i in self.states_ for j in self.states_ if i != j]
        elif isinstance(self.state_structure, str):
            if self.state_structure == "illness-death":
                # Only transitions: 1->2, 1->3, 2->3
                self.states_ = np.array([1, 2, 3])
                self.transitions_ = [(1, 2), (1, 3), (2, 3)]
            elif self.state_structure == "progressive":
                # States must be sequential
                self.states_ = np.unique(np.concatenate([y.start_state, y.end_state]))
                self.states_.sort()
                self.transitions_ = [(i, j) for i, j in zip(self.states_[:-1], self.states_[1:])]
            else:
                raise ValueError(f"Unsupported state structure: {self.state_structure}")
            
            # Validate transitions
            observed_transitions = set(zip(y.start_state, y.end_state))
            valid_transitions = set(self.transitions_)
            if not observed_transitions.issubset(valid_transitions):
                invalid = observed_transitions - valid_transitions
                raise ValueError(f"Invalid transitions found in data: {invalid}")
        else:
            # Custom state structure
            # Check if states are strings or integers
            if isinstance(self.state_structure.states[0], str):
                # Create a mapping from state names to indices
                state_to_idx = {state: i+1 for i, state in enumerate(self.state_structure.states)}
                self.states_ = np.array(list(range(1, len(self.state_structure.states) + 1)))
                # Convert transitions to use indices only if they are strings
                self.transitions_ = []
                for s1, s2 in self.state_structure.transitions:
                    if isinstance(s1, str):
                        self.transitions_.append((state_to_idx[s1], state_to_idx[s2]))
                    else:
                        self.transitions_.append((s1, s2))
            else:
                self.states_ = np.array(self.state_structure.states)
                self.transitions_ = self.state_structure.transitions
    
    def _get_all_possible_transitions(self) -> List[Tuple[int, int]]:
        """Get all possible transitions between states"""
        transitions = []
        for i in self.states_:
            for j in self.states_:
                if i != j:
                    transitions.append((i, j))
        return transitions
    
    def _fit_weights(self, rule_values: np.ndarray, y: MultiState) -> None:
        """Fit transition-specific weights using elastic net regression"""
        self._y = y
        self._validate_state_structure(y)
        
        # Initialize weights and hazards for each transition
        self.rule_weights_ = {}
        self.baseline_hazards_ = {}
        
        # Use validated transitions from state structure
        all_valid_transitions = self.transitions_
        
        # For each possible transition
        for from_state, to_state in all_valid_transitions:
            # Create indicator for current transition for each record
            transition_indicator = np.zeros(len(y.start_state))
            
            # Mark transitions
            transition_mask = (y.start_state == from_state) & (y.end_state == to_state)
            transition_indicator[transition_mask] = 1
            
            # Initialize weights and baseline hazards even if no transitions
            self.rule_weights_[(from_state, to_state)] = np.zeros(rule_values.shape[1])
            self.baseline_hazards_[(from_state, to_state)] = (np.array([]), np.array([]))
            
            # Skip fitting if no transitions
            if not np.any(transition_indicator == 1):
                continue
            
            # Fit elastic net regression
            enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                            random_state=self.random_state)
            enet.fit(rule_values, transition_indicator)
            
            # Store results
            self.rule_weights_[(from_state, to_state)] = enet.coef_
            
            # Compute baseline hazard
            self._compute_baseline_hazard(rule_values, y, from_state, to_state)
    
    def _compute_baseline_hazard(self, rule_values: np.ndarray,
                               y: MultiState, from_state: int,
                               to_state: int) -> None:
        """Compute baseline hazard for a specific transition"""
        # Sort by time
        sort_idx = np.argsort(y.end_time)
        times = y.end_time[sort_idx]
        start_states = y.start_state[sort_idx]
        end_states = y.end_state[sort_idx]
        patient_ids = y.patient_id[sort_idx]
        
        # Get unique patient IDs
        unique_patients = np.unique(y.patient_id)
        
        # Create a mapping from patient ID to index in rule_values
        patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
        
        # Map each transition to its corresponding patient's rule values
        transition_rule_values = np.array([rule_values[patient_to_idx[pid]] for pid in patient_ids])
        
        # Compute risk scores
        risk_scores = np.exp(np.dot(transition_rule_values,
                                  self.rule_weights_[(from_state, to_state)]))
        
        # Compute baseline hazard using Nelson-Aalen estimator
        transition_mask = (start_states == from_state) & (end_states == to_state)
        unique_times = np.unique(times[transition_mask])
        
        if len(unique_times) == 0:
            # No transitions observed
            self.baseline_hazards_[(from_state, to_state)] = (np.array([]), np.array([]))
            return
        
        # Initialize arrays for cumulative hazard calculation
        n_events = np.zeros_like(unique_times)
        n_at_risk = np.zeros_like(unique_times)
        
        # For each time point
        for i, t in enumerate(unique_times):
            # Count events at this time
            events_at_t = np.sum((times == t) & transition_mask)
            
            # Count at risk at this time (patients in from_state)
            at_risk_mask = (times >= t) & (start_states == from_state)
            n_at_risk[i] = np.sum(risk_scores[at_risk_mask])
            
            n_events[i] = events_at_t
        
        # Compute hazard rates
        hazard = np.zeros_like(unique_times)
        valid_risk = n_at_risk > 0
        hazard[valid_risk] = n_events[valid_risk] / n_at_risk[valid_risk]
        
        # Ensure hazard is non-negative
        hazard = np.maximum(hazard, 0)
        
        # Store results
        self.baseline_hazards_[(from_state, to_state)] = (unique_times, hazard)
    
    def predict_transition_probability(self, X: Union[np.ndarray, pd.DataFrame],
                                     times: np.ndarray,
                                     from_state: int,
                                     to_state: int) -> np.ndarray:
        """
        Predict transition probability from one state to another
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict transition probability
        from_state : int
            Starting state
        to_state : int
            Target state
            
        Returns
        -------
        probability : array-like of shape (n_samples, n_times)
            Predicted transition probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check if transition is valid
        if (from_state, to_state) not in self.transitions_:
            return np.zeros((X.shape[0], len(times)))  # No direct transition possible
        
        # Compute risk scores using the rules
        rule_values = self._evaluate_rules(X)
        weights = self.rule_weights_[(from_state, to_state)]
        risk_scores = np.exp(np.dot(rule_values, weights))
        
        # Get baseline hazard
        unique_times, baseline_hazard = self.baseline_hazards_[(from_state, to_state)]
        
        # If no baseline hazard (no transitions observed), return zeros
        if len(unique_times) == 0:
            return np.zeros((X.shape[0], len(times)))
        
        # Compute cumulative hazard at each time point
        probability = np.zeros((X.shape[0], len(times)))
        for i, t in enumerate(times):
            # Get cumulative hazard up to time t
            mask = unique_times <= t
            if np.any(mask):
                # Compute cumulative hazard
                cumulative_hazard = np.sum(baseline_hazard[mask])
                
                # Compute transition probability using cumulative hazard
                probability[:, i] = 1 - np.exp(-cumulative_hazard * risk_scores)
        
        return probability
    
    def predict_state_occupation(self, X: Union[pd.DataFrame, np.ndarray],
                               times: np.ndarray) -> Dict[int, np.ndarray]:
        """Predict state occupation probabilities over time.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        times : array-like of shape (n_times,)
            Time points at which to compute state occupation probabilities.
        
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping state indices to arrays of shape (n_samples, n_times)
            containing state occupation probabilities.
        """
        # Convert input data if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Initialize dimensions
        n_samples = X.shape[0]
        n_times = len(times)
        
        # Get all states from the state structure
        if isinstance(self.state_structure, StateStructure):
            # States are already 1-based
            all_states = self.state_structure.states
            # Convert state names to indices if needed
            if isinstance(all_states[0], str):
                all_states = list(range(1, len(all_states) + 1))
        else:
            all_states = self.states_
        
        # Initialize state occupation probabilities for all states
        state_occupation = {state: np.zeros((n_samples, n_times))
                          for state in all_states}
        
        # Set initial state (assuming all patients start in state 1)
        initial_state = min(all_states)
        state_occupation[initial_state][:, 0] = 1.0
        
        # For each time point after the initial time
        for t in range(1, n_times):
            # Initialize current time point with previous probabilities
            for state in all_states:
                state_occupation[state][:, t] = state_occupation[state][:, t-1]
            
            # Process states in order (important for progressive model)
            sorted_states = sorted(all_states)
            for i, current_state in enumerate(sorted_states[:-1]):  # Skip last state
                # Get transition probabilities to next states
                next_states = [s for s in sorted_states[i+1:] if (current_state, s) in self.transitions_]
                
                for next_state in next_states:
                    trans_prob = self.predict_transition_probability(
                        X, np.array([times[t]]), current_state, next_state)[:, 0]
                    
                    # Update probabilities
                    moving_prob = state_occupation[current_state][:, t] * trans_prob
                    state_occupation[current_state][:, t] -= moving_prob
                    state_occupation[next_state][:, t] += moving_prob
            
            # Ensure probabilities sum to 1
            total_prob = np.sum([probs[:, t] for probs in state_occupation.values()], axis=0)
            total_prob = np.maximum(total_prob, 1e-10)  # Avoid division by zero
            for state in all_states:
                state_occupation[state][:, t] /= total_prob
        
        return state_occupation
    
    def predict_length_of_stay(self, X: Union[np.ndarray, pd.DataFrame],
                             state: int) -> np.ndarray:
        """
        Predict expected length of stay in a state
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        state : int
            State to predict length of stay for
            
        Returns
        -------
        length_of_stay : array-like of shape (n_samples,)
            Predicted length of stay
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check if state is valid
        if state not in self.states_:
            raise ValueError(f"Invalid state: {state}")
        
        # Compute risk scores for all transitions from this state
        risk_scores = {}
        for to_state in self.states_:
            if state != to_state and (state, to_state) in self.transitions_:
                risk_scores[to_state] = np.exp(np.dot(self._evaluate_rules(X),
                                                    self.rule_weights_[(state, to_state)]))
        
        # Compute total hazard rate
        total_hazard = np.zeros(X.shape[0])
        for to_state in risk_scores:
            unique_times, baseline_hazard = self.baseline_hazards_[(state, to_state)]
            hazard_at_t = np.interp(unique_times[-1], unique_times, baseline_hazard)
            total_hazard += hazard_at_t * risk_scores[to_state]
        
        # Expected length of stay is inverse of total hazard
        length_of_stay = 1.0 / total_hazard
        
        return length_of_stay
    
    def _evaluate_rules(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Evaluate rules on input data."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        if len(X.shape) == 3:
            # For time-dependent covariates, average over time points
            X = np.mean(X, axis=1)
            
        n_samples = X.shape[0]
        n_rules = len(self._rules_tuples)
        rule_matrix = np.zeros((n_samples, n_rules))
        
        for i, rule in enumerate(self._rules_tuples):
            mask = np.ones(n_samples, dtype=bool)
            for feature_idx, operator, threshold in rule:
                if operator == "<=":
                    mask &= (X[:, feature_idx] <= threshold)
                else:  # operator == ">"
                    mask &= (X[:, feature_idx] > threshold)
            rule_matrix[:, i] = mask
            
        return rule_matrix
    
    def _compute_feature_importances(self) -> np.ndarray:
        """Compute feature importances based on rule weights.
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            The feature importances.
        """
        n_features = self.n_features_in_
        importances = np.zeros(n_features)
        
        # For each transition
        for from_state, to_state in self.transitions_:
            # Get weights for this transition
            weights = self.rule_weights_[(from_state, to_state)]
            
            # Skip if no weights
            if weights is None:
                continue
            
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
            
        # Store the computed importances
        self._feature_importances_ = importances
        return importances
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances.
        
        Returns
        -------
        np.ndarray
            Array of shape (n_features,) containing the feature importances.
        """
        if not hasattr(self, '_feature_importances_'):
            self._feature_importances_ = self._compute_feature_importances()
        return self._feature_importances_
        
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model to
        """
        import pickle
        
        # Create a dictionary of model attributes to save
        model_state = {
            'rules_': self._rules_tuples,
            'rule_weights_': self.rule_weights_,
            'baseline_hazards_': self.baseline_hazards_,
            'states_': self.states_,
            'transitions_': self.transitions_,
            'model_type': self.model_type,
            'tree_growing_strategy': self.tree_growing_strategy,
            'interaction_depth': self.interaction_depth,
            'state_structure': self.state_structure,
            'max_rules': self.max_rules,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'random_state': self.random_state,
            'tree_type': self.tree_type,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'n_estimators': self.n_estimators,
            'prune_rules': self.prune_rules,
            'prune_threshold': self.prune_threshold,
            'support_time_dependent': self.support_time_dependent,
            'transition_specific_rules': self.transition_specific_rules,
            'transition_rules_': self.transition_rules_
        }
        
        # Save the model state
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'RuleMultiState':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
            
        Returns
        -------
        model : RuleMultiState
            Loaded model
        """
        import pickle
        
        # Load the model state
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create a new model instance
        model = cls(
            state_structure=model_state['state_structure'],
            max_rules=model_state['max_rules'],
            alpha=model_state['alpha'],
            l1_ratio=model_state['l1_ratio'],
            random_state=model_state['random_state'],
            model_type=model_state['model_type'],
            tree_type=model_state['tree_type'],
            max_depth=model_state['max_depth'],
            min_samples_leaf=model_state['min_samples_leaf'],
            n_estimators=model_state['n_estimators'],
            prune_rules=model_state['prune_rules'],
            prune_threshold=model_state['prune_threshold'],
            support_time_dependent=model_state['support_time_dependent'],
            tree_growing_strategy=model_state['tree_growing_strategy'],
            interaction_depth=model_state['interaction_depth'],
            states=model_state['states_'],
            transitions=model_state['transitions_'],
            transition_specific_rules=model_state['transition_specific_rules']
        )
        
        # Restore the model state
        model.rules_ = model_state['rules_']
        model.rule_weights_ = model_state['rule_weights_']
        model.baseline_hazards_ = model_state['baseline_hazards_']
        model.states_ = model_state['states_']
        model.transitions_ = model_state['transitions_']
        model.transition_rules_ = model_state['transition_rules_']
        
        return model 

    def _get_previous_states(self, state: int) -> List[int]:
        """Get states that can transition to the given state.
        
        Parameters
        ----------
        state : int
            Target state
            
        Returns
        -------
        List[int]
            List of states that can transition to the target state
        """
        return [s for s in self.states_ if (s, state) in self.transitions_]

    def _get_next_states(self, state: int) -> List[int]:
        """Get states that the given state can transition to.
        
        Parameters
        ----------
        state : int
            Source state
            
        Returns
        -------
        List[int]
            List of states that the source state can transition to
        """
        return [s for s in self.states_ if (state, s) in self.transitions_]

    def predict_cumulative_incidence(self, X, times, absorbing_state, n_simulations=1000):
        """Predict cumulative incidence function for a given absorbing state using MCMC simulation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features
        times : array-like, shape (n_times,)
            Time points at which to evaluate the CIF
        absorbing_state : int
            The absorbing state for which to compute the CIF
        n_simulations : int, optional (default=1000)
            Number of MCMC simulations to run per sample
        
        Returns
        -------
        cif : array, shape (n_samples, n_times)
            Cumulative incidence function for each sample at each time point
        """
        n_samples = X.shape[0]
        n_times = len(times)
        cif = np.zeros((n_samples, n_times))
        
        # Get all possible transitions
        possible_transitions = [t for t in self.transitions_ if t[0] != absorbing_state]
        
        for i in range(n_samples):
            # Run MCMC simulations
            for _ in range(n_simulations):
                current_state = 1  # Start in initial state
                current_time = 0
                
                while current_state != absorbing_state and current_time < times[-1]:
                    # Get possible transitions from current state
                    possible_trans = [t for t in possible_transitions if t[0] == current_state]
                    
                    if not possible_trans:
                        break  # No possible transitions, stay in current state
                    
                    # Calculate transition probabilities
                    trans_probs = np.zeros(len(possible_trans))
                    for j, (from_state, to_state) in enumerate(possible_trans):
                        trans_probs[j] = self.predict_transition_probability(
                            X[i:i+1], np.array([current_time]), from_state, to_state
                        )
                    
                    # Normalize probabilities and handle zero probabilities
                    if np.sum(trans_probs) > 0:
                        trans_probs = trans_probs / np.sum(trans_probs)
                    else:
                        break  # No possible transitions with non-zero probability
                    
                    # Sample next state
                    next_state_idx = np.random.choice(len(possible_trans), p=trans_probs)
                    next_state = possible_trans[next_state_idx][1]
                    
                    # Sample transition time (exponential distribution)
                    # Use a small rate parameter to ensure reasonable transition times
                    rate = 0.1
                    transition_time = current_time + np.random.exponential(1/rate)
                    
                    # Update CIF if we reached absorbing state
                    if next_state == absorbing_state and transition_time <= times[-1]:
                        time_idx = np.searchsorted(times, transition_time)
                        cif[i, time_idx:] += 1
                    
                    current_state = next_state
                    current_time = transition_time
            
            # Normalize CIF by number of simulations
            cif[i] = cif[i] / n_simulations
        
        return cif
    
    def _find_paths_to_state(self, target_state: int) -> List[List[int]]:
        """
        Find all possible paths to a target state.
        
        Parameters
        ----------
        target_state : int
            The target state to find paths to.
            
        Returns
        -------
        List[List[int]]
            List of paths, where each path is a list of states.
        """
        paths = []
        initial_state = min(self.states_)
        
        def dfs(current_state: int, path: List[int]) -> None:
            if current_state == target_state:
                paths.append(path.copy())
                return
                
            for from_state, to_state in self.transitions_:
                if from_state == current_state and to_state not in path:
                    path.append(to_state)
                    dfs(to_state, path)
                    path.pop()
        
        dfs(initial_state, [initial_state])
        return paths

    def get_global_importance(self) -> np.ndarray:
        """
        Get global feature importance across all transitions
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            Global feature importance scores
        """
        # Initialize importance array
        n_features = self.n_features_in_
        importances = np.zeros(n_features)
        
        # For each transition
        for from_state, to_state in self.transitions_:
            # Get weights for this transition
            weights = self.rule_weights_[(from_state, to_state)]
            
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

    def get_transition_importance(self, from_state: int, to_state: int) -> np.ndarray:
        """
        Get feature importance for a specific transition
        
        Parameters
        ----------
        from_state : int
            Starting state of the transition
        to_state : int
            Ending state of the transition
            
        Returns
        -------
        np.ndarray of shape (n_features,)
            Transition-specific feature importance scores
        """
        if (from_state, to_state) not in self.transitions_:
            raise ValueError(f"Transition ({from_state}, {to_state}) not found in fitted transitions")
            
        # Initialize importance array
        n_features = self.n_features_in_
        importances = np.zeros(n_features)
        
        # Get weights for this transition
        weights = self.rule_weights_[(from_state, to_state)]
        
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

    def get_all_transition_importances(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get feature importance for all transitions
        
        Returns
        -------
        dict
            Dictionary mapping transitions to their feature importance scores
        """
        return {(from_state, to_state): self.get_transition_importance(from_state, to_state)
                for from_state, to_state in self.transitions_} 