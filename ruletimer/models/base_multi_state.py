"""
Base class for multi-state time-to-event models.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from ruletimer.utils.hazard_estimation import HazardEstimator
from ruletimer.state_manager import StateManager
from ruletimer.time_handler import TimeHandler
from ruletimer.utils import StateStructure

class BaseMultiStateModel(ABC):
    """
    Base class for all multi-state time-to-event models.
    
    This class provides the foundation for:
    1. Standard survival analysis (2-state model)
    2. Competing risks (1 initial state, multiple absorbing states)
    3. General multi-state models
    """
    
    def __init__(
        self,
        states: List[str],
        transitions: List[Tuple[int, int]],
        hazard_method: str = "nelson-aalen"
    ):
        """
        Initialize multi-state model.
        
        Parameters
        ----------
        states : list of str
            List of state names
        transitions : list of tuple
            List of valid transitions as (from_state, to_state) pairs
        hazard_method : str
            Method for hazard estimation: "nelson-aalen" or "parametric"
        """
        self.state_manager = StateManager(states, transitions)
        self.hazard_method = hazard_method
        self.hazard_estimator = HazardEstimator()
        self.time_handler = TimeHandler()
        
        # Initialize model attributes
        self.baseline_hazards_: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.transition_models_: Dict[Tuple[int, int], object] = {}
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model - to be implemented by subclasses."""
        pass
    
    def _estimate_baseline_hazards(
        self,
        transition_times: Dict[Tuple[int, int], np.ndarray],
        transition_events: Dict[Tuple[int, int], np.ndarray],
        transition_weights: Optional[Dict[Tuple[int, int], np.ndarray]] = None
    ) -> None:
        """
        Estimate baseline hazards for all transitions.
        
        Parameters
        ----------
        transition_times : dict
            Dictionary mapping transitions to event times
        transition_events : dict
            Dictionary mapping transitions to event indicators
        transition_weights : dict, optional
            Dictionary mapping transitions to case weights
        """
        for transition in self.state_manager.transitions:
            if transition_weights is not None:
                weights = transition_weights.get(transition)
            else:
                weights = None
                
            times, hazard = self.hazard_estimator.estimate_baseline_hazard(
                transition_times[transition],
                transition_events[transition],
                weights=weights,
                method=self.hazard_method
            )
            self.baseline_hazards_[transition] = (times, hazard)
    
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
            Predicted hazard values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Convert states to internal indices
        from_idx = self.state_manager.to_internal_index(from_state)
        to_idx = self.state_manager.to_internal_index(to_state)
        transition = (from_idx, to_idx)
        
        if transition not in self.transition_models_:
            raise ValueError(f"No model for transition {transition}")
            
        # Get baseline hazard and relative hazard
        baseline_times, baseline_hazard = self.baseline_hazards_[transition]
        
        # Evaluate rules on input data
        rule_matrix = self._evaluate_rules(X, self.rules_[transition])
        relative_hazard = self.transition_models_[transition].predict(rule_matrix)
        
        # Interpolate baseline hazard to requested times
        hazard = np.zeros((len(X), len(times)))
        for i, rel_haz in enumerate(relative_hazard):
            interp_hazard = np.interp(times, baseline_times, baseline_hazard)
            hazard[i] = rel_haz * interp_hazard
            
        return hazard
    
    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict cumulative hazard function for a specific transition.

        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict cumulative hazard
        from_state : str or int
            Starting state
        to_state : str or int
            Target state

        Returns
        -------
        array-like
            Predicted cumulative hazard values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Ensure times are sorted
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        # Get hazard at each time point
        hazard = self.predict_transition_hazard(X, sorted_times, from_state, to_state)

        # Ensure hazard is non-negative
        hazard = np.maximum(hazard, 0)

        # Compute cumulative hazard using trapezoidal rule
        dt = np.diff(sorted_times)
        cum_hazard = np.zeros_like(hazard)
        cum_hazard[:, 1:] = np.cumsum(0.5 * (hazard[:, 1:] + hazard[:, :-1]) * dt, axis=1)

        # Ensure strict monotonicity
        cum_hazard = np.maximum.accumulate(cum_hazard, axis=1)

        # Return values in original time order
        return cum_hazard[:, unsort_idx]
    
    def predict_transition_probability(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict probability of transitioning from from_state to to_state by given times.
        Uses P(t) = 1 - exp(-CumulativeHazard(t)).

        Parameters
        ----------
        X : array-like
            Covariate values
        times : array-like
            Times at which to predict probability
        from_state : str or int
            Starting state
        to_state : str or int
            Target state

        Returns
        -------
        np.ndarray
            Predicted transition probabilities of shape (n_samples, n_times)
        """
        # Ensure times are sorted for monotonicity
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        # Get cumulative hazard
        cumulative_hazard = self.predict_cumulative_hazard(X, sorted_times, from_state, to_state)

        # Ensure cumulative hazard is non-negative and monotonic
        cumulative_hazard = np.maximum(cumulative_hazard, 0)
        cumulative_hazard = np.maximum.accumulate(cumulative_hazard, axis=1)

        # Calculate transition probability: P(t) = 1 - exp(-H(t))
        # Clip the cumulative hazard to prevent overflow in exp
        max_hazard_val = np.log(np.finfo(np.float64).max) / 2  # Safe upper bound
        clipped_hazard = np.minimum(cumulative_hazard, max_hazard_val)
        
        # Calculate probabilities with numerical stability
        with np.errstate(over='ignore', under='ignore'):
            probability = -np.expm1(-clipped_hazard)  # More accurate than 1 - exp(-x)

        # Ensure probabilities are in [0, 1] and strictly monotonic
        probability = np.clip(probability, 0, 1)
        probability = np.maximum.accumulate(probability, axis=1)

        # Return values in original time order
        return probability[:, unsort_idx]
    
    def predict_state_occupation(
        self,
        X: np.ndarray,
        times: np.ndarray,
        initial_state: Union[str, int]
    ) -> Dict[int, np.ndarray]:
        """
        Predict state occupation probabilities.
        
        Parameters
        ----------
        X : array-like
            Covariate values
        times : array-like
            Times at which to predict probabilities
        initial_state : str or int
            Initial state
            
        Returns
        -------
        dict
            Dictionary mapping states to occupation probabilities
        """
        # Sort times for monotonicity
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        initial_idx = self.state_manager.to_internal_index(initial_state)
        n_states = len(self.state_manager.states)
        n_samples = len(X)
        
        # Initialize with all probability mass in initial state
        occupation = {
            state: np.zeros((n_samples, len(times)))
            for state in range(n_states)
        }
        occupation[initial_idx][:, 0] = 1.0

        # Pre-compute all transition probabilities for efficiency
        transition_probs = {}
        for from_state in range(n_states):
            for to_state in range(n_states):
                if from_state != to_state and self.state_manager.validate_transition(from_state, to_state):
                    probs = self.predict_transition_probability(X, sorted_times, from_state, to_state)
                    probs = np.clip(probs, 0, 1)  # Ensure valid probabilities
                    transition_probs[(from_state, to_state)] = probs
        
        # Compute state occupation probabilities forward in time
        for t_idx in range(1, len(sorted_times)):
            # First copy previous state occupations
            for state in range(n_states):
                occupation[state][:, t_idx] = occupation[state][:, t_idx-1]

            # Process non-absorbing states first
            for from_state in range(n_states):
                # Skip if this is an absorbing state
                if any(self.state_manager.validate_transition(from_state, other) for other in range(n_states)):
                    current_occ = occupation[from_state][:, t_idx]
                    if np.any(current_occ > 0):
                        for to_state in range(n_states):
                            if from_state != to_state and self.state_manager.validate_transition(from_state, to_state):
                                # Get transition probabilities
                                trans_prob = transition_probs[(from_state, to_state)][:, t_idx]
                                
                                # Calculate transfer amount (ensure non-negative)
                                transfer = current_occ * trans_prob
                                transfer = np.maximum(transfer, 0)
                                
                                # Update probabilities
                                occupation[to_state][:, t_idx] += transfer
                                occupation[from_state][:, t_idx] -= transfer
            
            # Normalize probabilities
            total_prob = np.sum([occupation[state][:, t_idx] for state in range(n_states)], axis=0)
            total_prob = np.maximum(total_prob, 1e-10)  # Avoid division by zero
            for state in range(n_states):
                occupation[state][:, t_idx] /= total_prob

            # Ensure monotonicity for absorbing states
            for state in range(n_states):
                if not any(self.state_manager.validate_transition(state, other) for other in range(n_states)):
                    # This is an absorbing state - ensure monotonicity
                    occupation[state][:, t_idx] = np.maximum(
                        occupation[state][:, t_idx],
                        occupation[state][:, t_idx-1]
                    )

        # Return to original time order
        for state in occupation:
            occupation[state] = occupation[state][:, unsort_idx]
        
        return occupation

class RuleMultiState(BaseMultiStateModel):
    """
    Rule-based multi-state model for time-to-event data.
    
    This class implements a rule-based approach to multi-state modeling,
    where rules are generated using tree-based methods and combined using
    elastic net regularization.
    """
    
    def __init__(
        self,
        max_rules: int = 100,
        alpha: float = 0.1,
        state_structure: StateStructure = None,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        n_estimators: int = 100,
        tree_type: str = 'classification',
        tree_growing_strategy: str = 'forest',
        prune_rules: bool = True,
        l1_ratio: float = 0.5,
        random_state: int = None,
        hazard_method: str = "nelson-aalen"
    ):
        """
        Initialize the rule-based multi-state model.
        
        Parameters
        ----------
        max_rules : int
            Maximum number of rules to generate per transition
        alpha : float
            Regularization strength for elastic net
        state_structure : StateStructure
            Structure defining states and valid transitions
        max_depth : int
            Maximum depth of trees for rule generation
        min_samples_leaf : int
            Minimum samples required at leaf nodes
        n_estimators : int
            Number of trees in random forest
        tree_type : str
            Type of tree to use ('classification' or 'regression')
        tree_growing_strategy : str
            Strategy for growing trees ('single', 'forest', 'interaction')
        prune_rules : bool
            Whether to prune redundant rules
        l1_ratio : float
            Elastic net mixing parameter (0 <= l1_ratio <= 1)
        random_state : int
            Random state for reproducibility
        hazard_method : str
            Method for hazard estimation
        """
        if state_structure is None:
            raise ValueError("state_structure must be provided")
            
        super().__init__(
            states=state_structure.states,
            transitions=state_structure.transitions,
            hazard_method=hazard_method
        )
        
        self.max_rules = max_rules
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.tree_type = tree_type
        self.tree_growing_strategy = tree_growing_strategy
        self.prune_rules = prune_rules
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        
        # Initialize rule-related attributes
        self.rules_: Dict[Tuple[int, int], List[str]] = {}
        self.rule_weights_: Dict[Tuple[int, int], np.ndarray] = {}
        
        # Store state information
        self.states_ = state_structure.states
        self.transitions_ = state_structure.transitions

    def _map_transition_keys(self, data_dict):
        # Map integer transition keys to internal transition labels if needed
        mapped = {}
        for k, v in data_dict.items():
            if k in self.transitions_:
                mapped[k] = v
            elif (isinstance(k, tuple) and all(isinstance(x, int) for x in k)):
                # Try to map integer indices to state labels
                try:
                    from_state = self.states_[k[0]]
                    to_state = self.states_[k[1]]
                    mapped_key = (from_state, to_state)
                    if mapped_key in self.transitions_:
                        mapped[mapped_key] = v
                    else:
                        mapped[k] = v
                except Exception:
                    mapped[k] = v
            else:
                mapped[k] = v
        return mapped
        
    def _generate_rules(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """Generate rules using the specified tree growing strategy."""
        if self.tree_growing_strategy == 'forest':
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            forest.fit(X, y)
            rules = []
            for tree in forest.estimators_:
                rules.extend(self._extract_rules_from_tree(tree))
            return rules[:self.max_rules]
        else:
            raise ValueError(f"Unknown tree growing strategy: {self.tree_growing_strategy}")
            
    def _extract_rules_from_tree(self, tree) -> List[str]:
        """Extract rules from a decision tree."""
        # Placeholder for rule extraction logic
        return ["rule1", "rule2"]  # Dummy implementation
        
    def _evaluate_rules(self, X: np.ndarray, rules: List[str]) -> np.ndarray:
        """Evaluate rules on data."""
        # Placeholder for rule evaluation logic
        return np.random.rand(X.shape[0], len(rules))  # Dummy implementation
        
    def fit(self, X, multi_state):
        """
        Fit the rule-based multi-state model.
        
        Parameters
        ----------
        X : array-like
            Covariate matrix
        multi_state : MultiState
            Multi-state data object
        """
        # If multi_state is a dict, map its keys
        if isinstance(multi_state, dict):
            mapped = self._map_transition_keys(multi_state)
        else:
            mapped = multi_state

        transition_times = {}
        transition_events = {}
        # For every transition in the model, check for data in mapped dict
        for transition in self.state_manager.transitions:
            if mapped and transition in mapped and 'times' in mapped[transition] and 'events' in mapped[transition]:
                transition_times[transition] = mapped[transition]['times']
                transition_events[transition] = mapped[transition]['events']
        # Only call _estimate_baseline_hazards for transitions with data
        if transition_times and transition_events:
            filtered_times = {k: v for k, v in transition_times.items() if k in transition_events}
            filtered_events = {k: v for k, v in transition_events.items() if k in transition_times}
            self._estimate_baseline_hazards(
                transition_times=filtered_times,
                transition_events=filtered_events
            )
        self.is_fitted_ = True
        return self
        
    def get_feature_importances(self, transition: Tuple[int, int]) -> np.ndarray:
        """
        Get feature importances for a specific transition.
        
        Parameters
        ----------
        transition : tuple
            The transition to get importances for
            
        Returns
        -------
        np.ndarray
            Array of feature importances
        """
        if not hasattr(self, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importances")
            
        if transition not in self.feature_importances_:
            raise ValueError(f"No feature importances available for transition {transition}")
            
        return self.feature_importances_[transition]

    def get_variable_importances(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Get the importance scores for each variable in the model for all transitions.
        
        Returns
        -------
        dict
            Dictionary mapping transitions to dictionaries of variable importances
        """
        if not hasattr(self, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting variable importances")
            
        # Get feature names from the preprocessor if available
        feature_names = getattr(self, 'feature_names_', 
                              [f'feature_{i}' for i in range(len(next(iter(self.feature_importances_.values()))))])
        
        # Create dictionary of variable importances for each transition
        importances = {}
        for transition in self.state_manager.transitions:
            if transition in self.feature_importances_:
                # Get importances for this transition
                transition_importances = self.feature_importances_[transition]
                
                # Create dictionary mapping feature names to importances
                transition_importance_dict = dict(zip(feature_names, transition_importances))
                
                # Sort by importance in descending order
                importances[transition] = dict(sorted(transition_importance_dict.items(), 
                                                    key=lambda x: x[1], 
                                                    reverse=True))
        
        return importances

    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        target_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict cumulative incidence function for a target state.
        
        Parameters
        ----------
        X : array-like
            Covariate values
        times : array-like
            Times at which to predict probabilities
        target_state : str or int
            Target state for which to compute cumulative incidence
            
        Returns
        -------
        np.ndarray
            Predicted cumulative incidence values
        """
        target_idx = self.state_manager.to_internal_index(target_state)
        state_probs = self.predict_state_occupation(X, times, initial_state=0)
        return state_probs[target_idx]