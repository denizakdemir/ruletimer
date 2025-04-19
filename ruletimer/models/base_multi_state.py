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
        Predict cumulative transition-specific hazard.
        
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
            Predicted cumulative hazard values
        """
        hazard = self.predict_transition_hazard(X, times, from_state, to_state)
        cumulative_hazard = np.zeros_like(hazard)
        
        for i in range(len(X)):
            _, cum_haz = self.hazard_estimator.estimate_cumulative_hazard(
                times, hazard[i]
            )
            cumulative_hazard[i] = cum_haz
            
        return cumulative_hazard
    
    def predict_transition_probability(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict transition probability between states.
        
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
            Predicted transition probabilities
        """
        cumulative_hazard = self.predict_cumulative_hazard(
            X, times, from_state, to_state
        )
        
        # For direct transitions, use CIF transform
        return self.hazard_estimator.transform_hazard(
            cumulative_hazard, transform="cif"
        )
    
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
        initial_idx = self.state_manager.to_internal_index(initial_state)
        n_states = len(self.state_manager.states)
        n_samples = len(X)
        
        # Initialize with all probability mass in initial state
        occupation = {
            state: np.zeros((n_samples, len(times)))
            for state in range(n_states)
        }
        occupation[initial_idx][:, 0] = 1.0
        
        # Compute state occupation probabilities using Aalen-Johansen
        for t_idx in range(1, len(times)):
            # First compute all transitions
            for from_state in range(n_states):
                if occupation[from_state][:, t_idx-1].sum() > 0:
                    # Copy previous probabilities
                    occupation[from_state][:, t_idx] = occupation[from_state][:, t_idx-1]
                    
                    # Compute transitions to other states
                    for to_state in range(n_states):
                        if from_state != to_state and self.state_manager.validate_transition(from_state, to_state):
                            # Get transition probability
                            trans_prob = self.predict_transition_probability(
                                X,
                                times[t_idx-1:t_idx+1],
                                from_state,
                                to_state
                            )[:, -1]
                            
                            # Ensure probabilities are between 0 and 1
                            trans_prob = np.clip(trans_prob, 0, 1)
                            
                            # Update occupation probability
                            occupation[to_state][:, t_idx] += (
                                occupation[from_state][:, t_idx-1] * trans_prob
                            )
                            
                            # Subtract from source state
                            occupation[from_state][:, t_idx] -= (
                                occupation[from_state][:, t_idx-1] * trans_prob
                            )
            
            # Normalize probabilities to ensure they sum to 1
            total_prob = np.sum([occupation[state][:, t_idx] for state in range(n_states)], axis=0)
            for state in range(n_states):
                occupation[state][:, t_idx] = np.clip(
                    occupation[state][:, t_idx] / total_prob,
                    0, 1
                )
        
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
        # Generate rules and fit transition-specific models
        for transition in self.transitions_:
            # Extract transition-specific data
            mask = (multi_state.start_state == transition[0]) & (
                (multi_state.end_state == transition[1]) |
                (multi_state.end_state == -1)  # Include censored
            )
            X_trans = X[mask]
            y_trans = (multi_state.end_state[mask] == transition[1]).astype(int)
            
            # Generate rules
            rules = self._generate_rules(X_trans, y_trans)
            self.rules_[transition] = rules
            
            # Evaluate rules
            rule_matrix = self._evaluate_rules(X_trans, rules)
            
            # Fit elastic net
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state
            )
            model.fit(rule_matrix, y_trans)
            
            self.rule_weights_[transition] = model.coef_
            self.transition_models_[transition] = model
            
            # Estimate baseline hazard
            times = multi_state.end_time[mask] - multi_state.start_time[mask]
            events = (multi_state.end_state[mask] == transition[1]).astype(int)
            baseline_times, baseline_hazard = self.hazard_estimator.estimate_baseline_hazard(
                times=times,
                events=events,
                method=self.hazard_method
            )
            self.baseline_hazards_[transition] = (baseline_times, baseline_hazard)
            
        self.is_fitted_ = True
        return self
        
    def get_feature_importances(self, transition: Tuple[int, int]) -> np.ndarray:
        """Get feature importances for a specific transition."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importances")
            
        if transition not in self.rule_weights_:
            raise ValueError(f"No model for transition {transition}")
            
        return np.abs(self.rule_weights_[transition])

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