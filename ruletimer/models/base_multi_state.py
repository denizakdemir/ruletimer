"""
Base class for multi-state time-to-event models.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from ruletimer.utils.hazard_estimation import HazardEstimator
from ruletimer.state_manager import StateManager
from ruletimer.time_handler import TimeHandler

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
        relative_hazard = self.transition_models_[transition].predict(X)
        
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
            for from_state in range(n_states):
                if occupation[from_state][:, t_idx-1].sum() > 0:
                    for to_state in range(n_states):
                        if self.state_manager.validate_transition(from_state, to_state):
                            # Get transition probability
                            trans_prob = self.predict_transition_probability(
                                X,
                                times[t_idx-1:t_idx+1],
                                from_state,
                                to_state
                            )[:, -1]
                            
                            # Update occupation probability
                            occupation[to_state][:, t_idx] += (
                                occupation[from_state][:, t_idx-1] * trans_prob
                            )
                            
                    # Remaining probability stays in current state
                    total_trans = sum(
                        occupation[to_state][:, t_idx]
                        for to_state in range(n_states)
                        if to_state != from_state
                    )
                    occupation[from_state][:, t_idx] = (
                        occupation[from_state][:, t_idx-1] - total_trans
                    )
        
        return occupation 