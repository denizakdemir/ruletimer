"""
Unified prediction utilities for multi-state time-to-event models.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

class UnifiedTransitionCalculator:
    """Calculate transition probabilities for multi-state models."""
    
    @staticmethod
    def calculate_transition_matrix(
        model,
        X: np.ndarray,
        times: np.ndarray,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate transition probability matrix P(s,t) for each time point.
        
        Parameters
        ----------
        model : RuleMultiState
            Fitted multi-state model
        X : array-like
            Feature matrix
        times : array-like
            Time points for prediction
        include_ci : bool, default=True
            Whether to include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        n_bootstrap : int, default=100
            Number of bootstrap samples if include_ci=True
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'times': Time points
            - 'states': State indices
            - 'matrix': Array of shape (n_samples, n_states, n_states, n_times)
            - 'lower': Lower CI if include_ci=True
            - 'upper': Upper CI if include_ci=True
        """
        # Get model information
        states = model.states_
        n_states = len(states)
        n_samples = X.shape[0]
        n_times = len(times)
        
        # Initialize transition matrix
        P = np.zeros((n_samples, n_states, n_states, n_times))
        
        # Calculate transition probabilities for each pair of states
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                if from_state != to_state and (from_state, to_state) in model.transitions_:
                    # Direct transition
                    P[:, i, j] = model.predict_transition_probability(
                        X, times, from_state, to_state
                    )
                elif from_state == to_state:
                    # Staying in same state (complementary probability)
                    next_states = model._get_next_states(from_state)
                    if next_states:
                        for next_state in next_states:
                            next_idx = list(states).index(next_state)
                            P[:, i, i] = 1 - np.sum(P[:, i, :], axis=1)
                    else:
                        # Absorbing state
                        P[:, i, i] = 1
        
        # Package results
        result = {
            'times': times,
            'states': states,
            'matrix': P
        }
        
        # Calculate confidence intervals if requested
        if include_ci:
            ci_result = UnifiedTransitionCalculator._calculate_transition_matrix_ci(
                model, X, times, states, alpha, n_bootstrap
            )
            result.update(ci_result)
        
        return result
    
    @staticmethod
    def _calculate_transition_matrix_ci(
        model,
        X: np.ndarray,
        times: np.ndarray,
        states: List,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate confidence intervals for transition matrix using bootstrap."""
        n_states = len(states)
        n_samples = X.shape[0]
        n_times = len(times)
        
        # Initialize arrays for bootstrap samples
        P_boots = np.zeros((n_bootstrap, n_samples, n_states, n_states, n_times))
        
        # Perform bootstrap
        for b in range(n_bootstrap):
            # Sample with replacement
            sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[sample_idx]
            
            # Calculate transition matrix for this bootstrap sample
            for i, from_state in enumerate(states):
                for j, to_state in enumerate(states):
                    if from_state != to_state and (from_state, to_state) in model.transitions_:
                        # Direct transition
                        P_boots[b, :, i, j] = model.predict_transition_probability(
                            X_boot, times, from_state, to_state
                        )
                    elif from_state == to_state:
                        # Staying in same state (complementary probability)
                        next_states = model._get_next_states(from_state)
                        if next_states:
                            for next_state in next_states:
                                next_idx = list(states).index(next_state)
                                P_boots[b, :, i, i] = 1 - np.sum(P_boots[b, :, i, :], axis=1)
                        else:
                            # Absorbing state
                            P_boots[b, :, i, i] = 1
        
        # Calculate confidence intervals
        lower = np.quantile(P_boots, alpha/2, axis=0)
        upper = np.quantile(P_boots, 1-alpha/2, axis=0)
        
        return {
            'lower': lower,
            'upper': upper
        }


class UnifiedPredictionCalculator:
    """Unified calculator for various predictions from multi-state models."""
    
    @staticmethod
    def survival_from_multistate(
        model,
        X: np.ndarray,
        times: np.ndarray,
        initial_state: int = 1,
        death_state: int = 2,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate survival probabilities from multi-state model.
        
        Parameters
        ----------
        model : RuleMultiState
            Fitted multi-state model
        X : array-like
            Feature matrix
        times : array-like
            Time points for prediction
        initial_state : int, default=1
            Initial state
        death_state : int, default=2
            Absorbing state representing death
        include_ci : bool, default=True
            Whether to include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        n_bootstrap : int, default=100
            Number of bootstrap samples if include_ci=True
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'times': Time points
            - 'survival': Array of shape (n_samples, n_times)
            - 'lower': Lower CI if include_ci=True
            - 'upper': Upper CI if include_ci=True
        """
        # Get state occupation probabilities
        state_occupation = model.predict_state_occupation(X, times)
        
        # Survival is probability of not being in death state
        survival = 1 - state_occupation[death_state]
        
        # Package results
        result = {
            'times': times,
            'survival': survival
        }
        
        # Calculate confidence intervals if requested
        if include_ci:
            # Initialize bootstrap samples
            n_samples = X.shape[0]
            n_times = len(times)
            survival_boots = np.zeros((n_bootstrap, n_samples, n_times))
            
            # Perform bootstrap
            for b in range(n_bootstrap):
                # Sample with replacement
                sample_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[sample_idx]
                
                # Calculate state occupation for this bootstrap sample
                state_occ_boot = model.predict_state_occupation(X_boot, times)
                
                # Calculate survival
                survival_boots[b] = 1 - state_occ_boot[death_state]
            
            # Calculate confidence intervals
            result['lower'] = np.quantile(survival_boots, alpha/2, axis=0)
            result['upper'] = np.quantile(survival_boots, 1-alpha/2, axis=0)
        
        return result
    
    @staticmethod
    def cumulative_incidence_from_multistate(
        model,
        X: np.ndarray,
        times: np.ndarray,
        initial_state: int = 1,
        target_states: Optional[List[int]] = None,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate cumulative incidence functions from multi-state model.
        
        Parameters
        ----------
        model : RuleMultiState
            Fitted multi-state model
        X : array-like
            Feature matrix
        times : array-like
            Time points for prediction
        initial_state : int, default=1
            Initial state
        target_states : list, optional
            Target states for CIF (if None, uses all states except initial)
        include_ci : bool, default=True
            Whether to include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        n_bootstrap : int, default=100
            Number of bootstrap samples if include_ci=True
            
        Returns
        -------
        dict
            Dictionary with keys for each target state, containing:
            - 'times': Time points
            - 'cif': Array of shape (n_samples, n_times)
            - 'lower': Lower CI if include_ci=True
            - 'upper': Upper CI if include_ci=True
        """
        # Get all states
        states = model.states_
        
        # If target states not specified, use all states except initial
        if target_states is None:
            target_states = [s for s in states if s != initial_state]
        
        # Get state occupation probabilities
        state_occupation = model.predict_state_occupation(X, times)
        
        # Initialize results
        result = {}
        
        # For each target state, CIF is just the state occupation probability
        for state in target_states:
            result[state] = {
                'times': times,
                'cif': state_occupation[state]
            }
        
        # Calculate confidence intervals if requested
        if include_ci:
            # Initialize bootstrap samples
            n_samples = X.shape[0]
            n_times = len(times)
            cif_boots = {
                state: np.zeros((n_bootstrap, n_samples, n_times))
                for state in target_states
            }
            
            # Perform bootstrap
            for b in range(n_bootstrap):
                # Sample with replacement
                sample_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[sample_idx]
                
                # Calculate state occupation for this bootstrap sample
                state_occ_boot = model.predict_state_occupation(X_boot, times)
                
                # Calculate CIF for each target state
                for state in target_states:
                    cif_boots[state][b] = state_occ_boot[state]
            
            # Calculate confidence intervals for each state
            for state in target_states:
                result[state]['lower'] = np.quantile(cif_boots[state], alpha/2, axis=0)
                result[state]['upper'] = np.quantile(cif_boots[state], 1-alpha/2, axis=0)
        
        return result 