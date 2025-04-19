"""
Unified hazard estimation module for all time-to-event models.
"""
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from scipy.interpolate import interp1d

class HazardEstimator:
    """Unified hazard estimation for all time-to-event models."""
    
    @staticmethod
    def estimate_baseline_hazard(
        times: np.ndarray,
        events: np.ndarray,
        weights: Optional[np.ndarray] = None,
        method: str = "nelson-aalen"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate baseline hazard using various methods.
        
        Parameters
        ----------
        times : np.ndarray
            Event/censoring times
        events : np.ndarray
            Event indicators (1 if event, 0 if censored)
        weights : np.ndarray, optional
            Case weights for hazard estimation
        method : str
            Estimation method: "nelson-aalen" or "parametric"
            
        Returns
        -------
        unique_times : np.ndarray
            Unique event times
        baseline_hazard : np.ndarray
            Estimated baseline hazard at unique times
        """
        if method not in ["nelson-aalen", "parametric"]:
            raise ValueError("Method must be 'nelson-aalen' or 'parametric'")
            
        # Sort times and get unique event times
        order = np.argsort(times)
        times = times[order]
        events = events[order]
        if weights is not None:
            weights = weights[order]
        else:
            weights = np.ones_like(times)
            
        unique_times = np.unique(times[events == 1])
        n_times = len(unique_times)
        baseline_hazard = np.zeros(n_times)
        
        if method == "nelson-aalen":
            # Nelson-Aalen estimator
            at_risk = np.zeros(n_times)
            events_at_time = np.zeros(n_times)
            
            for i, t in enumerate(unique_times):
                at_risk[i] = np.sum(weights[times >= t])
                events_at_time[i] = np.sum(weights[
                    (times == t) & (events == 1)
                ])
                
            # Compute hazard with smoothing for stability
            baseline_hazard = events_at_time / (at_risk + 1e-8)
            
        else:  # parametric
            # Implement parametric estimation (e.g., Weibull)
            # This is a placeholder for actual parametric implementation
            pass
            
        return unique_times, baseline_hazard
    
    @staticmethod
    def estimate_cumulative_hazard(
        times: np.ndarray, baseline_hazard: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate cumulative hazard using the trapezoidal rule.

        Parameters
        ----------
        times : np.ndarray
            Array of unique event times.
        baseline_hazard : np.ndarray
            Array of baseline hazard values corresponding to times.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - Array of prediction times
            - Array of cumulative hazard values
        """
        prediction_times = times.copy()
        cumulative_hazard = np.zeros_like(prediction_times, dtype=float)
        
        # Set first time point's cumulative hazard to baseline hazard
        cumulative_hazard[0] = baseline_hazard[0]
        
        # Calculate cumulative hazard for remaining time points
        for i in range(1, len(prediction_times)):
            # Find indices of times up to current prediction time
            mask = times <= prediction_times[i]
            cumulative_hazard[i] = np.sum(baseline_hazard[mask])
            
        return prediction_times, cumulative_hazard
    
    @staticmethod
    def transform_hazard(
        cumulative_hazard: np.ndarray,
        transform: str = "exp"
    ) -> np.ndarray:
        """
        Transform cumulative hazard to survival or CIF.
        
        Parameters
        ----------
        cumulative_hazard : np.ndarray
            Cumulative hazard values
        transform : str
            Transformation type: "exp" for survival, "cif" for CIF
            
        Returns
        -------
        np.ndarray
            Transformed values
        """
        if transform == "exp":
            return np.exp(-cumulative_hazard)
        elif transform == "cif":
            return 1 - np.exp(-cumulative_hazard)
        else:
            raise ValueError("Transform must be 'exp' or 'cif'") 