import numpy as np
from typing import Dict, Union, List
from scipy.interpolate import interp1d

class TimeHandler:
    """Unified time handling for all time-to-event models"""
    
    @staticmethod
    def validate_times(times: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Validate time values
        
        Parameters
        ----------
        times : array-like
            Time values to validate
            
        Returns
        -------
        np.ndarray
            Validated time values
            
        Raises
        ------
        ValueError
            If times are negative or non-numeric
        """
        try:
            times = np.asarray(times, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Times must be numeric")
            
        if np.any(times < 0):
            raise ValueError("Times must be non-negative")
            
        return times
    
    @staticmethod
    def get_time_points(data: object, n_points: int = 100) -> np.ndarray:
        """
        Get evenly spaced time points for prediction
        
        Parameters
        ----------
        data : object
            Data object with time attribute
        n_points : int, optional
            Number of time points to generate, by default 100
            
        Returns
        -------
        np.ndarray
            Array of evenly spaced time points
        """
        max_time = np.max(data.time)
        return np.linspace(0, max_time, n_points)
    
    @staticmethod
    def validate_time_dependent_covariates(covariates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Validate time-dependent covariates
        
        Parameters
        ----------
        covariates : dict
            Dictionary with 'time' and 'value' keys containing arrays
            
        Returns
        -------
        dict
            Validated covariates
            
        Raises
        ------
        ValueError
            If covariates are invalid
        """
        if not isinstance(covariates, dict):
            raise ValueError("Covariates must be a dictionary")
            
        if 'time' not in covariates or 'value' not in covariates:
            raise ValueError("Covariates must contain 'time' and 'value' keys")
            
        time = TimeHandler.validate_times(covariates['time'])
        value = np.asarray(covariates['value'])
        
        if len(time) != len(value):
            raise ValueError("Time and value arrays must have the same length")
            
        return {'time': time, 'value': value}
    
    @staticmethod
    def align_time_dependent_covariates(covariates: Dict[str, np.ndarray], 
                                      prediction_times: np.ndarray) -> np.ndarray:
        """
        Align time-dependent covariates with prediction times
        
        Parameters
        ----------
        covariates : dict
            Dictionary with 'time' and 'value' keys
        prediction_times : np.ndarray
            Times at which to predict covariate values
            
        Returns
        -------
        np.ndarray
            Aligned covariate values
        """
        covariates = TimeHandler.validate_time_dependent_covariates(covariates)
        prediction_times = TimeHandler.validate_times(prediction_times)
        
        # Create interpolation function
        interp = interp1d(covariates['time'], covariates['value'],
                         kind='linear', bounds_error=False,
                         fill_value=(covariates['value'][0], covariates['value'][-1]))
        
        return interp(prediction_times)
    
    @staticmethod
    def validate_time_intervals(start_times: np.ndarray, end_times: np.ndarray) -> None:
        """
        Validate time intervals
        
        Parameters
        ----------
        start_times : np.ndarray
            Start times
        end_times : np.ndarray
            End times
            
        Raises
        ------
        ValueError
            If intervals are invalid
        """
        start_times = TimeHandler.validate_times(start_times)
        end_times = TimeHandler.validate_times(end_times)
        
        if len(start_times) != len(end_times):
            raise ValueError("Start and end times must have the same length")
            
        if np.any(end_times < start_times):
            raise ValueError("End times must be greater than or equal to start times") 