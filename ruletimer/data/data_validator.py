"""
Data validation utilities for time-to-event data
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List

class DataValidator:
    """Validator for time-to-event data"""
    
    def validate_survival(self, 
                         time: Union[np.ndarray, pd.Series],
                         event: Union[np.ndarray, pd.Series]) -> bool:
        """Validate survival data
        
        Args:
            time: Array of event/censoring times
            event: Array of event indicators (0=censored, 1=event)
            
        Returns:
            True if data is valid, raises ValueError otherwise
        """
        # Convert to numpy arrays
        time = np.asarray(time)
        event = np.asarray(event)
        
        # Check lengths match
        if len(time) != len(event):
            raise ValueError("Time and event arrays must have same length")
        
        # Check for negative times
        if np.any(time < 0):
            raise ValueError("Event times cannot be negative")
        
        # Check event indicators are valid
        if not np.all(np.isin(event, [0, 1])):
            raise ValueError("Event indicators must be 0 or 1")
        
        return True
    
    def validate_competing_risks(self,
                               time: Union[np.ndarray, pd.Series],
                               event: Union[np.ndarray, pd.Series],
                               valid_events: Optional[List[int]] = None) -> bool:
        """Validate competing risks data
        
        Args:
            time: Array of event/censoring times
            event: Array of event types (0=censored, 1,2,...=event types)
            valid_events: Optional list of valid event types
            
        Returns:
            True if data is valid, raises ValueError otherwise
        """
        # Convert to numpy arrays
        time = np.asarray(time)
        event = np.asarray(event)
        
        # Check lengths match
        if len(time) != len(event):
            raise ValueError("Time and event arrays must have same length")
        
        # Check for negative times
        if np.any(time < 0):
            raise ValueError("Event times cannot be negative")
        
        # Check event types are non-negative
        if np.any(event < 0):
            raise ValueError("Event types cannot be negative")
        
        # Check event types are valid if specified
        if valid_events is not None:
            if not np.all(np.isin(event, valid_events)):
                raise ValueError(f"Event types must be one of {valid_events}")
        
        return True
    
    def validate_multi_state(self,
                           start_time: Union[np.ndarray, pd.Series],
                           end_time: Union[np.ndarray, pd.Series],
                           start_state: Union[np.ndarray, pd.Series],
                           end_state: Union[np.ndarray, pd.Series],
                           valid_transitions: Optional[List[tuple]] = None) -> bool:
        """Validate multi-state data
        
        Args:
            start_time: Array of transition start times
            end_time: Array of transition end times
            start_state: Array of starting states
            end_state: Array of ending states
            valid_transitions: Optional list of valid state transitions
            
        Returns:
            True if data is valid, raises ValueError otherwise
        """
        # Convert to numpy arrays
        start_time = np.asarray(start_time)
        end_time = np.asarray(end_time)
        start_state = np.asarray(start_state)
        end_state = np.asarray(end_state)
        
        # Check all arrays have same length
        lengths = [len(start_time), len(end_time), len(start_state), len(end_state)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All arrays must have same length")
        
        # Check for negative times
        if np.any(start_time < 0) or np.any(end_time < 0):
            raise ValueError("Times cannot be negative")
        
        # Check end times are not before start times
        if np.any(end_time < start_time):
            raise ValueError("End times cannot be before start times")
        
        # Check state transitions are valid if specified
        if valid_transitions is not None:
            transitions = list(zip(start_state, end_state))
            if not all(t in valid_transitions for t in transitions):
                raise ValueError(f"Invalid state transition found. Valid transitions are {valid_transitions}")
        
        return True 