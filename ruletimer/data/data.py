"""
Data structures for time-to-event analysis
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional

class Survival:
    def __getitem__(self, idx):
        return Survival(self.time[idx], self.event[idx])
    """Class for standard survival data"""
    
    def __init__(self, time: Union[np.ndarray, pd.Series], 
                 event: Union[np.ndarray, pd.Series]):
        """
        Initialize survival data
        
        Parameters
        ----------
        time : array-like
            Time to event or censoring
        event : array-like
            Event indicator (1 for event, 0 for censored)
        """
        self.time = np.asarray(time)
        self.event = np.asarray(event)
        self._validate()
    
    def _validate(self):
        """Validate the survival data"""
        if len(self.time) != len(self.event):
            raise ValueError("Time and event arrays must have the same length")
        if not np.all(self.time >= 0):
            raise ValueError("All times must be non-negative")
        if not np.all(np.isin(self.event, [0, 1])):
            raise ValueError("Event indicators must be 0 or 1")

class CompetingRisks:
    def __getitem__(self, idx):
        return CompetingRisks(self.time[idx], self.event[idx])
    """Class for competing risks data"""
    
    def __init__(self, time: Union[np.ndarray, pd.Series], 
                 event: Union[np.ndarray, pd.Series]):
        """
        Initialize competing risks data
        
        Parameters
        ----------
        time : array-like
            Time to event or censoring
        event : array-like
            Event type indicator (0 for censored, positive integers for different event types)
        """
        self.time = np.asarray(time)
        self.event = np.asarray(event)
        self._validate()
    
    def _validate(self):
        """Validate the competing risks data"""
        if len(self.time) != len(self.event):
            raise ValueError("Time and event arrays must have the same length")
        if not np.all(self.time >= 0):
            raise ValueError("All times must be non-negative")
        if not np.all(self.event >= 0):
            raise ValueError("Event types must be non-negative integers")
        if not np.all(np.isin(self.event, [0, 1, 2])):
            raise ValueError("Event indicators must be 0, 1, or 2")

class MultiState:
    def __getitem__(self, idx):
        return MultiState(self.start_time[idx], self.end_time[idx], self.start_state[idx], self.end_state[idx], self.patient_id[idx])
    """Class for multi-state data"""
    
    def __init__(self, start_time: Union[np.ndarray, pd.Series],
                 end_time: Union[np.ndarray, pd.Series],
                 start_state: Union[np.ndarray, pd.Series],
                 end_state: Union[np.ndarray, pd.Series],
                 patient_id: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Initialize multi-state data
        
        Parameters
        ----------
        start_time : array-like
            Time at which the observation starts
        end_time : array-like
            Time at which the observation ends
        start_state : array-like
            State at the start of observation (non-negative integers)
        end_state : array-like
            State at the end of observation (non-negative integers)
        patient_id : array-like, optional
            Unique identifier for each patient. If not provided, will be generated automatically.
        """
        self.start_time = np.asarray(start_time)
        self.end_time = np.asarray(end_time)
        self.start_state = np.asarray(start_state)
        self.end_state = np.asarray(end_state)
        
        if patient_id is None:
            self.patient_id = np.arange(len(start_time))
        else:
            self.patient_id = np.asarray(patient_id)
            
        self._validate()
    
    def _validate(self):
        """Validate the multi-state data"""
        if not (len(self.patient_id) == len(self.start_time) == len(self.end_time) == 
                len(self.start_state) == len(self.end_state)):
            raise ValueError("All arrays must have the same length")
        
        # Validate patient IDs
        if not np.all(self.patient_id >= 0):
            raise ValueError("Patient IDs must be non-negative integers")
        
        # Validate times
        if not np.all(self.start_time >= 0):
            raise ValueError("All start times must be non-negative")
        if not np.all(self.end_time >= 0):
            raise ValueError("All end times must be non-negative")
            
        # Validate states
        if not np.all(self.start_state >= 0):
            raise ValueError("States must be non-negative integers")
        if not np.all(self.end_state >= 0):
            raise ValueError("States must be non-negative integers")
        
        # Validate time ordering within patients
        for pid in np.unique(self.patient_id):
            mask = self.patient_id == pid
            if len(self.start_time[mask]) > 1:  # Only check if more than one observation
                if not np.all(np.diff(self.start_time[mask]) > 0):
                    raise ValueError(f"Start times must be strictly increasing for patient {pid}")
                if not np.all(np.diff(self.end_time[mask]) > 0):
                    raise ValueError(f"End times must be strictly increasing for patient {pid}")
        # Validate that start_state != end_state for all transitions
        if np.any(self.start_state == self.end_state):
            raise ValueError("Start state and end state must differ for all transitions.")