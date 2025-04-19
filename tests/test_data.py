import numpy as np
import pandas as pd
import pytest
from ruletimer.data import Survival, CompetingRisks, MultiState

def test_survival_data_initialization():
    """Test initialization of Survival data structure"""
    time = np.array([1, 2, 3, 4])
    event = np.array([1, 0, 1, 0])
    data = Survival(time, event)
    assert np.array_equal(data.time, time)
    assert np.array_equal(data.event, event)

def test_survival_data_validation():
    """Test validation of Survival data"""
    # Test invalid time values
    with pytest.raises(ValueError):
        Survival(np.array([-1, 2, 3]), np.array([1, 0, 1]))
    
    # Test invalid event values
    with pytest.raises(ValueError):
        Survival(np.array([1, 2, 3]), np.array([2, 0, 1]))
    
    # Test mismatched lengths
    with pytest.raises(ValueError):
        Survival(np.array([1, 2, 3]), np.array([1, 0]))

def test_competing_risks_initialization():
    """Test initialization of CompetingRisks data structure"""
    time = np.array([1, 2, 3, 4])
    event = np.array([0, 1, 2, 0])
    data = CompetingRisks(time, event)
    assert np.array_equal(data.time, time)
    assert np.array_equal(data.event, event)

def test_competing_risks_validation():
    """Test validation of CompetingRisks data"""
    # Test invalid time values
    with pytest.raises(ValueError):
        CompetingRisks(np.array([-1, 2, 3]), np.array([0, 1, 2]))
    
    # Test invalid event values
    with pytest.raises(ValueError):
        CompetingRisks(np.array([1, 2, 3]), np.array([-1, 0, 1]))
    
    # Test mismatched lengths
    with pytest.raises(ValueError):
        CompetingRisks(np.array([1, 2, 3]), np.array([0, 1]))

def test_multi_state_initialization():
    """Test initialization of MultiState data structure"""
    patient_id = np.array([1, 1, 2, 2])
    start_time = np.array([0, 1, 0, 2])
    end_time = np.array([1, 2, 2, 3])
    start_state = np.array([1, 2, 1, 2])  # States start from 1
    end_state = np.array([2, 3, 2, 3])    # States start from 1
    
    data = MultiState(start_time, end_time, start_state, end_state, patient_id)
    assert np.array_equal(data.patient_id, patient_id)
    assert np.array_equal(data.start_time, start_time)
    assert np.array_equal(data.end_time, end_time)
    assert np.array_equal(data.start_state, start_state)
    assert np.array_equal(data.end_state, end_state)

def test_multi_state_validation():
    """Test validation of MultiState data"""
    # Test invalid time ordering
    with pytest.raises(ValueError):
        MultiState(
            np.array([1, 2]),  # start_time
            np.array([2, 1]),  # end_time (invalid: not ordered)
            np.array([1, 1]),  # start_state
            np.array([2, 2]),  # end_state
            patient_id=np.array([1, 1])
        )  
    # Test invalid state transitions (same start and end state)
    with pytest.raises(ValueError):
        MultiState(
            np.array([1, 1]),  # start_time
            np.array([0, 1]),  # end_time
            np.array([1, 2]),  # start_state
            np.array([1, 1]),  # end_state (invalid: same as start state)
            np.array([1, 1])   # patient_id
        )
    
    # Test mismatched lengths
    with pytest.raises(ValueError):
        MultiState(
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([2])  # Mismatched length
        )

def test_multi_state_patient_ordering():
    """Test patient-specific time ordering in MultiState data"""
    patient_id = np.array([1, 1, 2, 2])
    start_time = np.array([0, 1, 0, 2])  # Ordered within each patient
    end_time = np.array([1, 2, 2, 3])
    start_state = np.array([1, 2, 1, 2])  # States start from 1
    end_state = np.array([2, 3, 2, 3])    # States start from 1
    
    data = MultiState(start_time, end_time, start_state, end_state, patient_id)
    # Check if times are strictly increasing within each patient
    assert np.all(np.diff(data.start_time[data.patient_id == 1]) > 0)
    assert np.all(np.diff(data.start_time[data.patient_id == 2]) > 0) 