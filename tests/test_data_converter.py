import numpy as np
import pandas as pd
import pytest
from ruletimer.data import MultiState
from ruletimer.data.data_converter import MultiStateDataConverter

# Test data for person-period format
def create_person_period_data():
    return pd.DataFrame({
        'ID': [1, 1, 1, 2, 2],
        'Time': [1, 2, 3, 1, 2],
        'State': ['A', 'B', 'B', 'A', 'A'],
        'Censored': [0, 0, 1, 0, 1]
    })

# Test data for transition format
def create_transition_data():
    return pd.DataFrame({
        'ID': [1, 1, 2],
        'FromState': ['A', 'B', 'A'],
        'ToState': ['B', 'B', 'A'],
        'StartTime': [1, 2, 1],
        'EndTime': [2, 3, 2],
        'Censored': [0, 1, 1]
    })

# Test data for long format
def create_long_format_data():
    return pd.DataFrame({
        'ID': [1, 1, 1, 2, 2],
        'Time': [1, 2, 3, 1, 2],
        'StateA': [1, 0, 0, 1, 1],
        'StateB': [0, 1, 1, 0, 0],
        'Censored': [0, 0, 1, 0, 1]
    })

# Test data for wide format
def create_wide_format_data():
    return pd.DataFrame({
        'ID': [1, 2],
        'State_T1': ['A', 'A'],
        'State_T2': ['B', 'A'],
        'State_T3': ['B', None],
        'CensorTime': [3, 2],
        'CensorState': ['B', 'A']
    })

def create_interval_censored_data():
    """Create test data for interval censoring"""
    return pd.DataFrame({
        'ID': [1, 1, 2, 2],
        'Time': [1, 2, 1, 2],
        'State': ['A', None, 'A', None],
        'Censored': [0, 1, 0, 1],
        'NextKnownState': ['B', None, 'A', None],
        'NextKnownTime': [3, None, 4, None]
    })

def create_counting_process_data():
    """Create test data in counting process format"""
    return pd.DataFrame({
        'ID': [1, 1, 1, 2, 2],
        'FromState': ['A', 'B', 'B', 'A', 'A'],
        'ToState': ['B', 'C', 'C', 'B', 'C'],
        'Count': [1, 0, 0, 0, 0],
        'Exposure': [1, 2, 3, 1, 2],
        'Censored': [0, 1, 1, 1, 1]
    })

def create_multiple_transitions_data():
    """Create test data with multiple state transitions"""
    return pd.DataFrame({
        'ID': [1, 1, 1, 1, 2, 2],
        'Time': [1, 2, 3, 4, 1, 2],
        'State': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Censored': [0, 0, 0, 1, 0, 1]
    })

def test_from_person_period():
    """Test conversion from person-period format to MultiState"""
    data = create_person_period_data()
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # Three transitions: A->B, B->B(censored), A->A(censored)
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # 0 for censored

def test_from_transition_format():
    """Test conversion from transition format to MultiState"""
    data = create_transition_data()
    msm_data = MultiStateDataConverter.from_transition_format(
        data=data,
        id_col='ID',
        from_state_col='FromState',
        to_state_col='ToState',
        start_time_col='StartTime',
        end_time_col='EndTime',
        censored_col='Censored'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # Three transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # 0 for censored

def test_from_long_format():
    """Test conversion from long format to MultiState"""
    data = create_long_format_data()
    msm_data = MultiStateDataConverter.from_long_format(
        data=data,
        id_col='ID',
        time_col='Time',
        state_cols=['StateA', 'StateB'],
        censored_col='Censored'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # Three transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # StateA=1, StateB=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # 0 for censored

def test_from_wide_format():
    """Test conversion from wide format to MultiState"""
    data = create_wide_format_data()
    msm_data = MultiStateDataConverter.from_wide_format(
        data=data,
        id_col='ID',
        time_points=['State_T1', 'State_T2', 'State_T3'],
        state_cols=['State_T1', 'State_T2', 'State_T3'],
        censor_time_col='CensorTime'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 2  # Two transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 1]))  # A=1
    assert np.array_equal(msm_data.end_state, np.array([2, 0]))  # 0 for censored

def test_to_person_period():
    """Test conversion from MultiState to person-period format"""
    msm_data = MultiState(
        patient_id=np.array([1, 1]),
        start_time=np.array([1, 2]),
        end_time=np.array([2, 3]),
        start_state=np.array([1, 2]),
        end_state=np.array([2, 0])
    )
    
    df = MultiStateDataConverter.to_person_period(msm_data)
    
    # Verify the conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Two transitions, each with start and end
    assert np.array_equal(df['ID'].values, np.array([1, 1, 1, 1]))
    assert np.array_equal(df['Time'].values, np.array([1, 2, 2, 3]))
    assert np.array_equal(df['State'].values, np.array([1, 2, 2, 2]))
    assert np.array_equal(df['Censored'].values, np.array([0, 0, 0, 1]))

def test_to_transition_format():
    """Test conversion from MultiState to transition format"""
    msm_data = MultiState(
        patient_id=np.array([1, 1]),
        start_time=np.array([1, 2]),
        end_time=np.array([2, 3]),
        start_state=np.array([1, 2]),
        end_state=np.array([2, 0])
    )
    
    df = MultiStateDataConverter.to_transition_format(msm_data)
    
    # Verify the conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert np.array_equal(df['ID'].values, np.array([1, 1]))
    assert np.array_equal(df['FromState'].values, np.array([1, 2]))
    assert np.array_equal(df['ToState'].values, np.array([2, 0]))
    assert np.array_equal(df['StartTime'].values, np.array([1, 2]))
    assert np.array_equal(df['EndTime'].values, np.array([2, 3]))
    assert np.array_equal(df['Censored'].values, np.array([0, 1]))

def test_to_long_format():
    """Test conversion from MultiState to long format"""
    msm_data = MultiState(
        patient_id=np.array([1, 1]),
        start_time=np.array([1, 2]),
        end_time=np.array([2, 3]),
        start_state=np.array([1, 2]),
        end_state=np.array([2, 0])
    )
    
    df = MultiStateDataConverter.to_long_format(msm_data, state_names=['A', 'B'])
    
    # Verify the conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Two transitions, each with start and end
    assert np.array_equal(df['ID'].values, np.array([1, 1, 1, 1]))
    assert np.array_equal(df['Time'].values, np.array([1, 2, 2, 3]))
    assert np.array_equal(df['Censored'].values, np.array([0, 0, 0, 1]))
    assert np.array_equal(df['StateA'].values, np.array([1, 0, 0, 0]))
    assert np.array_equal(df['StateB'].values, np.array([0, 1, 1, 1]))

def test_round_trip_conversion():
    """Test round-trip conversion between formats"""
    # Start with person-period format
    data = create_person_period_data()
    
    # Convert to MultiState
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )
    
    # Convert back to person-period
    df = MultiStateDataConverter.to_person_period(msm_data)
    
    # Verify the round-trip conversion preserves the structure
    assert len(df) == 6  # Three transitions, each with start and end
    assert np.array_equal(df['ID'].values, np.array([1, 1, 1, 1, 2, 2]))
    assert np.array_equal(df['Time'].values, np.array([1, 2, 2, 3, 1, 2]))
    assert np.array_equal(df['Censored'].values, np.array([0, 0, 0, 1, 0, 1]))

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    # Test invalid person-period data
    data = pd.DataFrame({
        'ID': [1, 1],
        'Time': [2, 1],  # Invalid: times not ordered
        'State': ['A', 'B'],
        'Censored': [0, 0]
    })
    
    with pytest.raises(ValueError):
        MultiStateDataConverter.from_person_period(
            data=data,
            id_col='ID',
            time_col='Time',
            state_col='State',
            censored_col='Censored'
        )
    
    # Test missing columns
    data = pd.DataFrame({
        'ID': [1, 1],
        'Time': [1, 2],
        'State': ['A', 'B']
    })
    
    with pytest.raises(KeyError):
        MultiStateDataConverter.from_person_period(
            data=data,
            id_col='ID',
            time_col='Time',
            state_col='State',
            censored_col='Censored'  # Missing column
        )

def test_interval_censoring():
    """Test handling of interval censored data"""
    data = create_interval_censored_data()
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 2  # Two transitions (one per patient)
    assert np.array_equal(msm_data.patient_id, np.array([1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 1]))  # A=1
    assert np.array_equal(msm_data.end_state, np.array([0, 0]))  # Both censored

def test_counting_process():
    """Test conversion from counting process format"""
    data = create_counting_process_data()
    msm_data = MultiStateDataConverter.from_counting_process(
        data=data,
        id_col='ID',
        from_state_col='FromState',
        to_state_col='ToState',
        count_col='Count',
        exposure_col='Exposure',
        censored_col='Censored'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # One transition + two censored observations
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))  # Patient 1's transition and censoring, Patient 2's censoring
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # One transition (A->B) and two censored

def test_multiple_transitions():
    """Test handling of multiple state transitions"""
    data = create_multiple_transitions_data()
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )
    
    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 4  # Four transitions for patient 1, one for patient 2
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 3, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 4, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 1, 0, 0]))  # Last transitions censored

def test_time_ordering_validation():
    """Test validation of time ordering"""
    # Create data with invalid time ordering
    data = pd.DataFrame({
        'ID': [1, 1],
        'Time': [2, 1],  # Times in wrong order
        'State': ['A', 'B'],
        'Censored': [0, 0]
    })
    
    # Verify that validation raises error
    with pytest.raises(ValueError, match="Times must be non-decreasing within each patient"):
        MultiStateDataConverter.from_person_period(
            data=data,
            id_col='ID',
            time_col='Time',
            state_col='State',
            censored_col='Censored'
        )

def test_missing_values():
    """Test handling of missing values"""
    data = pd.DataFrame({
        'ID': [1, 1, 2],
        'Time': [1, 2, 1],
        'State': ['A', None, 'B'],
        'Censored': [0, 1, 0]
    })
    
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )
    
    # Verify that missing values are handled correctly
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 2  # Two transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([0, 0]))  # Both censored 