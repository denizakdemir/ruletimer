"""
Tests for multi-state modeling
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from ruletimer.models.multi_state import RuleMultiState
from ruletimer.data import MultiState

def test_general_multi_state():
    """Test general multi-state model"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    patient_ids = np.arange(100)
    start_times = np.zeros(100)
    end_times = np.random.exponential(scale=1.0, size=100)
    start_states = np.random.choice([1, 2, 3], size=100)  # States start from 1
    
    # Ensure end states are different from start states
    end_states = np.array([
        np.random.choice([s for s in [1, 2, 3] if s != start_state])
        for start_state in start_states
    ])
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(model_type='general', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    probabilities = model.predict_state_occupation(X, times_pred)
    
    # Check output format
    assert isinstance(probabilities, dict)
    assert all(isinstance(probabilities[state], np.ndarray) for state in probabilities)
    assert all(probabilities[state].shape == (100, 10) for state in probabilities)
    
    # Check probabilities
    assert all(np.all(probabilities[state] >= 0) for state in probabilities)
    assert all(np.all(probabilities[state] <= 1) for state in probabilities)
    assert all(np.allclose(np.sum([probabilities[state][:, i] for state in probabilities], axis=0), 1.0)
              for i in range(10))

def test_illness_death_model():
    """Test illness-death model"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    patient_ids = np.arange(100)
    start_times = np.zeros(100)
    end_times = np.random.exponential(scale=1.0, size=100)
    start_states = np.random.choice([1, 2], size=100)  # States start from 1
    
    # Ensure valid transitions for illness-death model:
    # 1 (healthy) -> 2 (ill) or 3 (dead)
    # 2 (ill) -> 3 (dead)
    end_states = np.where(start_states == 1,
                         np.random.choice([2, 3], size=100),  # From healthy to ill or dead
                         3)  # From ill to dead
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(model_type='general', state_structure='illness-death', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    probabilities = model.predict_state_occupation(X, times_pred)
    
    # Check states
    assert set(probabilities.keys()) == {1, 2, 3}
    
    # Check transitions
    assert (1, 2) in model.transitions_
    assert (1, 3) in model.transitions_
    assert (2, 3) in model.transitions_
    assert len(model.transitions_) == 3

def test_progressive_model():
    """Test progressive model"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    patient_ids = np.arange(100)
    start_times = np.zeros(100)
    end_times = np.random.exponential(scale=1.0, size=100)
    
    # For progressive model, ensure each state transitions to the next state
    start_states = np.random.choice([1, 2, 3], size=100)  # States start from 1
    end_states = start_states + 1  # Each state transitions to next state
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(model_type='general', state_structure='progressive', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    probabilities = model.predict_state_occupation(X, times_pred)
    
    # Check states
    assert set(probabilities.keys()) == {1, 2, 3, 4}
    
    # Check transitions
    assert (1, 2) in model.transitions_
    assert (2, 3) in model.transitions_
    assert (3, 4) in model.transitions_
    assert len(model.transitions_) == 3

def test_transition_probability():
    """Test transition probability prediction"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    patient_ids = np.arange(100)
    start_times = np.zeros(100)
    end_times = np.random.exponential(scale=1.0, size=100)
    start_states = np.random.choice([1, 2, 3], size=100)  # States start from 1
    
    # Ensure end states are different from start states
    end_states = np.array([
        np.random.choice([s for s in [1, 2, 3] if s != start_state])
        for start_state in start_states
    ])
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(model_type='general', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    for from_state, to_state in model.transitions_:
        prob = model.predict_transition_probability(X, times_pred, from_state, to_state)
        
        # Check output format
        assert isinstance(prob, np.ndarray)
        assert prob.shape == (100, 10)
        
        # Check probabilities
        assert np.all(prob >= 0)
        assert np.all(prob <= 1)

def test_length_of_stay():
    """Test length of stay prediction"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    patient_ids = np.arange(100)
    start_times = np.zeros(100)
    end_times = np.random.exponential(scale=1.0, size=100)
    start_states = np.random.choice([1, 2, 3], size=100)  # States start from 1
    
    # Ensure end states are different from start states
    end_states = np.array([
        np.random.choice([s for s in [1, 2, 3] if s != start_state])
        for start_state in start_states
    ])
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(model_type='general', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    for state in model.states_:
        length_of_stay = model.predict_length_of_stay(X, state)
        
        # Check output format
        assert isinstance(length_of_stay, np.ndarray)
        assert length_of_stay.shape == (100,)
        
        # Check values
        assert np.all(length_of_stay >= 0)

def test_time_dependent_covariates():
    """Test support for time-dependent covariates"""
    # Generate synthetic data with time-dependent features
    n_samples = 100
    n_features = 5
    n_time_points = 10
    
    X = np.random.randn(n_samples, n_features, n_time_points)
    patient_ids = np.arange(n_samples)
    start_times = np.zeros(n_samples)
    end_times = np.random.exponential(scale=1.0, size=n_samples)
    start_states = np.random.choice([1, 2, 3], size=n_samples)  # States start from 1
    
    # Ensure end states are different from start states
    end_states = np.array([
        np.random.choice([s for s in [1, 2, 3] if s != start_state])
        for start_state in start_states
    ])
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(support_time_dependent=True, max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    probabilities = model.predict_state_occupation(X, times_pred)
    
    # Check output format
    assert isinstance(probabilities, dict)
    assert all(isinstance(probabilities[state], np.ndarray) for state in probabilities)
    assert all(probabilities[state].shape == (100, 10) for state in probabilities)

def test_cumulative_incidence():
    """Test cumulative incidence function prediction"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    patient_ids = np.arange(100)
    start_times = np.zeros(100)
    end_times = np.random.exponential(scale=1.0, size=100)
    start_states = np.random.choice([1, 2, 3], size=100)  # States start from 1
    
    # Ensure end states are different from start states
    end_states = np.array([
        np.random.choice([s for s in [1, 2, 3] if s != start_state])
        for start_state in start_states
    ])
    
    # Create multi-state data
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(model_type='general', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    
    # Test CIF for each absorbing state
    for absorbing_state in [3, 4]:  # Assuming these are absorbing states
        cif = model.predict_cumulative_incidence(X, times_pred, absorbing_state)
        
        # Check output format
        assert isinstance(cif, np.ndarray)
        assert cif.shape == (100, 10)
        
        # Check probabilities
        assert np.all(cif >= 0)
        assert np.all(cif <= 1)
        
        # Check monotonicity (CIF should be non-decreasing)
        for i in range(cif.shape[0]):
            assert np.all(np.diff(cif[i]) >= 0), \
                f"CIF for sample {i} is not non-decreasing" 