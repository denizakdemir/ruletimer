import numpy as np
import pytest
from ruletimer import Survival, CompetingRisks, MultiState, RuleSurvival, RuleCompetingRisks, RuleMultiState
from ruletimer.utils.utils import StateStructure

def test_survival_curve_monotonicity():
    """Test that survival curves are non-increasing"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    y = Survival(time, event)
    
    # Fit model
    model = RuleSurvival(max_rules=10, alpha=0.1)
    model.fit(X, y)
    
    # Generate test data
    X_test = np.random.randn(10, n_features)
    times = np.linspace(0, 2, 100)  # More time points for better testing
    
    # Predict survival probabilities
    survival_probs = model.predict_survival(X_test, times)
    
    # Check monotonicity for each sample
    for i in range(survival_probs.shape[0]):
        # Survival should be non-increasing
        assert np.all(np.diff(survival_probs[i]) <= 0), \
            f"Survival curve for sample {i} is not non-increasing"
        
        # Survival should be between 0 and 1
        assert np.all(survival_probs[i] >= 0) and np.all(survival_probs[i] <= 1), \
            f"Survival probabilities for sample {i} are not in [0,1] range"

def test_cif_monotonicity():
    """Test that cumulative incidence functions are non-decreasing"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.4, 0.4])
    
    y = CompetingRisks(time, event)
    
    # Fit model
    model = RuleCompetingRisks(max_rules=10, alpha=0.1)
    model.fit(X, y)
    
    # Generate test data
    X_test = np.random.randn(10, n_features)
    times = np.linspace(0, 2, 100)  # More time points for better testing
    event_types = [1, 2]
    
    # Predict cumulative incidence functions
    cif = model.predict_cumulative_incidence(X_test, times, event_types)
    
    # Check monotonicity for each event type and sample
    for event_type in event_types:
        for i in range(cif[event_type].shape[0]):
            # CIF should be non-decreasing
            assert np.all(np.diff(cif[event_type][i]) >= 0), \
                f"CIF for event {event_type}, sample {i} is not non-decreasing"
            
            # CIF should be between 0 and 1
            assert np.all(cif[event_type][i] >= 0) and np.all(cif[event_type][i] <= 1), \
                f"CIF probabilities for event {event_type}, sample {i} are not in [0,1] range"

def test_state_occupation_monotonicity():
    """Test that state occupation probabilities are non-increasing for absorbing states"""
    # Define state structure
    states = ["Healthy", "Mild", "Moderate", "Severe", "Death"]
    transitions = [(1,2), (2,3), (3,4), (2,4), (3,5), (4,5)]  # Using 1-based indexing consistently
    structure = StateStructure(states, transitions)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    
    # Generate multi-state data (using 1-based indexing for MultiState class)
    patient_ids = np.arange(n_samples)
    start_times = np.zeros(n_samples)
    end_times = np.random.exponential(scale=1, size=n_samples)
    start_states = np.ones(n_samples)  # Start in state 1 (Healthy)
    end_states = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])  # Using 1-based indexing
    
    y = MultiState(patient_ids, start_times, end_times, start_states, end_states)
    
    # Fit model
    model = RuleMultiState(structure, max_rules=10, alpha=0.1)
    model.fit(X, y)
    
    # Generate test data
    X_test = np.random.randn(10, n_features)
    times = np.linspace(0, 2, 100)  # More time points for better testing
    
    # Predict state occupation probabilities
    state_probs = model.predict_state_occupation(X_test, times)
    
    # Check monotonicity for absorbing state (Death)
    death_state = 5  # Last state (Death) in 1-based indexing
    for i in range(state_probs[death_state].shape[0]):
        # Probability of being in death state should be non-decreasing
        assert np.all(np.diff(state_probs[death_state][i]) >= 0), \
            f"Death state probability for sample {i} is not non-decreasing"
        
        # Probabilities should be between 0 and 1
        assert np.all(state_probs[death_state][i] >= 0) and np.all(state_probs[death_state][i] <= 1), \
            f"Death state probabilities for sample {i} are not in [0,1] range" 