"""Tests for visualization functions"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from ruletimer.visualization.visualization import (
    plot_rule_importance,
    plot_cumulative_incidence,
    plot_state_transitions,
    plot_state_occupation
)
from ruletimer.models.base import BaseRuleEnsemble
from ruletimer.utils.utils import StateStructure

class MockRuleEnsemble(BaseRuleEnsemble):
    """Mock rule ensemble for testing visualization functions"""
    
    def __init__(self):
        super().__init__()
        self._rules_tuples = ["feature_0 <= 0.5", "feature_1 > 0.3"]
        self.rule_weights_ = np.array([0.8, -0.5])
        self.state_structure = StateStructure(
            states=[1, 2, 3],  # Using 1-based indexing for states (0 is reserved for censoring)
            transitions=[(1, 2), (2, 3)],  # States are 1-based
            state_names=["Healthy", "Mild", "Severe"]  # Providing state names separately
        )
        self._y = np.rec.fromarrays([np.array([1.0, 2.0, 3.0])], names=['time'])
        self.baseline_hazards_ = {
            (1, 2): (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3])),
            (2, 3): (np.array([0, 1, 2]), np.array([0.2, 0.3, 0.4]))
        }
    
    def _fit_weights(self, rule_values, y):
        pass
    
    def _compute_feature_importances(self):
        pass

    @property
    def rules_(self):
        # Return a dict to match the expected interface in plot_rule_importance
        # Use a single transition key for simplicity
        return {(1, 2): list(self._rules_tuples)}

    def get_rule_importances(self):
        # Return a dict with the same keys as rules_
        if isinstance(self.rule_weights_, dict):
            return self.rule_weights_
        return {(1, 2): np.array(self.rule_weights_)}

    def predict_cumulative_incidence(self, X, times, event_types=None, event_type=None, **kwargs):
        n_samples = X.shape[0]
        n_times = len(times)
        # Always return a dict of 2D numpy arrays for each requested event_type
        if event_types is not None:
            return {et: np.random.rand(n_samples, n_times) for et in event_types}
        elif event_type is not None:
            # Accept event_type as int or str
            if isinstance(event_type, str) and event_type.startswith("Event"):
                try:
                    et = int(event_type.replace("Event", ""))
                except Exception:
                    et = event_type
            else:
                et = event_type
            return {et: np.random.rand(n_samples, n_times)}
        else:
            # Default: return a dict with a single key 0
            return {0: np.random.rand(n_samples, n_times)}

    def predict_state_occupation(self, X, times, initial_state=None, **kwargs):
        n_samples = X.shape[0]
        n_times = len(times)
        # Return a dict for each state (0-based, to match visualization expectations)
        return {i: np.random.rand(n_samples, n_times) for i in range(len(self.state_structure.states))}
    
    def _evaluate_rules(self, X):
        return np.random.rand(X.shape[0], len(self.rules_))

    def predict_transition_hazard(self, X, times, from_state, to_state):
        n_samples = X.shape[0]
        n_times = len(times)
        # Return a dummy hazard array
        return np.random.rand(n_samples, n_times)

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    return MockRuleEnsemble()

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return X

def test_plot_rule_importance(mock_model, sample_data):
    """Test rule importance plotting"""
    plt.close('all')
    plot_rule_importance(mock_model, top_n=2)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence(mock_model, sample_data):
    """Test cumulative incidence plotting"""
    plt.close('all')
    plot_cumulative_incidence(
        mock_model,
        sample_data,
        event_types=[0, 1],
        times=np.linspace(0, 5, 10)
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_transitions(mock_model, sample_data):
    """Test state transition plotting"""
    plt.close('all')
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_occupation(mock_model, sample_data):
    """Test state occupation plotting"""
    plt.close('all')
    times = np.linspace(0, 5, 10)
    state_probs = mock_model.predict_state_occupation(sample_data, times)
    plot_state_occupation(
        times,
        state_probs,
        state_names=mock_model.state_structure.states
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_rule_importance_empty(mock_model):
    """Test rule importance plotting with empty rules"""
    plt.close('all')
    mock_model._rules_tuples = []
    mock_model.rule_weights_ = np.array([])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_no_times(mock_model, sample_data):
    """Test plotting cumulative incidence with empty times array"""
    plt.close('all')
    with pytest.raises(ValueError):
        plot_cumulative_incidence(
            mock_model,
            sample_data,
            event_types=[1],
            times=np.array([])
        )
    plt.close('all')

def test_plot_state_occupation_missing_state(mock_model, sample_data):
    """Test state occupation plotting with missing state"""
    plt.close('all')
    times = np.linspace(0, 5, 10)
    state_probs = mock_model.predict_state_occupation(sample_data, times)
    del state_probs[0]  # Remove one state (changed to 0-based)
    plot_state_occupation(
        times,
        state_probs,
        state_names=mock_model.state_structure.states
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_transitions_invalid_times(mock_model, sample_data):
    """Test plotting state transitions with invalid times"""
    plt.close('all')
    with pytest.raises(ValueError):
        plot_state_transitions(
            mock_model,
            sample_data,
            time=-1.0  # Negative time should raise error
        )
    plt.close('all')

def test_plot_state_occupation_single_state(mock_model, sample_data):
    """Test plotting state occupation with single state"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=["Single"],
        transitions=[]
    )
    times = np.linspace(0, 5, 10)
    with pytest.raises(ValueError):
        plot_state_occupation(
            times,
            {1: np.random.rand(sample_data.shape[0], len(times))},
            state_names=["Single"]
        )
    plt.close('all')

def test_plot_rule_importance_custom_names(mock_model):
    """Test plotting rule importance with custom rule names"""
    plt.close('all')
    mock_model._rules_tuples = ["Custom Rule 1", "Custom Rule 2", "Custom Rule 3"]
    mock_model.rule_weights_ = np.array([0.8, -0.5, 0.3])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_custom_labels(mock_model, sample_data):
    """Test plotting cumulative incidence with custom state labels"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=[1, 2, 3],  # Changed to 1-based indexing
        transitions=[(1, 2), (2, 3)]  # Changed to 1-based indexing
    )
    times = np.linspace(0, 5, 10)
    plot_cumulative_incidence(
        mock_model,
        sample_data,
        event_types=[0, 1],  # Changed to 0-based indexing
        times=times
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_rule_importance_no_rules(mock_model):
    """Test plotting rule importance with no rules"""
    plt.close('all')
    mock_model._rules_tuples = []
    mock_model.rule_weights_ = np.array([])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_no_event_types(mock_model, sample_data):
    """Test plotting cumulative incidence with no event types"""
    plt.close('all')
    with pytest.raises(ValueError):
        plot_cumulative_incidence(
            mock_model,
            sample_data,
            event_types=[],
            times=np.linspace(0, 5, 10)
        )
    plt.close('all')

def test_plot_state_transitions_no_transitions(mock_model, sample_data):
    """Test plotting state transitions with no transitions"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=["A", "B"],
        transitions=[]
    )
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_occupation_empty_probs(mock_model, sample_data):
    """Test plotting state occupation with empty probabilities"""
    plt.close('all')
    times = np.linspace(0, 5, 10)
    with pytest.raises(ValueError):
        plot_state_occupation(
            times,
            {},
            state_names=["A", "B"]
        )
    plt.close('all')

def test_plot_rule_importance_dict_weights(mock_model):
    """Test plotting rule importance with dictionary weights"""
    plt.close('all')
    mock_model._rules_tuples = ["Rule 1", "Rule 2"]
    mock_model.rule_weights_ = {
        (1, 2): np.array([0.5, -0.3]),  # Changed to 1-based indexing
        (2, 3): np.array([0.2, 0.4])    # Changed to 1-based indexing
    }
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_invalid_event_type(mock_model, sample_data):
    """Test plotting cumulative incidence with invalid event type"""
    plt.close('all')
    # Modify mock signature to accept event_type keyword argument and potentially others (**kwargs)
    def mock_predict_cumulative_incidence(X, times, event_type=None, **kwargs):
        # Extract integer event type from the string "Event{event_type}"
        current_event_type_int = -1 # Default invalid
        if event_type is not None and isinstance(event_type, str) and event_type.startswith("Event"):
            try:
                current_event_type_int = int(event_type.replace("Event", ""))
            except ValueError:
                pass # Keep -1 if conversion fails
        elif isinstance(event_type, int):
             # Handle if event_type is passed as int (though plot function uses string)
            current_event_type_int = event_type

        # The test expects a KeyError for invalid types (e.g., > 3 based on original mock logic)
        # Let's simulate this check based on the extracted integer type
        if current_event_type_int > 3: # Check the integer value
            raise KeyError(f"Invalid event type: {current_event_type_int}")

        # Return a valid structure for valid types, using the integer type as key
        # This part might not even be reached if the type is invalid and raises KeyError
        return {current_event_type_int: np.zeros((X.shape[0], len(times)))}

    # Store original method before mocking
    original_predict_method = mock_model.predict_cumulative_incidence
    mock_model.predict_cumulative_incidence = mock_predict_cumulative_incidence

    try:
        # Expect KeyError because event_type 999 > 3
        with pytest.raises(KeyError):
            plot_cumulative_incidence(
                mock_model,
                sample_data,
                event_types=[999],  # Invalid event type
                times=np.linspace(0, 5, 10)
            )
    finally:
        # Restore original method
        mock_model.predict_cumulative_incidence = original_predict_method
        plt.close('all')

def test_plot_state_transitions_no_rule_weights(mock_model, sample_data):
    """Test plotting state transitions with missing rule weights"""
    plt.close('all')
    # Align state structure with the mock's hardcoded rules_ key (1, 2)
    # Use states 1, 2, 3 and transitions (1, 2), (2, 3)
    mock_model.state_structure = StateStructure(
        states=[1, 2, 3], # Use 1-based integer states
        transitions=[(1, 2), (2, 3)], # Define transitions
        state_names=["A", "B", "C"] # Optional names
    )
    # Set the underlying attributes that rules_ and get_rule_importances use
    mock_model._rules_tuples = ["Rule 1", "Rule 2"] # Define rules
    # Provide weights only for the transition (1, 2), matching the rules_ property key
    mock_model.rule_weights_ = {(1, 2): np.array([0.5, 0.3])}
    # Provide baseline hazards for the transition with rules
    mock_model.baseline_hazards_ = {(1, 2): (np.array([0, 1]), np.array([0.1, 0.2]))}

    # Mock predict_transition_hazard
    original_predict_hazard = getattr(mock_model, 'predict_transition_hazard', None)

    def mock_predict_hazard(X, times, from_state, to_state):
        n_samples = X.shape[0]
        n_times = len(times)
        # The plot function calls this only for transitions in model.rules_
        # In this setup, that's only (1, 2)
        if (from_state, to_state) == (1, 2):
            return np.random.rand(n_samples, n_times) * 0.1
        elif original_predict_hazard:
            return original_predict_hazard(X, times, from_state, to_state)
        else:
            # This path shouldn't be hit by plot_state_transitions given the check
            raise ValueError(f"Unexpected transition requested in mock: ({from_state}, {to_state})")

    mock_model.predict_transition_hazard = mock_predict_hazard

    # Plot function should now work, drawing state 1, 2, 3 and an arrow for (1, 2)
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums() # Check if figure was created
    plt.close('all')

    # Restore original method
    if original_predict_hazard:
        mock_model.predict_transition_hazard = original_predict_hazard
    elif hasattr(mock_model, 'predict_transition_hazard'):
        delattr(mock_model, 'predict_transition_hazard')