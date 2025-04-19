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
        self.rules_ = ["feature_0 <= 0.5", "feature_1 > 0.3"]
        self.rule_weights_ = np.array([0.8, -0.5])
        self.state_structure = StateStructure(
            states=["Healthy", "Mild", "Severe"],
            transitions=[(1, 2), (2, 3)]  # States start from 1
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
    
    def predict_cumulative_incidence(self, X, times, event_types):
        n_samples = X.shape[0]
        n_times = len(times)
        return {
            event_type: np.random.rand(n_samples, n_times)
            for event_type in event_types
        }
    
    def predict_state_occupation(self, X, times):
        n_samples = X.shape[0]
        n_times = len(times)
        return {
            i+1: np.random.rand(n_samples, n_times)
            for i in range(len(self.state_structure.states))
        }
    
    def _evaluate_rules(self, X):
        return np.random.rand(X.shape[0], len(self.rules_))

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
        event_types=[1, 2],
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
    mock_model.rules_ = []
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
    del state_probs[1]  # Remove one state
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
    mock_model.rules_ = ["Custom Rule 1", "Custom Rule 2", "Custom Rule 3"]
    mock_model.rule_weights_ = np.array([0.8, -0.5, 0.3])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_custom_labels(mock_model, sample_data):
    """Test plotting cumulative incidence with custom state labels"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=["Healthy", "Sick", "Recovered"],
        transitions=[(1, 2), (2, 3)]
    )
    times = np.linspace(0, 5, 10)
    plot_cumulative_incidence(
        mock_model,
        sample_data,
        event_types=[1, 2],
        times=times
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_rule_importance_no_rules(mock_model):
    """Test plotting rule importance with no rules"""
    plt.close('all')
    mock_model.rules_ = []
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
    mock_model.rules_ = ["Rule 1", "Rule 2"]
    mock_model.rule_weights_ = {
        (1, 2): np.array([0.5, -0.3]),
        (2, 3): np.array([0.2, 0.4])
    }
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_invalid_event_type(mock_model, sample_data):
    """Test plotting cumulative incidence with invalid event type"""
    plt.close('all')
    def mock_predict_cumulative_incidence(X, times, event_types):
        if any(event_type > 2 for event_type in event_types):
            raise KeyError("Invalid event type")
        return {1: np.zeros((X.shape[0], len(times)))}
    mock_model.predict_cumulative_incidence = mock_predict_cumulative_incidence
    with pytest.raises(KeyError):
        plot_cumulative_incidence(
            mock_model,
            sample_data,
            event_types=[999],  # Invalid event type
            times=np.linspace(0, 5, 10)
        )
    plt.close('all')

def test_plot_state_transitions_no_rule_weights(mock_model, sample_data):
    """Test plotting state transitions with missing rule weights"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=["A", "B", "C"],
        transitions=[(1, 2), (2, 3)]
    )
    mock_model.rules_ = ["Rule 1", "Rule 2"]
    mock_model.rule_weights_ = {(1, 2): np.array([0.5, 0.3])}  # Missing (2, 3)
    mock_model.baseline_hazards_ = {(1, 2): (np.array([0, 1]), np.array([0.1, 0.2]))}
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')