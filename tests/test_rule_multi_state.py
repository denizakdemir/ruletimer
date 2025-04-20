"""
Tests for the RuleMultiState class.
"""
import numpy as np
import pytest
from ruletimer.models.rule_multi_state import RuleMultiState
from ruletimer.utils import StateStructure

def test_rule_multi_state_initialization():
    """Test initialization of RuleMultiState with different parameters."""
    # Test with default parameters
    model = RuleMultiState()
    assert model.max_rules == 100
    assert model.alpha == 0.1
    assert model.max_depth == 3
    assert model.min_samples_leaf == 10
    assert model.n_estimators == 100
    assert model.tree_type == 'classification'
    assert model.tree_growing_strategy == 'forest'
    assert model.prune_rules is True
    assert model.l1_ratio == 0.5
    
    # Test with custom parameters
    state_structure = StateStructure(
        states=['healthy', 'sick', 'dead'],
        transitions=[(0, 1), (1, 2)]
    )
    model = RuleMultiState(
        max_rules=50,
        alpha=0.5,
        state_structure=state_structure,
        max_depth=5,
        min_samples_leaf=20,
        n_estimators=200,
        tree_type='regression',
        tree_growing_strategy='single',
        prune_rules=False,
        l1_ratio=0.8,
        random_state=42
    )
    assert model.max_rules == 50
    assert model.alpha == 0.5
    assert model.state_structure == state_structure
    assert model.max_depth == 5
    assert model.min_samples_leaf == 20
    assert model.n_estimators == 200
    assert model.tree_type == 'regression'
    assert model.tree_growing_strategy == 'single'
    assert model.prune_rules is False
    assert model.l1_ratio == 0.8
    assert model.random_state == 42

def test_rule_generation():
    """Test rule generation functionality."""
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    model = RuleMultiState(
        max_rules=10,
        max_depth=2,
        n_estimators=5,
        random_state=42
    )
    
    rules = model._generate_rules(X, y)
    assert isinstance(rules, list)
    assert len(rules) <= 10  # Should not exceed max_rules
    assert all(isinstance(rule, str) for rule in rules)

def test_rule_evaluation():
    """Test rule evaluation functionality."""
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    rules = ['feature_0 > 2', 'feature_1 < 5']
    
    model = RuleMultiState()
    rule_values = model._evaluate_rules(X, rules)
    
    assert isinstance(rule_values, np.ndarray)
    assert rule_values.shape == (X.shape[0], len(rules))
    assert np.all(np.logical_or(rule_values == 0, rule_values == 1))

def test_fit_predict():
    """Test model fitting and prediction."""
    # Create a simple multi-state dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    states = ['healthy', 'sick', 'dead']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)
    
    # Create transition data
    transition_data = {
        (0, 1): {
            'times': np.array([1.0, 2.0, 3.0]),
            'events': np.array([1, 1, 0])
        },
        (1, 2): {
            'times': np.array([2.0, 3.0, 4.0]),
            'events': np.array([1, 0, 1])
        }
    }
    
    model = RuleMultiState(
        state_structure=state_structure,
        max_rules=5,
        random_state=42
    )
    
    # Test fitting
    model.fit(X, transition_data)
    assert model.is_fitted_
    assert len(model.rules_) == len(transitions)
    assert len(model.rule_importances_) == len(transitions)
    assert len(model.rule_coefficients_) == len(transitions)
    
    # Test prediction
    times = np.array([1.0, 2.0, 3.0])
    target_state = 'dead'
    predictions = model.predict_cumulative_incidence(X, times, target_state)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0], len(times))
    assert np.all(predictions >= 0) and np.all(predictions <= 1)

def test_feature_importances():
    """Test feature importance calculation."""
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    states = ['healthy', 'sick', 'dead']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)
    
    transition_data = {
        (0, 1): {
            'times': np.array([1.0, 2.0, 3.0]),
            'events': np.array([1, 1, 0])
        },
        (1, 2): {
            'times': np.array([2.0, 3.0, 4.0]),
            'events': np.array([1, 0, 1])
        }
    }
    
    model = RuleMultiState(
        state_structure=state_structure,
        max_rules=5,
        random_state=42
    )
    model.fit(X, transition_data)
    
    # Test feature importances for each transition
    for transition in transitions:
        importances = model.get_feature_importances(transition)
        assert isinstance(importances, np.ndarray)
        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)
        assert np.allclose(np.sum(importances), 1.0)

def test_error_handling():
    """Test error handling for invalid inputs."""
    model = RuleMultiState()
    
    # Test invalid state structure
    with pytest.raises(ValueError):
        invalid_states = ['state1', 'state2']
        invalid_transitions = [(0, 1), (2, 3)]  # Invalid state indices
        invalid_structure = StateStructure(
            states=invalid_states,
            transitions=invalid_transitions
        )
        RuleMultiState(state_structure=invalid_structure)
    
    # Test invalid tree type
    with pytest.raises(ValueError):
        RuleMultiState(tree_type='invalid_type')
    
    # Test invalid tree growing strategy
    with pytest.raises(ValueError):
        RuleMultiState(tree_growing_strategy='invalid_strategy')
    
    # Test invalid alpha
    with pytest.raises(ValueError):
        RuleMultiState(alpha=-0.1)
    
    # Test invalid l1_ratio
    with pytest.raises(ValueError):
        RuleMultiState(l1_ratio=1.5)

def test_empty_and_single_row():
    """Test model with empty and single-row datasets."""
    X_empty = np.empty((0, 2))
    X_single = np.array([[1, 2]])
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    # Should not raise error on fit with empty data
    with pytest.raises(Exception):
        model.fit(X_empty, {(0, 1): {'times': np.array([]), 'events': np.array([])}})
    # Single row
    model.fit(X_single, {(0, 1): {'times': np.array([1.0]), 'events': np.array([1])}})
    preds = model.predict_cumulative_incidence(X_single, np.array([1.0]), 1)
    assert preds.shape == (1, 1)
    assert np.all(preds >= 0) and np.all(preds <= 1)

def test_all_censored():
    """Test model with all transitions censored."""
    X = np.random.randn(5, 2)
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    # All events are censored (0)
    model.fit(X, {(0, 1): {'times': np.ones(5), 'events': np.zeros(5)}})
    preds = model.predict_cumulative_incidence(X, np.array([1.0, 2.0]), 1)
    assert np.all(preds >= 0) and np.all(preds <= 1)
    assert np.allclose(preds, 0)

def test_missing_transition():
    """Test model with missing transitions for some states."""
    X = np.random.randn(4, 2)
    states = ['a', 'b', 'c']
    transitions = [(0, 1)]  # No (1,2) or (0,2)
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, {(0, 1): {'times': np.ones(4), 'events': np.array([1, 0, 1, 0])}})
    # Should not raise error for missing transitions
    preds = model.predict_cumulative_incidence(X, np.array([1.0]), 1)
    assert preds.shape == (4, 1)

def test_large_state_space():
    """Test scalability with many states and transitions."""
    n_states = 10
    states = [str(i) for i in range(n_states)]
    transitions = [(i, i+1) for i in range(n_states-1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    X = np.random.randn(20, 3)
    transition_data = {t: {'times': np.random.rand(20), 'events': np.random.randint(0, 2, 20)} for t in transitions}
    model = RuleMultiState(state_structure=state_structure, max_rules=3)
    model.fit(X, transition_data)
    for t in transitions:
        assert len(model.rules_[t]) <= 3

def test_occupation_probability_sum():
    """Test that state occupation probabilities sum to 1 at each time point."""
    X = np.random.randn(10, 2)
    states = ['a', 'b', 'c']
    transitions = [(0, 1), (1, 2), (0, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)
    transition_data = {t: {'times': np.random.rand(10), 'events': np.random.randint(0, 2, 10)} for t in transitions}
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, transition_data)
    times = np.linspace(0, 5, 10)
    occ = model.predict_state_occupation(X, times, initial_state=0)
    total = np.zeros((X.shape[0], len(times)))
    for s in occ:
        total += occ[s]
    assert np.allclose(total, 1, atol=1e-5)

def test_absorbing_state_monotonicity():
    """Test that occupation probability for absorbing state is non-decreasing."""
    X = np.random.randn(5, 2)
    states = ['a', 'b', 'c']
    transitions = [(0, 1), (1, 2), (0, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)
    transition_data = {t: {'times': np.random.rand(5), 'events': np.random.randint(0, 2, 5)} for t in transitions}
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, transition_data)
    times = np.linspace(0, 5, 10)
    occ = model.predict_state_occupation(X, times, initial_state=0)
    # State 2 is absorbing
    diffs = np.diff(occ[2], axis=1)
    assert np.all(diffs >= -1e-10)

def test_feature_importance_normalization():
    """Test that feature importances are non-negative and sum to 1."""
    X = np.random.randn(6, 3)
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, {(0, 1): {'times': np.random.rand(6), 'events': np.random.randint(0, 2, 6)}})
    importances = model.get_feature_importances((0, 1))
    assert np.all(importances >= 0)
    assert np.isclose(np.sum(importances), 1)

def test_model_persistence(tmp_path):
    """Test that model predictions are consistent after saving and loading."""
    import joblib
    X = np.random.randn(8, 2)
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, {(0, 1): {'times': np.random.rand(8), 'events': np.random.randint(0, 2, 8)}})
    times = np.array([1.0, 2.0])
    preds = model.predict_cumulative_incidence(X, times, 1)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    preds2 = loaded.predict_cumulative_incidence(X, times, 1)
    assert np.allclose(preds, preds2)