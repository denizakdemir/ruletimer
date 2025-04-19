import numpy as np
import pytest
from ruletimer.state_manager import StateManager

def test_state_manager_initialization():
    """Test initialization of state manager"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    
    manager = StateManager(states, transitions)
    assert manager.states == states
    assert manager.transitions == transitions
    assert len(manager._state_to_idx) == len(states)
    assert len(manager._idx_to_state) == len(states)

def test_to_internal_index():
    """Test conversion to internal index"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    # Test string state names
    assert manager.to_internal_index("Healthy") == 0
    assert manager.to_internal_index("Disease") == 1
    assert manager.to_internal_index("Death") == 2
    
    # Test numeric indices
    assert manager.to_internal_index(0) == 0
    assert manager.to_internal_index(1) == 1
    assert manager.to_internal_index(2) == 2
    
    # Test invalid state
    with pytest.raises(ValueError):
        manager.to_internal_index("Invalid")

def test_to_external_state():
    """Test conversion to external state"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    # Test valid indices
    assert manager.to_external_state(0) == "Healthy"
    assert manager.to_external_state(1) == "Disease"
    assert manager.to_external_state(2) == "Death"
    
    # Test invalid index
    with pytest.raises(ValueError):
        manager.to_external_state(3)

def test_validate_transition():
    """Test transition validation"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    # Test valid transitions
    assert manager.validate_transition(0, 1)
    assert manager.validate_transition(1, 2)
    assert manager.validate_transition(0, 2)
    
    # Test invalid transitions
    assert not manager.validate_transition(0, 0)  # Self-transition
    assert not manager.validate_transition(2, 1)  # Reverse transition
    assert not manager.validate_transition(0, 3)  # Invalid state

def test_get_possible_transitions():
    """Test getting possible transitions"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    # Test from specific state
    assert manager.get_possible_transitions(0) == [1, 2]
    assert manager.get_possible_transitions(1) == [2]
    assert manager.get_possible_transitions(2) == []
    
    # Test all transitions
    all_transitions = manager.get_possible_transitions()
    assert all_transitions == [(0, 1), (0, 2), (1, 2)]

def test_is_absorbing_state():
    """Test absorbing state detection"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    assert not manager.is_absorbing_state(0)
    assert not manager.is_absorbing_state(1)
    assert manager.is_absorbing_state(2)

def test_get_absorbing_states():
    """Test getting all absorbing states"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    assert manager.get_absorbing_states() == [2]

def test_validate_state_sequence():
    """Test state sequence validation"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)
    
    # Test valid sequence
    assert manager.validate_state_sequence([0, 1, 2])
    assert manager.validate_state_sequence([0, 2])
    
    # Test invalid sequences
    assert not manager.validate_state_sequence([0, 0])  # Self-transition
    assert not manager.validate_state_sequence([2, 1])  # Reverse transition
    assert not manager.validate_state_sequence([0, 3])  # Invalid state 