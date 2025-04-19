"""Tests for utility functions and classes"""

import numpy as np
import pytest
from ruletimer.utils import StateStructure

@pytest.fixture
def simple_structure():
    """Create a simple state structure for testing"""
    states = ["Healthy", "Mild", "Severe"]
    transitions = [(0, 1), (1, 2)]
    return StateStructure(states, transitions)

@pytest.fixture
def complex_structure():
    """Create a more complex state structure for testing"""
    states = ["Healthy", "Mild", "Moderate", "Severe"]
    transitions = [(0, 1), (1, 2), (2, 3), (1, 3), (0, 2)]
    return StateStructure(states, transitions, initial_state=0)

def test_state_structure_initialization():
    """Test initialization of StateStructure"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.states == states
    assert state_structure.transitions == transitions
    assert state_structure.initial_state == 0

def test_state_structure_initialization_custom_initial():
    """Test initialization of StateStructure with custom initial state"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    initial_state = 1
    state_structure = StateStructure(states, transitions, initial_state)
    
    assert state_structure.initial_state == initial_state

def test_transitions_from_state():
    """Test getting transitions from a state"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.transitions_from_state(0) == {1}
    assert state_structure.transitions_from_state(1) == {2}
    assert state_structure.transitions_from_state(2) == set()

def test_transition_idx():
    """Test getting transition index"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.transition_idx(0, 1) == (0, 1)
    assert state_structure.transition_idx(1, 2) == (1, 2)
    assert state_structure.transition_idx(0, 2) is None

def test_get_next_states():
    """Test getting next states"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.get_next_states(0) == {1}
    assert state_structure.get_next_states(1) == {2}
    assert state_structure.get_next_states(2) == set()

def test_get_previous_states():
    """Test getting previous states"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.get_previous_states(0) == set()
    assert state_structure.get_previous_states(1) == {0}
    assert state_structure.get_previous_states(2) == {1}

def test_is_valid_transition():
    """Test checking if a transition is valid"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.is_valid_transition(0, 1) is True
    assert state_structure.is_valid_transition(1, 2) is True
    assert state_structure.is_valid_transition(0, 2) is False
    assert state_structure.is_valid_transition(2, 0) is False

def test_get_state_name():
    """Test getting state name"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.get_state_name(0) == 'State 0'
    assert state_structure.get_state_name(1) == 'State 1'
    assert state_structure.get_state_name(2) == 'State 2'

def test_get_state_index():
    """Test getting state index"""
    states = ['State 0', 'State 1', 'State 2']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.get_state_index('State 0') == 0
    assert state_structure.get_state_index('State 1') == 1
    assert state_structure.get_state_index('State 2') == 2

def test_invalid_transitions():
    """Test handling invalid transitions"""
    states = ['State 0', 'State 1', 'State 2']
    
    # Test invalid state indices
    with pytest.raises(ValueError):
        StateStructure(states, [(-1, 0)])
    
    with pytest.raises(ValueError):
        StateStructure(states, [(0, 3)])
    
    # Test invalid transition format
    with pytest.raises(ValueError):
        StateStructure(states, [(0, 1, 2)])  # Too many elements
    
    with pytest.raises(ValueError):
        StateStructure(states, [(0,)])  # Too few elements

def test_complex_transitions():
    """Test complex transition structure"""
    states = ['State 0', 'State 1', 'State 2', 'State 3']
    transitions = [(0, 1), (0, 2), (1, 3), (2, 3)]
    state_structure = StateStructure(states, transitions)
    
    assert state_structure.transitions_from_state(0) == {1, 2}
    assert state_structure.transitions_from_state(1) == {3}
    assert state_structure.transitions_from_state(2) == {3}
    assert state_structure.transitions_from_state(3) == set()
    
    assert state_structure.get_previous_states(3) == {1, 2}
    assert state_structure.get_previous_states(1) == {0}
    assert state_structure.get_previous_states(2) == {0}
    assert state_structure.get_previous_states(0) == set()

def test_state_structure_initialization(simple_structure):
    """Test initialization of StateStructure"""
    assert len(simple_structure.states) == 3
    assert len(simple_structure.transitions) == 2
    assert simple_structure.initial_state == 0

def test_transitions_from_state(simple_structure):
    """Test getting transitions from a state"""
    assert simple_structure.transitions_from_state(0) == {1}
    assert simple_structure.transitions_from_state(1) == {2}
    assert simple_structure.transitions_from_state(2) == set()
    assert simple_structure.transitions_from_state(3) == set()  # Non-existent state

def test_transition_idx(simple_structure):
    """Test getting transition indices"""
    assert simple_structure.transition_idx(0, 1) == (0, 1)
    assert simple_structure.transition_idx(1, 2) == (1, 2)
    assert simple_structure.transition_idx(0, 2) is None  # Invalid transition
    assert simple_structure.transition_idx(2, 0) is None  # Invalid transition

def test_get_next_states(complex_structure):
    """Test getting next possible states"""
    assert complex_structure.get_next_states(0) == {1, 2}
    assert complex_structure.get_next_states(1) == {2, 3}
    assert complex_structure.get_next_states(2) == {3}
    assert complex_structure.get_next_states(3) == set()

def test_get_previous_states(complex_structure):
    """Test getting previous possible states"""
    assert complex_structure.get_previous_states(0) == set()
    assert complex_structure.get_previous_states(1) == {0}
    assert complex_structure.get_previous_states(2) == {0, 1}
    assert complex_structure.get_previous_states(3) == {1, 2}

def test_is_valid_transition(complex_structure):
    """Test transition validity checking"""
    assert complex_structure.is_valid_transition(0, 1) is True
    assert complex_structure.is_valid_transition(1, 3) is True
    assert complex_structure.is_valid_transition(0, 3) is False
    assert complex_structure.is_valid_transition(3, 0) is False

def test_get_state_name(simple_structure):
    """Test getting state names"""
    assert simple_structure.get_state_name(0) == "Healthy"
    assert simple_structure.get_state_name(1) == "Mild"
    assert simple_structure.get_state_name(2) == "Severe"
    with pytest.raises(IndexError):
        simple_structure.get_state_name(3)

def test_get_state_index(simple_structure):
    """Test getting state indices"""
    assert simple_structure.get_state_index("Healthy") == 0
    assert simple_structure.get_state_index("Mild") == 1
    assert simple_structure.get_state_index("Severe") == 2
    with pytest.raises(ValueError):
        simple_structure.get_state_index("Unknown")

def test_complex_transitions(complex_structure):
    """Test complex transition patterns"""
    # Test multiple paths to same state
    assert len(complex_structure.get_previous_states(3)) == 2
    
    # Test direct and indirect transitions
    assert 3 in complex_structure.get_next_states(1)  # Direct
    assert 3 in complex_structure.transitions_from_state(1)  # Direct
    
    # Test initial state connections
    assert 0 not in complex_structure.get_previous_states(0)
    assert len(complex_structure.get_next_states(0)) == 2

def test_invalid_transitions(simple_structure):
    """Test handling of invalid transitions"""
    # Test non-existent states
    assert simple_structure.get_next_states(99) == set()
    assert simple_structure.get_previous_states(99) == set()
    assert simple_structure.transition_idx(99, 0) is None
    assert simple_structure.transition_idx(0, 99) is None
    
    # Test reverse transitions
    assert simple_structure.transition_idx(2, 1) is None
    assert simple_structure.transition_idx(1, 0) is None 