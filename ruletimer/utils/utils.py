"""
Utility classes and functions
"""

class StateStructure:
    """Class for managing state structure in multi-state models"""
    
    def __init__(self, states, transitions):
        """
        Initialize state structure
        
        Parameters
        ----------
        states : list
            List of state names
        transitions : list of tuples
            List of valid transitions as (from_state_idx, to_state_idx) pairs
        """
        self.states = states
        self.transitions = transitions
        self._state_to_idx = {state: idx for idx, state in enumerate(states)}
        self._idx_to_state = {idx: state for idx, state in enumerate(states)}
    
    def get_state_index(self, state):
        """Get index of a state"""
        return self._state_to_idx[state]
    
    def get_state_name(self, idx):
        """Get name of a state"""
        return self._idx_to_state[idx]
    
    def is_valid_transition(self, from_state, to_state):
        """Check if a transition is valid"""
        return (from_state, to_state) in self.transitions
    
    def get_next_states(self, state):
        """Get all possible next states from a given state"""
        return [to_state for from_state, to_state in self.transitions 
                if from_state == state]
    
    def get_previous_states(self, state):
        """Get all possible previous states to a given state"""
        return [from_state for from_state, to_state in self.transitions 
                if to_state == state]
    
    def transitions_from_state(self, state):
        """Get all transitions from a given state"""
        return [(from_state, to_state) for from_state, to_state in self.transitions 
                if from_state == state]
    
    def transitions_to_state(self, state):
        """Get all transitions to a given state"""
        return [(from_state, to_state) for from_state, to_state in self.transitions 
                if to_state == state] 