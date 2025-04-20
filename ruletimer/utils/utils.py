"""
Utility classes and functions
"""

class StateStructure:
    """Class for managing state structure in multi-state models"""
    
    def __init__(self, states, transitions, state_names=None):
        """
        Initialize state structure
        
        Parameters
        ----------
        states : list
            List of state indices
        transitions : list of tuples
            List of valid transitions as (from_state_idx, to_state_idx) pairs
        state_names : list, optional
            List of state names. If not provided, state indices will be used as names.
        """
        self.states = states
        # Map integer transitions to state values if needed
        if len(states) > 0 and not all(isinstance(s, int) for s in states):
            # If states are not all int, map int transitions to state values
            mapped_transitions = []
            for from_state, to_state in transitions:
                if isinstance(from_state, int) and from_state < len(states):
                    from_state_val = states[from_state]
                else:
                    from_state_val = from_state
                if isinstance(to_state, int) and to_state < len(states):
                    to_state_val = states[to_state]
                else:
                    to_state_val = to_state
                mapped_transitions.append((from_state_val, to_state_val))
            self.transitions = mapped_transitions
        else:
            self.transitions = transitions
        
        # Validate transitions
        for from_state, to_state in self.transitions:
            if from_state not in states or to_state not in states:
                raise ValueError(f"Invalid transition {(from_state, to_state)}: states must be in {states}")
        
        # Set up state names
        if state_names is None:
            state_names = [str(state) for state in states]
        elif len(state_names) != len(states):
            raise ValueError("Length of state_names must match length of states")
            
        self.state_names = state_names
        self._state_to_idx = {state: idx for idx, state in enumerate(states)}
        self._idx_to_state = {idx: state for idx, state in enumerate(states)}
        self._state_to_name = {state: name for state, name in zip(states, state_names)}
        self._name_to_state = {name: state for state, name in zip(states, state_names)}
    
    def get_state_index(self, state):
        """Get index of a state"""
        if isinstance(state, str) and state in self._name_to_state:
            state = self._name_to_state[state]
        return self._state_to_idx[state]
    
    def get_state_name(self, state):
        """Get name of a state"""
        if isinstance(state, int):
            state = self._idx_to_state[state]
        return self._state_to_name[state]
    
    def is_valid_transition(self, from_state, to_state):
        """Check if a transition is valid"""
        # Convert state names to indices if necessary
        if isinstance(from_state, str):
            from_state = self._name_to_state[from_state]
        if isinstance(to_state, str):
            to_state = self._name_to_state[to_state]
        return (from_state, to_state) in self.transitions
    
    def get_next_states(self, state):
        """Get all possible next states from a given state"""
        if isinstance(state, str):
            state = self._name_to_state[state]
        return [to_state for from_state, to_state in self.transitions 
                if from_state == state]
    
    def get_previous_states(self, state):
        """Get all possible previous states to a given state"""
        if isinstance(state, str):
            state = self._name_to_state[state]
        return [from_state for from_state, to_state in self.transitions 
                if to_state == state]
    
    def transitions_from_state(self, state):
        """Get all transitions from a given state"""
        if isinstance(state, str):
            state = self._name_to_state[state]
        return [(from_state, to_state) for from_state, to_state in self.transitions 
                if from_state == state]
    
    def transitions_to_state(self, state):
        """Get all transitions to a given state"""
        if isinstance(state, str):
            state = self._name_to_state[state]
        return [(from_state, to_state) for from_state, to_state in self.transitions 
                if to_state == state]
    
    def get_state_names(self):
        """Get list of all state names"""
        return self.state_names.copy()
    
    def get_transition_names(self):
        """Get list of all transitions with state names"""
        return [(self.get_state_name(from_state), self.get_state_name(to_state))
                for from_state, to_state in self.transitions]