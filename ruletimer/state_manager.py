from typing import List, Tuple, Union, Optional
import numpy as np

class StateManager:
    """Manage state indices and transitions"""
    
    def __init__(self, states: List[str], transitions: List[Tuple[int, int]]):
        """
        Initialize state manager
        
        Parameters
        ----------
        states : list of str
            List of state names
        transitions : list of tuple
            List of valid transitions as (from_state, to_state) pairs
        """
        self.states = states
        self.transitions = transitions
        
        # Use 0-based indexing internally
        self._state_to_idx = {state: i for i, state in enumerate(states)}
        self._idx_to_state = {i: state for i, state in enumerate(states)}
        
        # Validate transitions
        self._validate_transitions()
    
    def _validate_transitions(self) -> None:
        """Validate transition definitions"""
        n_states = len(self.states)
        for from_state, to_state in self.transitions:
            if from_state >= n_states or to_state >= n_states:
                raise ValueError(f"Invalid transition: state index out of range")
            if from_state == to_state:
                raise ValueError(f"Invalid transition: self-transition not allowed")
    
    def to_internal_index(self, state: Union[str, int]) -> int:
        """
        Convert external state representation to internal index
        
        Parameters
        ----------
        state : str or int
            State name or index
            
        Returns
        -------
        int
            Internal state index
            
        Raises
        ------
        ValueError
            If state is invalid
        """
        if isinstance(state, str):
            if state not in self._state_to_idx:
                raise ValueError(f"Invalid state name: {state}")
            return self._state_to_idx[state]
        elif isinstance(state, int):
            if state not in self._idx_to_state:
                raise ValueError(f"Invalid state index: {state}")
            return state
        else:
            raise ValueError("State must be string or integer")
    
    def to_external_state(self, idx: int) -> str:
        """
        Convert internal index to external state representation
        
        Parameters
        ----------
        idx : int
            Internal state index
            
        Returns
        -------
        str
            State name
            
        Raises
        ------
        ValueError
            If index is invalid
        """
        if idx not in self._idx_to_state:
            raise ValueError(f"Invalid state index: {idx}")
        return self._idx_to_state[idx]
    
    def get_possible_transitions(self, from_state: Optional[Union[str, int]] = None) -> List[Union[int, Tuple[int, int]]]:
        """
        Get possible transitions from a state or all transitions if no state is specified

        Parameters
        ----------
        from_state : str or int, optional
            State to get transitions from. If None, returns all transitions.

        Returns
        -------
        list
            If from_state is None: List of (from_state, to_state) tuples
            If from_state is provided: List of possible target state indices
        """
        if from_state is None:
            # Return all transitions sorted by from_state, then to_state
            return sorted(self.transitions, key=lambda x: (x[0], x[1]))
        
        from_idx = self.to_internal_index(from_state)
        # Return only target states when from_state is provided
        return sorted([to_idx for from_idx_, to_idx in self.transitions if from_idx_ == from_idx])

    def validate_transition(self, from_state: Union[str, int], to_state: Union[str, int]) -> bool:
        """
        Validate if a transition between two states is valid

        Parameters
        ----------
        from_state : str or int
            Starting state
        to_state : str or int
            Target state

        Returns
        -------
        bool
            True if transition is valid, False otherwise
        """
        try:
            from_idx = self.to_internal_index(from_state)
            to_idx = self.to_internal_index(to_state)
            return (from_idx, to_idx) in self.transitions
        except ValueError:
            return False
    
    def is_absorbing_state(self, state: Union[str, int]) -> bool:
        """
        Check if a state is absorbing (has no outgoing transitions)

        Parameters
        ----------
        state : str or int
            State to check

        Returns
        -------
        bool
            True if state is absorbing, False otherwise
        """
        state_idx = self.to_internal_index(state)
        # A state is absorbing if it has no outgoing transitions
        return not any(from_idx == state_idx for from_idx, _ in self.transitions)

    def get_absorbing_states(self) -> List[int]:
        """
        Get all absorbing states in the system

        Returns
        -------
        list of int
            List of absorbing state indices
        """
        return [i for i in range(len(self.states)) if self.is_absorbing_state(i)]
    
    def validate_state_sequence(self, sequence: List[Union[str, int]]) -> bool:
        """
        Validate a sequence of states
        
        Parameters
        ----------
        sequence : list
            Sequence of states
            
        Returns
        -------
        bool
            True if sequence is valid
        """
        if len(sequence) < 2:
            return True
            
        # Convert all states to internal indices
        try:
            indices = [self.to_internal_index(state) for state in sequence]
        except ValueError:
            return False
            
        # Check each transition
        for i in range(len(indices) - 1):
            if not self.validate_transition(indices[i], indices[i + 1]):
                return False
                
        return True 