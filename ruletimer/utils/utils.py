"""
Utility classes and functions
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np

class StateStructure:
    """Class for managing multi-state model structure (0-based indexing)"""
    
    def __init__(self, states: List[str], transitions: List[Tuple[int, int]],
                 initial_state: int = 0):
        """
        Initialize state structure
        
        Parameters
        ----------
        states : list of str
            List of state names
        transitions : list of tuple
            List of allowed transitions as (from_state, to_state) pairs using 0-based indexing
        initial_state : int, default=0
            Index of initial state (0-based)
        """
        self.states = states
        self.transitions = transitions
        self.initial_state = initial_state
        
        # Build transition dictionaries
        self._next_states = {}
        self._previous_states = {}
        
        for from_state, to_state in transitions:
            if from_state not in self._next_states:
                self._next_states[from_state] = set()
            if to_state not in self._previous_states:
                self._previous_states[to_state] = set()
            
            self._next_states[from_state].add(to_state)
            self._previous_states[to_state].add(from_state)
    
    def transitions_from_state(self, state: int) -> Set[int]:
        """
        Get all possible next states from a given state
        
        Parameters
        ----------
        state : int
            Current state index (1-based)
            
        Returns
        -------
        next_states : set of int
            Set of possible next state indices (1-based)
        """
        return self._next_states.get(state, set())
    
    def transition_idx(self, from_state: int, to_state: int) -> Optional[Tuple[int, int]]:
        """
        Get the transition tuple for a given state pair if it exists
        
        Parameters
        ----------
        from_state : int
            Starting state index (1-based)
        to_state : int
            Target state index (1-based)
            
        Returns
        -------
        transition : tuple or None
            (from_state, to_state) tuple if transition exists, None otherwise
        """
        return (from_state, to_state) if (from_state, to_state) in self.transitions else None
    
    def get_next_states(self, state: int) -> Set[int]:
        """
        Get states that can be reached from the given state
        
        Parameters
        ----------
        state : int
            Current state (1-based)
            
        Returns
        -------
        next_states : set of int
            Set of states that can be reached from the current state (1-based)
        """
        return self._next_states.get(state, set())
    
    def get_previous_states(self, state: int) -> Set[int]:
        """
        Get states that can reach the given state
        
        Parameters
        ----------
        state : int
            Current state (1-based)
            
        Returns
        -------
        previous_states : set of int
            Set of states that can reach the current state (1-based)
        """
        return self._previous_states.get(state, set())
    
    def is_valid_transition(self, from_state: int, to_state: int) -> bool:
        """
        Check if a transition is valid
        
        Parameters
        ----------
        from_state : int
            Starting state (1-based)
        to_state : int
            Target state (1-based)
            
        Returns
        -------
        is_valid : bool
            True if the transition is valid, False otherwise
        """
        return (from_state, to_state) in self.transitions
    
    def get_state_name(self, state: int) -> str:
        """
        Get the name of a state
        
        Parameters
        ----------
        state : int
            State index (0-based)
        
        Returns
        -------
        name : str
            State name
        """
        return self.states[state]
    
    def get_state_index(self, name: str) -> int:
        """
        Get the index of a state by name
        
        Parameters
        ----------
        name : str
            State name
        
        Returns
        -------
        index : int
            State index (0-based)
        """
        return self.states.index(name)