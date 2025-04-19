import numpy as np
import pandas as pd
from ruletimer.models.multi_state import RuleMultiState
from ruletimer.data import MultiState
from ruletimer.utils import StateStructure

# Create a simple test dataset
np.random.seed(42)
n_samples = 300  # Increased sample size for better distribution
n_features = 5

# Generate random features
X = np.random.randn(n_samples, n_features)

# Create synthetic multi-state data with valid transitions
patient_ids = []
start_times = []
end_times = []
start_states = []
end_states = []

# Generate transitions for each state
current_id = 0
for start_state in [1, 2, 3]:
    n_transitions = n_samples // 3
    
    # Generate transitions from current state to next state
    patient_ids.extend([current_id + i for i in range(n_transitions)])
    start_times.extend(np.zeros(n_transitions))
    end_times.extend(np.random.exponential(2, n_transitions))
    start_states.extend([start_state] * n_transitions)
    end_states.extend([start_state + 1] * n_transitions)
    
    current_id += n_transitions

# Convert to numpy arrays
patient_ids = np.array(patient_ids)
start_times = np.array(start_times)
end_times = np.array(end_times)
start_states = np.array(start_states)
end_states = np.array(end_states)

# Create MultiState object
y = MultiState(
    patient_id=patient_ids,
    start_time=start_times,
    end_time=end_times,
    start_state=start_states,
    end_state=end_states
)

# Create state structure
states = [1, 2, 3, 4]
transitions = [(1, 2), (2, 3), (3, 4)]
state_structure = StateStructure(states, transitions)

# Create and fit model
model = RuleMultiState(
    state_structure=state_structure,
    max_rules=10,
    max_depth=2,
    random_state=42
)
model.fit(X, y)

# Test times for prediction
test_times = np.linspace(0, 5, 10)

print("\nTesting transition probabilities:")
print("=================================")

# Test transition probabilities
for from_state in [1, 2, 3]:
    for to_state in [2, 3, 4]:
        if from_state < to_state:  # Only test valid transitions
            probs = model.predict_transition_probability(X, test_times, from_state, to_state)
            print(f"\nTransition {from_state}->{to_state}:")
            print(f"Shape: {probs.shape}")
            print(f"Mean probability: {probs.mean():.3f}")
            print(f"Min probability: {probs.min():.3f}")
            print(f"Max probability: {probs.max():.3f}")
            print(f"Probability over time: {probs[0]}")

print("\nTesting state occupation probabilities:")
print("======================================")

# Test state occupation probabilities
state_probs = model.predict_state_occupation(X, test_times)

print("\nState occupation probabilities over time (first sample):")
for state in sorted(state_probs.keys()):
    probs = state_probs[state]
    print(f"\nState {state}:")
    print(f"Shape: {probs.shape}")
    print(f"Mean probability: {probs.mean():.3f}")
    print(f"Min probability: {probs.min():.3f}")
    print(f"Max probability: {probs.max():.3f}")
    print(f"Probability over time: {probs[0]}")

# Verify probability sum to 1
total_prob = np.zeros_like(test_times)
for state in state_probs:
    total_prob += state_probs[state][0]

print("\nVerification:")
print("=============")
print(f"Sum of probabilities at each time point: {total_prob}")
print(f"Maximum deviation from 1.0: {np.abs(total_prob - 1.0).max():.6f}") 