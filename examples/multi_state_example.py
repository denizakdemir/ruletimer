"""
Example of using RuleMultiState for multi-state survival analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from ruletimer.models import RuleMultiState
from ruletimer.data import MultiState
from ruletimer.utils import StateStructure
from ruletimer.visualization import (
    plot_state_occupation,
    plot_state_transitions,
    plot_rule_importance,
    plot_importance_comparison
)

def generate_multi_state_data(n_samples=1000, n_features=5):
    """Generate example multi-state data."""
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate transition times based on features
    transitions = []
    start_times = []
    end_times = []
    start_states = []
    end_states = []
    sample_indices = []  # Keep track of which sample each transition belongs to
    
    for i in range(n_samples):
        # Start in state 0 (Healthy)
        current_state = 0
        current_time = 0.0
        
        while current_time < 10.0 and current_state < 2:
            start_states.append(current_state)
            start_times.append(current_time)
            sample_indices.append(i)
            
            # Generate transition time based on features
            if current_state == 0:  # Healthy -> Mild
                hazard = np.exp(0.5 * X[i, 0] + 0.3 * X[i, 1])
                time_to_next = np.random.exponential(scale=1/hazard)
            else:  # Mild -> Severe
                hazard = np.exp(0.3 * X[i, 2] + 0.2 * X[i, 3])
                time_to_next = np.random.exponential(scale=1/hazard)
            
            next_time = current_time + time_to_next
            
            # Ensure we have enough transitions
            if next_time < 10.0 or np.random.rand() < 0.3:  # 30% chance of transition even if time > 10
                # Transition occurs
                end_times.append(min(next_time, 10.0))
                end_states.append(current_state + 1)
                current_state += 1
                current_time = min(next_time, 10.0)
            else:
                # Censored
                end_times.append(10.0)
                end_states.append(current_state + 1)
                break
    
    # Convert to numpy arrays
    start_times = np.array(start_times)
    end_times = np.array(end_times)
    start_states = np.array(start_states)
    end_states = np.array(end_states)
    sample_indices = np.array(sample_indices)
    
    # Create state structure
    state_structure = StateStructure(
        states=[0, 1, 2],
        transitions=[(0, 1), (1, 2)],
        state_names=["Healthy", "Mild", "Severe"]
    )
    
    # Create multi-state data object
    multi_state = MultiState(
        start_state=start_states,
        end_state=end_states,
        start_time=start_times,
        end_time=end_times
    )
    
    # Get features for each transition
    X_expanded = X[sample_indices]
    
    return X_expanded, state_structure, multi_state

def main():
    # Generate data
    X, state_structure, multi_state = generate_multi_state_data()
    
    # Initialize and fit model
    model = RuleMultiState(
        state_structure=state_structure,
        max_rules=50,
        max_depth=3,
        n_estimators=100,
        min_support=0.01,  # Lower minimum support
        min_confidence=0.1,  # Lower minimum confidence
        max_impurity=0.01,  # Lower maximum impurity
        alpha=0.1,  # Increase regularization
        l1_ratio=0.5,  # Equal mix of L1 and L2
        min_samples_leaf=10,  # Minimum samples in leaf nodes
        tree_type='classification',  # Use classification trees
        tree_growing_strategy='forest',  # Use random forest
        prune_rules=True,  # Enable rule pruning
        random_state=42
    )
    model.fit(X, multi_state)
    
    # Make predictions
    test_times = np.linspace(0, 10, 100)
    state_probs = model.predict_state_occupation(X[:1], test_times, initial_state=0)  # Start from Healthy state
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot state occupation probabilities
    plt.subplot(2, 1, 1)
    plot_state_occupation(test_times, state_probs, state_names=state_structure.state_names)
    plt.title("State Occupation Probabilities")
    
    # Plot state transitions
    plt.subplot(2, 1, 2)
    plot_state_transitions(model, X[:1], test_times[-1])  # Plot transitions at the last time point
    plt.title("State Transitions")
    
    plt.tight_layout()
    plt.show()
    
    # Print some example rules
    print("\nExample Rules:")
    for transition, rules in model.rules_.items():
        print(f"\nTransition {state_structure.state_names[transition[0]]} -> {state_structure.state_names[transition[1]]}:")
        for i, rule in enumerate(rules[:3]):  # Show first 3 rules
            print(f"  Rule {i+1}: {rule}")

if __name__ == "__main__":
    main() 