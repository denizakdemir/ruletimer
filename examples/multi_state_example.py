"""
Example of using RuleMultiState for multi-state survival analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ruletimer.models import RuleMultiState
from ruletimer.utils import StateStructure
from ruletimer.visualization import plot_state_occupation, plot_transition_hazard

def generate_multi_state_data(n_samples=1000):
    """Generate synthetic multi-state data."""
    # Generate features
    np.random.seed(42)
    X = np.random.randn(n_samples, 5)
    
    # Define states and transitions
    states = ['healthy', 'sick', 'dead']
    transitions = [(0, 1), (1, 2)]  # healthy -> sick -> dead
    
    # Generate transition times and events
    transition_data = {}
    
    # Healthy to Sick transition
    hazard_healthy_sick = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
    times_healthy_sick = np.random.exponential(1/hazard_healthy_sick)
    events_healthy_sick = np.ones(n_samples)
    transition_data[(0, 1)] = {
        'times': times_healthy_sick,
        'events': events_healthy_sick
    }
    
    # Sick to Dead transition
    hazard_sick_dead = np.exp(0.3 * X[:, 2] + 0.4 * X[:, 3])
    times_sick_dead = times_healthy_sick + np.random.exponential(1/hazard_sick_dead)
    events_sick_dead = np.ones(n_samples)
    transition_data[(1, 2)] = {
        'times': times_sick_dead,
        'events': events_sick_dead
    }
    
    return X, states, transitions, transition_data

def main():
    # Generate data
    X, states, transitions, transition_data = generate_multi_state_data()
    
    # Create state structure
    state_structure = StateStructure(states=states, transitions=transitions)
    
    # Initialize and fit the model
    model = RuleMultiState(
        state_structure=state_structure,
        max_rules=50,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    
    print("Fitting RuleMultiState model...")
    model.fit(X, transition_data)
    print("Model fitting completed.")
    
    # Generate prediction times
    times = np.linspace(0, 10, 100)
    
    # Select a few samples for visualization
    sample_indices = [0, 1, 2]
    X_samples = X[sample_indices]
    
    # Plot state occupation probabilities
    plt.figure(figsize=(12, 6))
    plot_state_occupation(
        model,
        X_samples,
        times,
        initial_state='healthy',
        sample_labels=[f'Sample {i}' for i in sample_indices]
    )
    plt.title('State Occupation Probabilities')
    plt.tight_layout()
    plt.savefig('state_occupation.png')
    plt.close()
    
    # Plot transition hazards
    plt.figure(figsize=(12, 6))
    plot_transition_hazard(
        model,
        X_samples,
        times,
        from_state='healthy',
        to_state='sick',
        sample_labels=[f'Sample {i}' for i in sample_indices]
    )
    plt.title('Transition Hazard: Healthy -> Sick')
    plt.tight_layout()
    plt.savefig('transition_hazard.png')
    plt.close()
    
    # Print feature importances
    print("\nFeature Importances:")
    for transition in transitions:
        importances = model.get_feature_importances(transition)
        print(f"\nTransition {states[transition[0]]} -> {states[transition[1]]}:")
        for i, importance in enumerate(importances):
            print(f"  Feature {i}: {importance:.4f}")
    
    # Print some example rules
    print("\nExample Rules:")
    for transition, rules in model.rules_.items():
        print(f"\nTransition {states[transition[0]]} -> {states[transition[1]]}:")
        for i, rule in enumerate(rules[:3]):  # Show first 3 rules
            print(f"  Rule {i+1}: {rule}")

if __name__ == "__main__":
    main() 