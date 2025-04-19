import numpy as np
import pandas as pd
import pytest
from ruletimer.data import MultiState
from ruletimer.utils import StateStructure
from ruletimer.models import RuleMultiState
from ruletimer.visualization.visualization import (
    plot_state_occupation,
    plot_state_transitions,
    plot_rule_importance
)
import os
import matplotlib.pyplot as plt
import joblib

def test_multi_state_example():
    """Comprehensive example demonstrating multi-state survival analysis functionality"""
    # Define the state structure with meaningful state names
    states = [0, 1, 2, 3]  # Healthy, Mild, Moderate, Severe
    transitions = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    state_names = ["Healthy", "Mild", "Moderate", "Severe"]
    structure = StateStructure(
        states=states,
        transitions=transitions,
        state_names=state_names
    )
    
    # Generate synthetic data
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # Generate transition times and events
    transition_data = {}
    for transition in transitions:
        # Generate transition-specific hazard
        hazard = np.exp(0.1 * X[:, 0] + 0.2 * X[:, 1])
        times = np.random.exponential(1/hazard)
        events = np.random.binomial(1, 0.7, size=n_samples)  # 70% event rate
        transition_data[transition] = {
            'times': times,
            'events': events
        }
    
    # Initialize and fit model
    model = RuleMultiState(
        state_structure=structure,
        max_rules=10,
        max_depth=3,
        n_estimators=50,
        random_state=42
    )
    
    # Fit model
    model.fit(X, transition_data)
    
    # Make predictions
    times = np.linspace(0, 10, 100)
    predictions = model.predict_cumulative_incidence(
        X[:5],  # Predict for first 5 samples
        times,
        target_state="Severe"  # Use state name
    )
    
    # Basic assertions
    assert model.is_fitted_
    assert predictions.shape == (5, 100)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    # Check feature importances
    for transition in transitions:
        importances = model.get_feature_importances(transition)
        assert isinstance(importances, np.ndarray)
        assert len(importances) > 0
        assert np.all(importances >= 0)
        
    # Check rules
    for transition in transitions:
        rules = model.rules_[transition]
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert all(isinstance(rule, str) for rule in rules)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot state occupation probabilities using the visualization module
    plot_state_occupation(
        times,
        model.predict_state_occupation(X[:5], times, initial_state=0),
        state_names=state_names,
        figsize=(12, 8),
        title='State Occupation Probabilities'
    )
    plt.savefig('plots/state_occupation.png')
    plt.close()
    
    # Plot state transitions using the visualization module
    plot_state_transitions(model, X[:5], time=5.0, figsize=(12, 8))
    plt.savefig('plots/state_transitions.png')
    plt.close()
    
    # Plot rule importance using the visualization module
    fig = plot_rule_importance(model, top_n=10, figsize=(10, 6))
    fig.savefig('plots/rule_importance.png')
    plt.close()
    
    # Write detailed statistics to file
    with open('plots/statistics.txt', 'w') as f:
        f.write('Model Statistics:\n')
        f.write(f'Number of rules: {len(model.rules_)}\n')
        f.write(f'Number of states: {len(states)}\n')
        f.write(f'Number of transitions: {len(transitions)}\n')
        f.write(f'Number of patients: {n_samples}\n')
        
        f.write('\nTransition Statistics:\n')
        for transition in transitions:
            f.write(f'\n{state_names[transition[0]]} → {state_names[transition[1]]}:\n')
            importances = model.get_feature_importances(transition)
            for feature, importance in zip(X.columns, importances):
                f.write(f'{feature}: {importance:.4f}\n')
    
    # Test model persistence
    model_path = 'plots/multi_state_model.joblib'
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    
    # Verify loaded model predictions match original
    loaded_predictions = loaded_model.predict_cumulative_incidence(
        X[:5], times, target_state="Severe"
    )
    assert np.allclose(predictions, loaded_predictions)
    
    # Verify file creation
    assert os.path.exists('plots/state_occupation.png')
    assert os.path.exists('plots/state_transitions.png')
    assert os.path.exists('plots/rule_importance.png')
    assert os.path.exists('plots/statistics.txt')
    assert os.path.exists(model_path) 