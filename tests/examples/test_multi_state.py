import numpy as np
import pandas as pd
import pytest
from ruletimer.data import MultiState
from ruletimer.utils import StateStructure
from ruletimer.models.base_multi_state import RuleMultiState
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
    state_names = ["Healthy", "Mild", "Moderate", "Severe"]
    structure = StateStructure(
        states=states,
        transitions=[(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)],
        state_names=state_names
    )
    
    # Generate synthetic data with realistic clinical features
    n_patients = 1000
    n_features = 5
    feature_names = ['age', 'gender', 'bmi', 'blood_pressure', 'smoking']
    
    # Generate base features with realistic distributions
    X_base = pd.DataFrame({
        'age': np.random.normal(65, 10, n_patients),
        'gender': np.random.binomial(1, 0.5, n_patients),
        'bmi': np.random.normal(25, 5, n_patients),
        'blood_pressure': np.random.normal(120, 20, n_patients),
        'smoking': np.random.binomial(1, 0.3, n_patients)
    })
    
    # Generate transition times and states with realistic progression patterns
    start_times = []
    end_times = []
    start_states = []
    end_states = []
    patient_id_expanded = []
    
    for i in range(n_patients):
        current_state = 0  # Start in Healthy state
        current_time = 0
        max_time = 10
        has_transition = False
        
        # Generate transitions based on patient characteristics
        while current_time < max_time:
            # Base transition rate modified by patient characteristics
            base_rate = 0.5
            age_effect = 0.01 * X_base.iloc[i]['age']
            smoking_effect = 0.2 * X_base.iloc[i]['smoking']
            bp_effect = 0.005 * X_base.iloc[i]['blood_pressure']
            
            transition_rate = base_rate * np.exp(age_effect + smoking_effect + bp_effect)
            next_time = current_time + np.random.exponential(scale=1/transition_rate)
            
            if next_time > max_time:
                break
            
            # Determine next state based on current state and patient characteristics
            if current_state == 0:  # Healthy
                if X_base.iloc[i]['smoking'] == 1:
                    next_state = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                else:
                    next_state = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            elif current_state == 1:  # Mild
                if X_base.iloc[i]['blood_pressure'] > 140:
                    next_state = np.random.choice([2, 3], p=[0.4, 0.6])
                else:
                    next_state = np.random.choice([2, 3], p=[0.7, 0.3])
            elif current_state == 2:  # Moderate
                next_state = 3  # Must progress to Severe
            else:
                break
            
            patient_id_expanded.append(i)
            start_times.append(current_time)
            end_times.append(next_time)
            start_states.append(current_state)
            end_states.append(next_state)
            has_transition = True
            
            current_state = next_state
            current_time = next_time
        
        if not has_transition:
            patient_id_expanded.append(i)
            start_times.append(0)
            end_times.append(max_time)
            start_states.append(0)
            end_states.append(4)  # Censored state
    
    # Create MultiState object
    multi_state = MultiState(
        patient_id=patient_id_expanded,
        start_time=start_times,
        end_time=end_times,
        start_state=start_states,
        end_state=end_states
    )
    
    # Create and fit model with advanced parameters
    model = RuleMultiState(
        max_rules=32,
        alpha=0.01,
        state_structure=structure,
        max_depth=4,
        min_samples_leaf=10,
        n_estimators=200,
        tree_type='classification',
        tree_growing_strategy='forest',
        prune_rules=True,
        l1_ratio=0.2,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_base, multi_state)
    
    # Test predictions
    test_times = np.linspace(0, 10, 100)
    
    # Test state occupation probabilities
    state_probs = model.predict_state_occupation(X_base[:5], test_times, initial_state=0)
    assert set(state_probs.keys()) == set(states)
    for state in states:
        assert state_probs[state].shape == (5, len(test_times))
        assert np.all((state_probs[state] >= 0) & (state_probs[state] <= 1))
    
    # Test cumulative incidence functions
    for state in states:
        cif = model.predict_cumulative_incidence(X_base[:5], test_times, state)
        assert cif.shape == (5, len(test_times))
        assert np.all(cif >= 0)
        assert np.all(cif <= 1)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot state occupation probabilities using the visualization module
    plot_state_occupation(
        test_times,
        state_probs,
        state_names=state_names,
        figsize=(12, 8),
        title='State Occupation Probabilities'
    )
    plt.savefig('plots/state_occupation.png')
    plt.close()
    
    # Plot state transitions using the visualization module
    plot_state_transitions(model, X_base[:5], time=5.0, figsize=(12, 8))
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
        f.write(f'Number of transitions: {len(structure.transitions)}\n')
        f.write(f'Number of patients: {n_patients}\n')
        
        f.write('\nTransition Statistics:\n')
        for from_state, to_state in structure.transitions:
            f.write(f'\n{state_names[from_state]} → {state_names[to_state]}:\n')
            importances = model.get_feature_importances((from_state, to_state))
            for feature, importance in zip(X_base.columns, importances):
                f.write(f'{feature}: {importance:.4f}\n')
    
    # Test model persistence
    model_path = 'plots/multi_state_model.joblib'
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    
    # Verify loaded model predictions match original
    loaded_state_probs = loaded_model.predict_state_occupation(
        X_base[:5], test_times, initial_state=0
    )
    for state in states:
        assert np.allclose(state_probs[state], loaded_state_probs[state])
    
    # Verify file creation
    assert os.path.exists('plots/state_occupation.png')
    assert os.path.exists('plots/state_transitions.png')
    assert os.path.exists('plots/rule_importance.png')
    assert os.path.exists('plots/statistics.txt')
    assert os.path.exists(model_path) 