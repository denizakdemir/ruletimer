from ruletimer.models.competing_risks import RuleCompetingRisks
from ruletimer.data import CompetingRisks
from ruletimer.visualization.visualization import (
    plot_cumulative_incidence,
    plot_rule_importance
)
import numpy as np
import pandas as pd
import pytest
import os
import matplotlib.pyplot as plt
import joblib

def test_competing_risks_example():
    """Comprehensive example demonstrating competing risks analysis functionality"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data with realistic clinical features
    n_samples = 1000
    n_features = 5
    
    # Generate features with realistic distributions
    X = pd.DataFrame({
        'age': np.random.normal(65, 10, n_samples),
        'gender': np.random.binomial(1, 0.5, n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'blood_pressure': np.random.normal(120, 20, n_samples),
        'smoking': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Generate competing risks data with realistic hazard functions
    # Event 1: Cardiac event
    hazard1 = np.exp(
        0.02 * X['age'] + 
        0.01 * X['gender'] + 
        0.03 * X['smoking'] + 
        0.005 * X['blood_pressure']
    )
    
    # Event 2: Non-cardiac death
    hazard2 = np.exp(
        0.01 * X['age'] + 
        0.02 * X['bmi'] + 
        0.01 * X['blood_pressure']
    )
    
    # Generate times for each event type
    times1 = np.random.exponential(scale=5/hazard1)
    times2 = np.random.exponential(scale=5/hazard2)
    
    # Determine which event occurred first
    times = np.minimum(times1, times2)
    events = np.where(times1 <= times2, 1, 2)
    
    # Add censoring (30% censoring rate)
    censoring_times = np.random.exponential(scale=10, size=n_samples)
    censored = censoring_times < times
    times = np.minimum(times, censoring_times)
    events = np.where(censored, 0, events)
    
    # Create CompetingRisks object
    y = CompetingRisks(time=times, event=events)
    
    # Initialize and fit the model with advanced parameters
    model = RuleCompetingRisks(
        max_rules=32,
        max_depth=4,
        n_estimators=200,
        alpha=0.01,
        l1_ratio=0.5,
        hazard_method="nelson-aalen",
        min_samples_leaf=10,
        tree_type='classification',
        tree_growing_strategy='forest',
        prune_rules=True,
        random_state=42
    )
    
    # Fit the model
    model.fit(X, y)
    
    # Test predictions at specific time points
    test_times = np.linspace(0, 10, 100)
    
    # Test cause-specific hazard predictions
    for event_type in [1, 2]:
        hazard_vals = model.predict_cause_specific_hazard(X[:5], test_times, event_type)
        assert hazard_vals.shape == (5, len(test_times))
        assert np.all(hazard_vals >= 0)
    
    # Test cumulative incidence predictions
    for event_type in [1, 2]:
        cif_vals = model.predict_cumulative_incidence(X[:5], test_times, event_type)
        assert cif_vals.shape == (5, len(test_times))
        assert np.all((cif_vals >= 0) & (cif_vals <= 1))
    
    # Test feature importances
    for event_type in [1, 2]:
        importances = model.get_feature_importances(event_type)
        assert len(importances) == n_features
        assert np.all(importances >= 0)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot cumulative incidence functions using the visualization module
    cif_dict = {}
    for event_type in [1, 2]:
        cif_dict[event_type] = model.predict_cumulative_incidence(X[:5], test_times, event_type)
    
    plot_cumulative_incidence(
        test_times,
        cif_dict,
        event_names={1: 'Cardiac Event', 2: 'Non-cardiac Death'},
        figsize=(12, 8),
        title='Cumulative Incidence Functions'
    )
    plt.savefig('plots/cumulative_incidence.png')
    plt.close()
    
    # Plot rule importance using the visualization module
    fig = plot_rule_importance(model, top_n=10, figsize=(10, 6))
    fig.savefig('plots/rule_importance.png')
    plt.close()
    
    # Write detailed statistics to file
    with open('plots/statistics.txt', 'w') as f:
        f.write('Model Statistics:\n')
        f.write(f'Number of rules: {len(model.rules_)}\n')
        f.write(f'Number of features: {n_features}\n')
        f.write(f'Number of samples: {n_samples}\n')
        
        f.write('\nEvent Statistics:\n')
        for event_type in [1, 2]:
            f.write(f'\nEvent {event_type}:\n')
            importances = model.get_feature_importances(event_type)
            for feature, importance in zip(X.columns, importances):
                f.write(f'{feature}: {importance:.4f}\n')
        
        f.write('\nEvent Counts:\n')
        for event_type in [0, 1, 2]:
            count = np.sum(events == event_type)
            percentage = (count/n_samples)*100
            f.write(f'Event {event_type}: {count} ({percentage:.2f}%)\n')
    
    # Test model persistence
    model_path = 'plots/competing_risks_model.joblib'
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    
    # Verify loaded model predictions match original
    for event_type in [1, 2]:
        loaded_cif = loaded_model.predict_cumulative_incidence(X[:5], test_times, event_type)
        original_cif = model.predict_cumulative_incidence(X[:5], test_times, event_type)
        assert np.allclose(loaded_cif, original_cif)
    
    # Verify file creation
    assert os.path.exists('plots/cumulative_incidence.png')
    assert os.path.exists('plots/rule_importance.png')
    assert os.path.exists('plots/statistics.txt')
    assert os.path.exists(model_path) 