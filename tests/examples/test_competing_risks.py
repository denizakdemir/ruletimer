from ruletimer.models import RuleCompetingRisks
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
    
    # Convert DataFrame to numpy array
    X_array = X.to_numpy()
    
    # Initialize and fit the model with advanced parameters
    model = RuleCompetingRisks(
        n_rules=32,
        min_support=0.1,
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=1000,
        tol=1e-4,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_array, y)
    
    # Test predictions
    test_times = np.linspace(0, 10, 100)
    predictions = model.predict_cumulative_incidence(X[:5].to_numpy(), test_times, event_type="Event1")
    
    # Basic assertions
    assert model.is_fitted_
    assert predictions.shape == (5, 100)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    # Check feature importances
    importances = model.get_feature_importances("Event1")
    assert isinstance(importances, np.ndarray)
    assert len(importances) == X.shape[1]
    assert np.all(importances >= 0)
    
    # Check rules
    assert isinstance(model.transition_rules_, dict)
    assert len(model.transition_rules_) == 2  # Two event types
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot cumulative incidence functions using the visualization module
    plot_cumulative_incidence(
        model,
        X[:5].to_numpy(),
        event_types=[1, 2],
        times=test_times,
        figsize=(12, 8)
    )
    plt.savefig('plots/cumulative_incidence.png')
    plt.close()
    
    # Write detailed statistics to file
    with open('plots/statistics.txt', 'w') as f:
        f.write('Model Statistics:\n')
        f.write(f'Number of samples: {n_samples}\n')
        f.write(f'Number of features: {n_features}\n')
        
        f.write('\nEvent Statistics:\n')
        for event_type in ["Event1", "Event2"]:
            f.write(f'\nEvent {event_type}:\n')
            importances = model.get_feature_importances(event_type)
            for i, feat in enumerate(X.columns):
                f.write(f'{feat}: {importances[i]:.4f}\n')
        
        f.write('\nEvent Counts:\n')
        event_type_map = {0: "Initial", 1: "Event1", 2: "Event2"}
        for event_type in [0, 1, 2]:
            count = np.sum(events == event_type)
            percentage = (count/n_samples)*100
            event_name = event_type_map[event_type]
            f.write(f'{event_name}: {count} ({percentage:.1f}%)\n')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/competing_risks_model.pkl')
    
    # Load model and verify predictions match
    loaded_model = joblib.load('models/competing_risks_model.pkl')
    
    # Verify loaded model predictions match original
    for event_type in ["Event1", "Event2"]:
        loaded_cif = loaded_model.predict_cumulative_incidence(X[:5].to_numpy(), test_times, event_type)
        original_cif = model.predict_cumulative_incidence(X[:5].to_numpy(), test_times, event_type)
        np.testing.assert_array_almost_equal(loaded_cif, original_cif)
    
    # Verify file creation
    assert os.path.exists('plots/cumulative_incidence.png')
    assert os.path.exists('plots/statistics.txt')
    assert os.path.exists('models/competing_risks_model.pkl') 