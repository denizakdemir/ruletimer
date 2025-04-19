import numpy as np
import pandas as pd
import pytest
from ruletimer.models.survival import RuleSurvivalCox
from ruletimer.data import Survival
from ruletimer.visualization.visualization import plot_rule_importance
import os
import matplotlib.pyplot as plt
import joblib

def test_survival_analysis_example():
    """Comprehensive example demonstrating survival analysis functionality"""
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
    
    # Generate survival times with realistic hazard function
    hazard = np.exp(
        0.01 * X['age'] + 
        0.01 * X['gender'] + 
        0.005 * X['bmi'] + 
        0.01 * X['blood_pressure'] + 
        0.02 * X['smoking']
    )
    times = np.random.exponential(scale=5/hazard)
    
    # Add censoring (30% censoring rate)
    censoring_times = np.random.exponential(scale=10, size=n_samples)
    events = times <= censoring_times
    times = np.minimum(times, censoring_times)
    
    # Create Survival object
    y = Survival(time=times, event=events)
    
    # Initialize and fit the model with advanced parameters
    model = RuleSurvivalCox(
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
    
    # Test survival probability predictions
    survival_probs = model.predict_survival(X[:5], test_times)
    assert survival_probs.shape == (5, len(test_times))
    assert np.all((survival_probs >= 0) & (survival_probs <= 1))
    
    # Test hazard predictions
    hazard_vals = model.predict_hazard(X[:5], test_times)
    assert hazard_vals.shape == (5, len(test_times))
    assert np.all(hazard_vals >= 0)
    
    # Test cumulative hazard predictions
    cum_hazard_vals = model.predict_cumulative_hazard(X[:5], test_times)
    assert cum_hazard_vals.shape == (5, len(test_times))
    assert np.all(cum_hazard_vals >= 0)
    
    # Test feature importances
    importances = model._compute_feature_importances()
    assert len(importances) == n_features
    assert np.all(importances >= 0)
    
    # Test risk group predictions
    risk_scores = model.predict_hazard(X, [2.0])[:, 0]
    risk_groups = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])
    assert len(risk_groups) == n_samples
    assert len(np.unique(risk_groups)) == 3
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Generate and save visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot survival curves for different risk groups
    for group in ['Low', 'Medium', 'High']:
        group_mask = risk_groups == group
        group_survival = model.predict_survival(X[group_mask], test_times)
        mean_survival = np.mean(group_survival, axis=0)
        plt.plot(test_times, mean_survival, label=f'{group} Risk')
    
    plt.title('Survival Curves by Risk Group')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig('plots/survival_curves.png')
    plt.close()
    
    # Plot hazard curves
    plt.figure(figsize=(12, 8))
    for group in ['Low', 'Medium', 'High']:
        group_mask = risk_groups == group
        group_hazard = model.predict_hazard(X[group_mask], test_times)
        mean_hazard = np.mean(group_hazard, axis=0)
        plt.plot(test_times, mean_hazard, label=f'{group} Risk')
    
    plt.title('Hazard Curves by Risk Group')
    plt.xlabel('Time')
    plt.ylabel('Hazard Rate')
    plt.legend()
    plt.savefig('plots/hazard_curves.png')
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
        f.write('\nFeature Importances:\n')
        for feature, importance in zip(X.columns, importances):
            f.write(f'{feature}: {importance:.4f}\n')
        
        f.write('\nRisk Group Statistics:\n')
        for group in ['Low', 'Medium', 'High']:
            group_mask = risk_groups == group
            group_size = np.sum(group_mask)
            f.write(f'\n{group} Risk Group:\n')
            f.write(f'Number of patients: {group_size}\n')
            f.write(f'Percentage: {(group_size/n_samples)*100:.2f}%\n')
    
    # Verify file creation
    assert os.path.exists('plots/survival_curves.png')
    assert os.path.exists('plots/hazard_curves.png')
    assert os.path.exists('plots/rule_importance.png')
    assert os.path.exists('plots/statistics.txt')
    
    # Test model persistence
    model_path = 'plots/survival_model.joblib'
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    
    # Verify loaded model predictions match original
    loaded_survival = loaded_model.predict_survival(X[:5], test_times)
    assert np.allclose(survival_probs, loaded_survival) 