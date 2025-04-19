import numpy as np
import pandas as pd
import pytest
from ruletimer.models.survival import RuleSurvivalCox
from ruletimer.data import Survival
import os

def test_survival_analysis_example():
    """Test the survival analysis example functionality"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate features
    X = np.zeros((n_samples, n_features))
    X[:, 0] = np.random.normal(65, 10, n_samples)  # Age
    X[:, 1] = np.random.binomial(1, 0.5, n_samples)  # Gender
    X[:, 2] = np.random.normal(25, 5, n_samples)  # BMI
    X[:, 3] = np.random.normal(120, 20, n_samples)  # Blood pressure
    X[:, 4] = np.random.binomial(1, 0.3, n_samples)  # Smoking
    
    # Generate survival times
    hazard = np.exp(0.01 * X[:, 0] + 0.01 * X[:, 1] + 0.005 * X[:, 2] + 0.01 * X[:, 3] + 0.02 * X[:, 4])
    times = np.random.exponential(scale=5/hazard)
    events = np.random.binomial(1, 0.7, n_samples)
    
    # Create Survival object
    y = Survival(time=times, event=events)
    
    # Initialize and fit the model
    model = RuleSurvivalCox(
        max_rules=32,
        max_depth=4,
        n_estimators=200,
        alpha=0.01,
        l1_ratio=0.5,
        hazard_method="nelson-aalen",
        random_state=42
    )
    
    # Fit the model
    model.fit(X, y)
    
    # Test predictions
    test_times = np.array([1.0, 2.0, 3.0])
    survival_probs = model.predict_survival(X[:5], test_times)
    assert survival_probs.shape == (5, 3)
    assert np.all((survival_probs >= 0) & (survival_probs <= 1))
    
    # Test hazard predictions
    hazard_vals = model.predict_hazard(X[:5], test_times)
    assert hazard_vals.shape == (5, 3)
    assert np.all(hazard_vals >= 0)
    
    # Test cumulative hazard predictions
    cum_hazard_vals = model.predict_cumulative_hazard(X[:5], test_times)
    assert cum_hazard_vals.shape == (5, 3)
    assert np.all(cum_hazard_vals >= 0)
    
    # Test feature importances
    importances = model._compute_feature_importances()
    assert len(importances) == n_features
    assert np.all(importances >= 0)
    
    # Test risk group predictions
    risk_groups = pd.qcut(model.predict_hazard(X, [2.0])[:, 0], q=3, labels=['Low', 'Medium', 'High'])
    assert len(risk_groups) == n_samples
    assert len(np.unique(risk_groups)) == 3
    
    # Test survival curves for different risk groups
    for group in ['Low', 'Medium', 'High']:
        group_mask = risk_groups == group
        group_survival = model.predict_survival(X[group_mask], test_times)
        assert group_survival.shape[1] == len(test_times)
    
    # Check if plots directory exists
    assert os.path.exists('plots')
    
    # Check if statistics file exists
    assert os.path.exists('plots/statistics.txt') 