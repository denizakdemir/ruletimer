"""
Tests for competing risks analysis
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from ruletimer.models.competing_risks import RuleCompetingRisks
from ruletimer.data import CompetingRisks

def test_fine_gray_model():
    """Test Fine-Gray subdistribution hazard model"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    times = np.random.exponential(scale=1.0, size=100)
    events = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.4, 0.3])
    
    # Create competing risks data
    y = CompetingRisks(times, events)
    
    # Fit model
    model = RuleCompetingRisks(event_types=['event1', 'event2'], model_type='fine-gray', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    cif = model.predict_cumulative_incidence(X, times_pred)
    
    # Check output format
    assert isinstance(cif, dict)
    assert all(isinstance(cif[event], np.ndarray) for event in cif)
    assert all(cif[event].shape == (100, 10) for event in cif)
    
    # Check probabilities
    assert all(np.all(cif[event] >= 0) for event in cif)
    assert all(np.all(cif[event] <= 1) for event in cif)

def test_cause_specific_model():
    """Test cause-specific hazard model"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    times = np.random.exponential(scale=1.0, size=100)
    events = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.4, 0.3])
    
    # Create competing risks data
    y = CompetingRisks(times, events)
    
    # Fit model
    model = RuleCompetingRisks(event_types=['event1', 'event2'], model_type='cause-specific', max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    hazard = model.predict_hazard(X, times_pred)
    
    # Check output format
    assert isinstance(hazard, dict)
    assert all(isinstance(hazard[event], np.ndarray) for event in hazard)
    assert all(hazard[event].shape == (100, 10) for event in hazard)
    
    # Check hazard rates
    assert all(np.all(hazard[event] >= 0) for event in hazard)

def test_rule_generation():
    """Test rule generation for competing risks"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    times = np.random.exponential(scale=1.0, size=100)
    events = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.4, 0.3])
    
    # Create competing risks data
    y = CompetingRisks(times, events)
    
    # Test different tree growing strategies
    strategies = ['single', 'forest', 'interaction']
    for strategy in strategies:
        model = RuleCompetingRisks(event_types=['event1', 'event2'], tree_growing_strategy=strategy, max_rules=10)
        model.fit(X, y)
        
        # Check rules were generated
        assert hasattr(model, 'rules_')
        assert len(model.rules_) > 0
        assert len(model.rules_) <= 10

def test_feature_importance():
    """Test feature importance computation"""
    # Generate synthetic data
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    times = np.random.exponential(scale=1.0, size=100)
    events = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.4, 0.3])
    
    # Create competing risks data
    y = CompetingRisks(times, events)
    
    # Fit model
    model = RuleCompetingRisks(event_types=['event1', 'event2'], max_rules=10)
    model.fit(X, y)
    
    # Check feature importance
    assert hasattr(model, 'feature_importances_')
    assert isinstance(model.feature_importances_, np.ndarray)
    assert len(model.feature_importances_) == 5
    assert np.all(model.feature_importances_ >= 0)
    assert np.allclose(np.sum(model.feature_importances_), 1.0)

def test_time_dependent_covariates():
    """Test support for time-dependent covariates"""
    # Generate synthetic data with time-dependent features
    n_samples = 100
    n_features = 5
    n_time_points = 10
    
    X = np.random.randn(n_samples, n_features, n_time_points)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Create competing risks data
    y = CompetingRisks(times, events)
    
    # Fit model
    model = RuleCompetingRisks(event_types=['event1', 'event2'], support_time_dependent=True, max_rules=10)
    model.fit(X, y)
    
    # Test predictions
    times_pred = np.linspace(0, 2, 10)
    cif = model.predict_cumulative_incidence(X, times_pred)
    
    # Check output format
    assert isinstance(cif, dict)
    assert all(isinstance(cif[event], np.ndarray) for event in cif)
    assert all(cif[event].shape == (100, 10) for event in cif) 