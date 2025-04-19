"""
Tests for survival analysis functionality
"""

import numpy as np
import pandas as pd
import pytest
from ruletimer import Survival, RuleSurvival

def test_survival_data_validation():
    """Test validation of survival data"""
    # Valid data
    time = np.array([1, 2, 3])
    event = np.array([1, 0, 1])
    y = Survival(time, event)
    assert np.array_equal(y.time, time)
    assert np.array_equal(y.event, event)
    
    # Invalid data - different lengths
    with pytest.raises(ValueError):
        Survival(np.array([1, 2]), np.array([1, 0, 1]))
    
    # Invalid data - negative times
    with pytest.raises(ValueError):
        Survival(np.array([-1, 2, 3]), np.array([1, 0, 1]))
    
    # Invalid data - invalid event indicators
    with pytest.raises(ValueError):
        Survival(np.array([1, 2, 3]), np.array([1, 2, 1]))

def test_rule_survival_fit():
    """Test fitting of rule survival model"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    y = Survival(time, event)
    
    # Fit model
    model = RuleSurvival(max_rules=10, alpha=0.1)
    model.fit(X, y)
    
    # Check that rules were extracted
    assert model.rules_ is not None
    assert len(model.rules_) <= 10
    
    # Check that weights were fitted
    assert model.rule_weights_ is not None
    assert len(model.rule_weights_) == len(model.rules_)
    
    # Check that baseline hazard was computed
    assert model.baseline_hazard_ is not None
    assert len(model.baseline_hazard_) == 2
    assert len(model.baseline_hazard_[0]) > 0
    assert len(model.baseline_hazard_[1]) > 0

def test_rule_survival_predict():
    """Test prediction of rule survival model"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    y = Survival(time, event)
    
    # Fit model
    model = RuleSurvival(max_rules=10, alpha=0.1)
    model.fit(X, y)
    
    # Generate test data
    X_test = np.random.randn(10, n_features)
    times = np.linspace(0, 2, 5)
    
    # Predict survival probabilities
    survival_probs = model.predict_survival(X_test, times)
    
    # Check predictions
    assert survival_probs.shape == (10, 5)
    assert np.all(survival_probs >= 0)
    assert np.all(survival_probs <= 1)
    assert np.all(np.diff(survival_probs, axis=1) <= 0)  # Survival should be non-increasing

def test_rule_importance():
    """Test computation of rule importance"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    y = Survival(time, event)
    
    # Fit model
    model = RuleSurvival(max_rules=10, alpha=0.1)
    model.fit(X, y)
    
    # Get rule importance
    importance = model.feature_importances_
    
    # Check importance
    assert importance is not None
    assert len(importance) == n_features
    assert np.all(importance >= 0)
    assert np.allclose(np.sum(importance), 1.0)

def test_parametric_survival_weibull():
    """Test Weibull parametric survival model"""
    # Generate sample data
    X = np.random.randn(100, 5)
    time = np.abs(np.random.randn(100))
    event = np.random.binomial(1, 0.5, 100)
    
    # Create and fit model
    model = RuleSurvival(model_type='weibull', max_rules=5)
    model.fit(X, Survival(time, event))
    
    # Test predictions
    times = np.linspace(0, 5, 10)
    survival = model.predict_survival(X, times)
    hazard = model.predict_hazard(X, times)
    
    assert survival.shape == (X.shape[0], len(times))
    assert hazard.shape == (X.shape[0], len(times))
    assert np.all(survival >= 0) and np.all(survival <= 1)
    assert np.all(hazard >= 0)
    assert 'scale' in model.parametric_params_
    assert 'shape' in model.parametric_params_

def test_parametric_survival_exponential():
    """Test exponential parametric survival model"""
    # Generate sample data
    X = np.random.randn(100, 5)
    time = np.abs(np.random.randn(100))
    event = np.random.binomial(1, 0.5, 100)
    
    # Create and fit model
    model = RuleSurvival(model_type='exponential', max_rules=5)
    model.fit(X, Survival(time, event))
    
    # Test predictions
    times = np.linspace(0, 5, 10)
    survival = model.predict_survival(X, times)
    hazard = model.predict_hazard(X, times)
    
    assert survival.shape == (X.shape[0], len(times))
    assert hazard.shape == (X.shape[0], len(times))
    assert np.all(survival >= 0) and np.all(survival <= 1)
    assert np.all(hazard >= 0)
    assert 'scale' in model.parametric_params_
    assert 'shape' not in model.parametric_params_

def test_survival_invalid_model_type():
    """Test invalid survival model type"""
    X = np.random.randn(100, 5)
    time = np.abs(np.random.randn(100))
    event = np.random.binomial(1, 0.5, 100)
    
    model = RuleSurvival(model_type='invalid')
    with pytest.raises(ValueError):
        model.fit(X, Survival(time, event)) 