"""
Tests for rule generation and ensemble methods
"""

import numpy as np
import pytest
from ruletimer import Survival, CompetingRisks, MultiState
from ruletimer.models import RuleSurvival, RuleCompetingRisks, RuleMultiState

def test_rule_extraction():
    """Test rule extraction from decision trees"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Test different max_rules values
    for max_rules in [5, 10, 20]:
        model = RuleSurvival(max_rules=max_rules)
        model.fit(X, Survival(time, event))
        
        # Verify number of rules
        assert len(model.rules_) <= max_rules
        assert len(model.rules_) > 0
        
        # Verify rule format
        for rule in model.rules_:
            assert isinstance(rule, str)
            assert "if" in rule and "then" in rule
            
        # Verify rule activation
        rule_activation = model._get_rule_activation(X)
        assert rule_activation.shape == (n_samples, len(model.rules_))
        assert np.all(np.logical_or(rule_activation == 0, rule_activation == 1))

def test_rule_pruning():
    """Test rule pruning methods"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Test different pruning methods
    pruning_methods = ['importance', 'redundancy', 'complexity']
    for method in pruning_methods:
        model = RuleSurvival(max_rules=20, pruning_method=method)
        model.fit(X, Survival(time, event))
        
        # Verify pruning reduced number of rules
        assert len(model.rules_) <= 20
        
        # Verify rule importance
        importance = model.rule_importance()
        assert len(importance) == len(model.rules_)
        assert np.all(importance >= 0)
        
        # Verify no redundant rules
        if method == 'redundancy':
            rule_activation = model._get_rule_activation(X)
            correlations = np.corrcoef(rule_activation.T)
            np.fill_diagonal(correlations, 0)
            assert np.max(np.abs(correlations)) < 0.9

def test_rule_ensemble_training():
    """Test training of rule ensembles"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Test different regularization methods
    for regularization in ['l1', 'l2', 'elasticnet']:
        model = RuleSurvival(max_rules=10, regularization=regularization)
        model.fit(X, Survival(time, event))
        
        # Verify rule weights
        assert hasattr(model, 'rule_weights_')
        assert len(model.rule_weights_) == len(model.rules_)
        
        # Verify regularization effects
        if regularization == 'l1':
            assert np.sum(model.rule_weights_ != 0) <= len(model.rules_)
        elif regularization == 'l2':
            assert np.all(np.isfinite(model.rule_weights_))
        else:  # elasticnet
            assert np.all(np.isfinite(model.rule_weights_))
            assert np.sum(model.rule_weights_ != 0) <= len(model.rules_)

def test_rule_importance():
    """Test rule importance calculation"""
    # Generate synthetic data with known feature importance
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # Create outcome dependent on first two features
    time = np.exp(2 * X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n_samples))
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Fit model
    model = RuleSurvival(max_rules=20)
    model.fit(X, Survival(time, event))
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Verify importance reflects true relationship
    assert importance[0] > importance[2:]  # First feature should be most important
    assert importance[1] > importance[2:]  # Second feature should be second most important

def test_rule_interactions():
    """Test capturing of feature interactions in rules"""
    # Generate synthetic data with interaction
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # Create outcome with interaction between first two features
    interaction = X[:, 0] * X[:, 1]
    time = np.exp(interaction + np.random.normal(0, 0.1, n_samples))
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Fit model with interaction detection
    model = RuleSurvival(max_rules=20, detect_interactions=True)
    model.fit(X, Survival(time, event))
    
    # Verify interaction rules
    interaction_rules = [rule for rule in model.rules_ 
                        if 'feature_0' in rule and 'feature_1' in rule]
    assert len(interaction_rules) > 0
    
    # Verify interaction importance
    importance = model.rule_importance()
    interaction_importance = sum(imp for rule, imp in zip(model.rules_, importance)
                               if 'feature_0' in rule and 'feature_1' in rule)
    assert interaction_importance > 0

def test_competing_risks_rule_generation():
    """Test rule generation for competing risks"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Test different event-specific rule sets
    model = RuleCompetingRisks(max_rules=10, event_specific_rules=True)
    model.fit(X, CompetingRisks(time, event))
    
    # Verify event-specific rules
    assert hasattr(model, 'event_rules_')
    assert isinstance(model.event_rules_, dict)
    for event_type in [1, 2]:
        assert event_type in model.event_rules_
        assert len(model.event_rules_[event_type]) > 0

def test_multi_state_rule_generation():
    """Test rule generation for multi-state models"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # Define states and transitions
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    
    # Generate transition times and states
    start_times = np.zeros(n_samples)
    end_times = np.random.exponential(scale=1, size=n_samples)
    start_states = np.zeros(n_samples)
    end_states = np.random.choice([1, 2], size=n_samples, p=[0.7, 0.3])
    
    # Test transition-specific rule sets
    model = RuleMultiState(states=states, transitions=transitions,
                          max_rules=10, transition_specific_rules=True)
    model.fit(X, MultiState(start_times, end_times, start_states, end_states))
    
    # Verify transition-specific rules
    assert hasattr(model, 'transition_rules_')
    assert isinstance(model.transition_rules_, dict)
    for transition in transitions:
        assert transition in model.transition_rules_
        assert len(model.transition_rules_[transition]) > 0 