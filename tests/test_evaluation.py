"""
Tests for model evaluation and metrics
"""

import numpy as np
import pytest
from ruletimer import Survival, CompetingRisks, MultiState
from ruletimer.models import RuleSurvival, RuleCompetingRisks, RuleMultiState
from ruletimer.evaluation import ModelEvaluator

def test_survival_metrics():
    """Test evaluation metrics for survival models"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Split into train and test
    train_idx = np.random.choice(n_samples, size=800, replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = Survival(time[train_idx], event[train_idx])
    y_test = Survival(time[test_idx], event[test_idx])
    
    # Fit model
    model = RuleSurvival(max_rules=10)
    model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Test concordance index
    c_index = evaluator.concordance_index(model, X_test, y_test)
    assert 0 <= c_index <= 1
    
    # Test time-dependent AUC
    times = np.linspace(0, 5, 10)
    auc = evaluator.time_dependent_auc(model, X_test, y_test, times)
    assert auc.shape == (len(times),)
    assert np.all(0 <= auc) and np.all(auc <= 1)
    
    # Test Brier score
    brier = evaluator.brier_score(model, X_test, y_test, times)
    assert brier.shape == (len(times),)
    assert np.all(0 <= brier) and np.all(brier <= 1)
    
    # Test calibration
    calibration = evaluator.calibration(model, X_test, y_test, times)
    assert isinstance(calibration, dict)
    assert 'observed' in calibration and 'predicted' in calibration
    assert len(calibration['observed']) == len(calibration['predicted'])

def test_competing_risks_metrics():
    """Test evaluation metrics for competing risks models"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Split into train and test
    train_idx = np.random.choice(n_samples, size=800, replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = CompetingRisks(time[train_idx], event[train_idx])
    y_test = CompetingRisks(time[test_idx], event[test_idx])
    
    # Fit model
    model = RuleCompetingRisks(max_rules=10)
    model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Test cause-specific concordance
    c_index = evaluator.cause_specific_concordance(model, X_test, y_test)
    assert isinstance(c_index, dict)
    for event_type in [1, 2]:
        assert 0 <= c_index[event_type] <= 1
    
    # Test time-dependent AUC for each event type
    times = np.linspace(0, 5, 10)
    auc = evaluator.time_dependent_auc_competing_risks(model, X_test, y_test, times)
    assert isinstance(auc, dict)
    for event_type in [1, 2]:
        assert auc[event_type].shape == (len(times),)
        assert np.all(0 <= auc[event_type]) and np.all(auc[event_type] <= 1)
    
    # Test Brier score for each event type
    brier = evaluator.brier_score_competing_risks(model, X_test, y_test, times)
    assert isinstance(brier, dict)
    for event_type in [1, 2]:
        assert brier[event_type].shape == (len(times),)
        assert np.all(0 <= brier[event_type]) and np.all(brier[event_type] <= 1)

def test_multi_state_metrics():
    """Test evaluation metrics for multi-state models"""
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
    
    # Split into train and test
    train_idx = np.random.choice(n_samples, size=800, replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = MultiState(start_times[train_idx], end_times[train_idx],
                        start_states[train_idx], end_states[train_idx])
    y_test = MultiState(start_times[test_idx], end_times[test_idx],
                       start_states[test_idx], end_states[test_idx])
    
    # Fit model
    model = RuleMultiState(states=states, transitions=transitions, max_rules=10)
    model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Test transition-specific concordance
    c_index = evaluator.transition_concordance(model, X_test, y_test)
    assert isinstance(c_index, dict)
    for transition in transitions:
        assert 0 <= c_index[transition] <= 1
    
    # Test time-dependent AUC for each transition
    times = np.linspace(0, 5, 10)
    auc = evaluator.time_dependent_auc_multi_state(model, X_test, y_test, times)
    assert isinstance(auc, dict)
    for transition in transitions:
        assert auc[transition].shape == (len(times),)
        assert np.all(0 <= auc[transition]) and np.all(auc[transition] <= 1)
    
    # Test Brier score for each transition
    brier = evaluator.brier_score_multi_state(model, X_test, y_test, times)
    assert isinstance(brier, dict)
    for transition in transitions:
        assert brier[transition].shape == (len(times),)
        assert np.all(0 <= brier[transition]) and np.all(brier[transition] <= 1)

def test_model_comparison():
    """Test comparison of different models"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Split into train and test
    train_idx = np.random.choice(n_samples, size=800, replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = Survival(time[train_idx], event[train_idx])
    y_test = Survival(time[test_idx], event[test_idx])
    
    # Fit different models
    models = {
        'rule_survival': RuleSurvival(max_rules=10),
        'rule_survival_large': RuleSurvival(max_rules=20),
        'rule_survival_small': RuleSurvival(max_rules=5)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models using different metrics
    times = np.linspace(0, 5, 10)
    comparison = evaluator.compare_models(models, X_test, y_test, times)
    
    assert isinstance(comparison, dict)
    assert 'concordance' in comparison
    assert 'auc' in comparison
    assert 'brier' in comparison
    
    # Verify comparison results
    for metric in ['concordance', 'auc', 'brier']:
        assert isinstance(comparison[metric], dict)
        for model_name in models:
            assert model_name in comparison[metric]

def test_cross_validation():
    """Test cross-validation for model evaluation"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Test different cross-validation strategies
    cv_strategies = ['kfold', 'time', 'event']
    for strategy in cv_strategies:
        cv_results = evaluator.cross_validate(
            RuleSurvival(max_rules=10),
            X,
            Survival(time, event),
            cv=5,
            strategy=strategy
        )
        
        assert isinstance(cv_results, dict)
        assert 'test_concordance' in cv_results
        assert 'test_auc' in cv_results
        assert 'test_brier' in cv_results
        
        # Verify cross-validation results
        for metric in ['test_concordance', 'test_auc', 'test_brier']:
            assert len(cv_results[metric]) == 5  # 5 folds
            assert np.all(np.isfinite(cv_results[metric]))

def test_hyperparameter_tuning():
    """Test hyperparameter tuning with cross-validation"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Define parameter grid
    param_grid = {
        'max_rules': [5, 10, 20],
        'alpha': [0.01, 0.1, 1.0],
        'regularization': ['l1', 'l2']
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Perform grid search
    best_params, cv_results = evaluator.grid_search(
        RuleSurvival(),
        X,
        Survival(time, event),
        param_grid,
        cv=5
    )
    
    # Verify results
    assert isinstance(best_params, dict)
    assert all(param in best_params for param in param_grid)
    
    assert isinstance(cv_results, dict)
    assert 'mean_test_score' in cv_results
    assert 'std_test_score' in cv_results
    assert 'params' in cv_results 