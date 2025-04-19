"""
Tests for verifying mathematical relationships between probability functions
in survival analysis, competing risks, and multi-state models.
"""

import numpy as np
import pytest
from ruletimer import Survival, CompetingRisks, MultiState
from ruletimer.models import RuleSurvival, RuleCompetingRisks, RuleMultiState

def test_survival_hazard_relationship():
    """Test the relationship between survival and hazard functions"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Fit model
    model = RuleSurvival(max_rules=10)
    model.fit(X, Survival(time, event))
    
    # Get predictions
    times = np.linspace(0, 5, 100)
    survival = model.predict_survival(X, times)
    hazard = model.predict_hazard(X, times)
    cumulative_hazard = model.predict_cumulative_hazard(X, times)
    
    # Verify relationships
    # S(t) = exp(-H(t))
    assert np.allclose(survival, np.exp(-cumulative_hazard), rtol=1e-3)
    
    # h(t) = -d/dt log(S(t))
    log_survival = np.log(survival)
    hazard_approx = -np.diff(log_survival, axis=1) / np.diff(times)
    assert np.allclose(hazard[:, 1:], hazard_approx, rtol=1e-2)

def test_competing_risks_relationships():
    """Test relationships between CIF, cause-specific hazards, and survival"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Fit model
    model = RuleCompetingRisks(max_rules=10)
    model.fit(X, CompetingRisks(time, event))
    
    # Get predictions
    times = np.linspace(0, 5, 100)
    cif = model.predict_cumulative_incidence(X, times)
    cause_specific_hazard = model.predict_hazard(X, times)
    overall_survival = model.predict_survival(X, times)
    
    # Verify relationships
    # Overall survival = exp(-sum of cause-specific hazards)
    total_hazard = np.zeros_like(overall_survival)
    for event_type in cause_specific_hazard:
        total_hazard += cause_specific_hazard[event_type]
    cumulative_total_hazard = np.cumsum(total_hazard, axis=1) * np.diff(times)[0]
    assert np.allclose(overall_survival, np.exp(-cumulative_total_hazard), rtol=1e-3)
    
    # CIF(t) = integral_0^t S(u-) h_k(u) du
    for event_type in cif:
        cif_approx = np.zeros_like(cif[event_type])
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            cif_approx[:, i] = cif_approx[:, i-1] + overall_survival[:, i-1] * cause_specific_hazard[event_type][:, i] * dt
        assert np.allclose(cif[event_type], cif_approx, rtol=1e-2)

def test_multi_state_relationships():
    """Test relationships between transition probabilities and hazards in multi-state models"""
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
    start_states = np.zeros(n_samples)  # All start in Healthy state
    end_states = np.random.choice([1, 2], size=n_samples, p=[0.7, 0.3])
    
    # Fit model
    model = RuleMultiState(states=states, transitions=transitions, max_rules=10)
    model.fit(X, MultiState(start_times, end_times, start_states, end_states))
    
    # Get predictions
    times = np.linspace(0, 5, 100)
    transition_probs = model.predict_transition_probabilities(X, times)
    transition_hazards = model.predict_transition_hazards(X, times)
    
    # Verify relationships
    # P_ij(s,t) = exp(-integral_s^t sum_k h_ik(u) du) * h_ij(t)
    for (i, j) in transitions:
        # Get transition probability
        P_ij = transition_probs[(i, j)]
        
        # Calculate total hazard out of state i
        total_hazard = np.zeros_like(P_ij)
        for (k, l) in transitions:
            if k == i:
                total_hazard += transition_hazards[(k, l)]
        
        # Approximate integral
        integral = np.zeros_like(P_ij)
        for t_idx in range(1, len(times)):
            dt = times[t_idx] - times[t_idx-1]
            integral[:, t_idx] = integral[:, t_idx-1] + total_hazard[:, t_idx] * dt
        
        # Calculate approximate transition probability
        P_ij_approx = np.exp(-integral) * transition_hazards[(i, j)]
        
        assert np.allclose(P_ij, P_ij_approx, rtol=1e-2)

def test_unified_modeling_approach():
    """Test that the modeling approach is consistent across different model types"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # Test survival model
    time_surv = np.random.exponential(scale=1, size=n_samples)
    event_surv = np.random.binomial(1, 0.7, size=n_samples)
    model_surv = RuleSurvival(max_rules=10)
    model_surv.fit(X, Survival(time_surv, event_surv))
    
    # Test competing risks model
    time_cr = np.random.exponential(scale=1, size=n_samples)
    event_cr = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    model_cr = RuleCompetingRisks(max_rules=10)
    model_cr.fit(X, CompetingRisks(time_cr, event_cr))
    
    # Test multi-state model
    time_ms = np.random.exponential(scale=1, size=n_samples)
    start_state = np.zeros(n_samples)
    end_state = np.ones(n_samples)
    model_ms = RuleMultiState(states=["State0", "State1"], 
                            transitions=[(0, 1)], 
                            max_rules=10)
    model_ms.fit(X, MultiState(np.zeros(n_samples), time_ms, 
                             start_state, end_state))
    
    # Verify consistent rule generation
    assert hasattr(model_surv, 'rules_')
    assert hasattr(model_cr, 'rules_')
    assert hasattr(model_ms, 'rules_')
    
    # Verify consistent feature importance calculation
    assert hasattr(model_surv, 'feature_importances_')
    assert hasattr(model_cr, 'feature_importances_')
    assert hasattr(model_ms, 'feature_importances_')
    
    # Verify consistent prediction methods
    times = np.linspace(0, 5, 10)
    assert hasattr(model_surv, 'predict_survival')
    assert hasattr(model_cr, 'predict_survival')
    assert hasattr(model_ms, 'predict_survival')
    
    # Verify consistent hazard prediction
    assert hasattr(model_surv, 'predict_hazard')
    assert hasattr(model_cr, 'predict_hazard')
    assert hasattr(model_ms, 'predict_hazard')

def test_non_parametric_approach():
    """Test that the models use non-parametric approaches where appropriate"""
    # Test survival model
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    # Fit non-parametric model
    model = RuleSurvival(model_type='nonparametric', max_rules=10)
    model.fit(X, Survival(time, event))
    
    # Verify non-parametric baseline hazard
    assert hasattr(model, 'baseline_hazard_')
    assert isinstance(model.baseline_hazard_, tuple)
    assert len(model.baseline_hazard_) == 2
    
    # Verify no parametric assumptions
    assert not hasattr(model, 'parametric_params_')
    
    # Test competing risks model
    model_cr = RuleCompetingRisks(model_type='nonparametric', max_rules=10)
    model_cr.fit(X, CompetingRisks(time, event))
    
    # Verify non-parametric approach
    assert hasattr(model_cr, 'baseline_hazard_')
    assert isinstance(model_cr.baseline_hazard_, dict)
    
    # Test multi-state model
    model_ms = RuleMultiState(states=["State0", "State1"], 
                            transitions=[(0, 1)], 
                            max_rules=10)
    model_ms.fit(X, MultiState(np.zeros(n_samples), time, 
                             np.zeros(n_samples), np.ones(n_samples)))
    
    # Verify non-parametric approach
    assert hasattr(model_ms, 'baseline_hazard_')
    assert isinstance(model_ms.baseline_hazard_, dict) 