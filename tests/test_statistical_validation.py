"""
Statistical validation tests for RuleTimeR models.

This module contains tests that verify the statistical properties and predictive
performance of RuleTimeR models, including:
- Monotonicity of survival curves
- Proper bounds for probabilities
- Feature-specific risk factors
- Model calibration
- Predictive performance
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd

from ruletimer.data import Survival, CompetingRisks, MultiState
from ruletimer.models import RuleSurvivalCox, RuleCompetingRisks, RuleMultiState
from ruletimer.utils import StateStructure

def test_survival_curve_monotonicity():
    """Test that survival curves are monotonically decreasing."""
    X = np.random.randn(100, 5)
    times = np.random.exponential(scale=5, size=100)
    events = np.random.binomial(1, 0.7, size=100)
    y = Survival(time=times, event=events)
    
    model = RuleSurvivalCox(max_rules=32, random_state=42)
    model.fit(X, y)
    
    # Predict at many time points
    test_times = np.linspace(0, 10, 100)
    survival_probs = model.predict_survival(X[:5], test_times)
    
    # Check monotonicity for each sample
    for i in range(survival_probs.shape[0]):
        diffs = np.diff(survival_probs[i])
        assert np.all(diffs <= 0 + 1e-10), f"Survival curve for sample {i} is not monotonically decreasing"

def test_survival_function_bounds():
    """Test that survival probabilities are between 0 and 1, start near 1, and approach proper limits."""
    # Create data with two clear risk groups
    X = np.zeros((100, 2))
    X[:50, 0] = 1.0  # High risk group
    
    # Generate survival times based on risk groups
    times = np.zeros(100)
    times[:50] = np.random.exponential(scale=2, size=50)  # High risk: shorter times
    times[50:] = np.random.exponential(scale=5, size=50)  # Low risk: longer times
    events = np.random.binomial(1, 0.8, size=100)
    
    y = Survival(time=times, event=events)
    model = RuleSurvivalCox(random_state=42)
    model.fit(X, y)
    
    test_times = np.linspace(0, 20, 100)
    survival_probs = model.predict_survival(X[[0, 50]], test_times)  # One from each group
    
    # Check bounds
    assert np.all(survival_probs >= 0), "Survival probabilities below 0"
    assert np.all(survival_probs <= 1), "Survival probabilities above 1"
    
    # Check initial values (should be close to 1)
    assert np.all(survival_probs[:, 0] > 0.95), "Survival probability at time 0 not near 1"
    
    # Check that high-risk group has lower survival than low-risk group
    assert np.mean(survival_probs[0]) < np.mean(survival_probs[1]), "High risk group doesn't have lower survival"
    
    # Check asymptotic behavior at long times
    last_time_idx = -10  # Look at the last few time points
    assert np.mean(survival_probs[0, last_time_idx:]) < 0.2, "Survival doesn't approach proper limit for high risk"

def test_competing_risks_cif_properties():
    """Test that cumulative incidence functions satisfy basic properties."""
    # Create data with two competing events
    X = np.random.randn(100, 3)
    times = np.random.exponential(scale=5, size=100)
    events = np.random.choice([0, 1, 2], size=100, p=[0.2, 0.4, 0.4])
    y = CompetingRisks(time=times, event=events)
    
    model = RuleCompetingRisks(random_state=42)
    model.fit(X, y)
    
    # Predict at many time points
    test_times = np.linspace(0, 10, 100)
    cifs = model.predict_cumulative_incidence(X[:5], test_times)
    
    # Convert dictionary to array for easier testing
    cif_array = np.stack([cifs[event_type] for event_type in model.event_types], axis=1)
    
    # Check that CIFs are monotonically increasing
    for i in range(cif_array.shape[0]):
        for j in range(cif_array.shape[1]):
            diffs = np.diff(cif_array[i, j])
            assert np.all(diffs >= -1e-10), f"CIF for sample {i}, event {j} is not monotonically increasing"
    
    # Check that CIFs are between 0 and 1
    assert np.all(cif_array >= 0), "CIF values below 0"
    assert np.all(cif_array <= 1), "CIF values above 1"
    
    # Check that sum of CIFs is less than or equal to 1
    assert np.all(np.sum(cif_array, axis=1) <= 1 + 1e-10), "Sum of CIFs exceeds 1"

def test_multistate_occupation_probabilities():
    """Test that state occupation probabilities sum to 1 and follow expected patterns."""
    # Define state structure for illness-death model
    states = [0, 1, 2]  # Healthy, Ill, Dead
    transitions = [(0, 1), (1, 2), (0, 2)]
    state_names = ["Healthy", "Ill", "Dead"]
    structure = StateStructure(states=states, transitions=transitions, state_names=state_names)
    
    # Generate synthetic data
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    
    # Create transition data
    transition_data = {}
    for transition in transitions:
        # Generate transition-specific hazard
        hazard = np.exp(0.2 * X[:, 0])
        times = np.random.exponential(1/hazard)
        events = np.random.binomial(1, 0.7, size=n_samples)
        transition_data[transition] = {
            'times': times,
            'events': events
        }
    
    # Fit model
    model = RuleMultiState(state_structure=structure, random_state=42)
    model.fit(X, transition_data)
    
    # Predict state occupation probabilities
    times = np.linspace(0, 10, 50)
    state_probs = model.predict_state_occupation(X[:5], times, initial_state=0)
    
    # Check that probabilities sum to 1 at all times
    total_probs = np.zeros((5, len(times)))
    for state in states:
        total_probs += state_probs[state]
    
    assert np.allclose(total_probs, 1.0, atol=1e-5), "State occupation probabilities don't sum to 1"
    
    # Check initial state probabilities
    assert np.all(state_probs[0][:, 0] > 0.95), "Initial state probability not close to 1"
    assert np.all(state_probs[1][:, 0] < 0.05), "Non-initial state probability not close to 0"
    assert np.all(state_probs[2][:, 0] < 0.05), "Non-initial state probability not close to 0"
    
    # Check absorbing state properties (Dead is absorbing)
    for i in range(5):
        # Dead state probability should be monotonically increasing
        diffs = np.diff(state_probs[2][i])
        assert np.all(diffs >= 0 - 1e-10), f"Absorbing state not monotonically increasing for sample {i}"

def test_model_calibration():
    """Test that model predictions are well-calibrated against observed outcomes."""
    # Create dataset with known risk groups
    n_samples = 500
    X = np.zeros((n_samples, 1))
    
    # Define 5 risk groups
    for i in range(5):
        X[i*100:(i+1)*100, 0] = i
    
    # Generate survival times based on risk groups
    times = np.zeros(n_samples)
    for i in range(5):
        scale = 5.0 / (i + 1)  # Higher risk groups have shorter survival times
        times[i*100:(i+1)*100] = np.random.exponential(scale=scale, size=100)
    
    # Add censoring
    censor_times = np.random.exponential(scale=10, size=n_samples)
    events = times <= censor_times
    times = np.minimum(times, censor_times)
    
    y = Survival(time=times, event=events)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit model
    model = RuleSurvivalCox(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate calibration
    # Use quantile-based approach: divide predictions into quantiles and compare observed vs predicted
    
    # Select a time point for calibration
    calib_time = 2.0
    
    # Predict survival at calibration time
    pred_surv = model.predict_survival(X_test, np.array([calib_time]))[:, 0]
    
    # Group predictions into quintiles
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    pred_groups = discretizer.fit_transform(pred_surv.reshape(-1, 1)).flatten().astype(int)
    
    # Calculate observed survival for each group
    observed_surv = np.zeros(5)
    
    for group in range(5):
        mask = pred_groups == group
        if np.sum(mask) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(y_test.time[mask], y_test.event[mask])
            observed_surv[group] = kmf.predict(calib_time)
    
    # Calculate average predicted survival for each group
    predicted_surv = np.zeros(5)
    for group in range(5):
        mask = pred_groups == group
        if np.sum(mask) > 0:
            predicted_surv[group] = np.mean(pred_surv[mask])
    
    # Calculate calibration error
    calib_error = np.mean(np.abs(observed_surv - predicted_surv))
    
    # A well-calibrated model should have error < 0.1
    assert calib_error < 0.1, f"Model not well-calibrated. Calibration error: {calib_error}"
    
    # Check monotonicity of observed survival across risk groups
    # Higher risk groups should have lower observed survival
    for i in range(4):
        if observed_surv[i] > 0 and observed_surv[i+1] > 0:
            assert observed_surv[i] >= observed_surv[i+1] - 0.1, "Observed survival not monotonic across risk groups"

def test_predictive_performance():
    """Test that model has good predictive performance compared to a simple reference model."""
    # Create dataset
    X = np.random.randn(300, 5)
    
    # Generate survival times with dependency on first two features
    baseline_hazard = 0.1
    feature_weights = np.array([0.5, 0.3, 0, 0, 0])
    hazard = baseline_hazard * np.exp(np.dot(X, feature_weights))
    times = np.random.exponential(scale=1/hazard)
    
    # Add censoring
    censor_times = np.random.exponential(scale=5, size=300)
    events = times <= censor_times
    times = np.minimum(times, censor_times)
    
    y = Survival(time=times, event=events)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit rule-based model
    rule_model = RuleSurvivalCox(random_state=42)
    rule_model.fit(X_train, y_train)
    
    # Fit reference Cox model using only the first feature
    df_train = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(5)])
    df_train['time'] = y_train.time
    df_train['event'] = y_train.event
    
    # Use only first feature for reference model
    cox_ref = CoxPHFitter()
    cox_ref.fit(df_train[['X0', 'time', 'event']], duration_col='time', event_col='event')
    
    # Calculate concordance index for both models
    # Rule model predictions
    rule_risk = rule_model.predict_risk(X_test)
    rule_cindex = concordance_index(y_test.time, -rule_risk, y_test.event)
    
    # Cox model predictions
    df_test = pd.DataFrame(X_test, columns=[f'X{i}' for i in range(5)])
    cox_risk = cox_ref.predict_partial_hazard(df_test[['X0']])
    cox_cindex = concordance_index(y_test.time, cox_risk, y_test.event)
    
    # Rule model should outperform simple Cox model
    assert rule_cindex > cox_cindex, f"Rule model ({rule_cindex:.3f}) not better than reference model ({cox_cindex:.3f})"
    
    # Check absolute performance: c-index should be reasonable (> 0.6)
    assert rule_cindex > 0.6, f"Rule model has poor discriminative ability: c-index = {rule_cindex:.3f}" 