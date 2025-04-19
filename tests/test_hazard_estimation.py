"""
Tests for the hazard estimation module.
"""
import numpy as np
import pytest
from ruletimer.utils.hazard_estimation import HazardEstimator

@pytest.fixture
def sample_data():
    """Create sample survival data for testing."""
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    events = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return times, events, weights

def test_estimate_baseline_hazard_nelson_aalen(sample_data):
    """Test Nelson-Aalen estimator for baseline hazard."""
    times, events, weights = sample_data
    
    # Estimate baseline hazard
    unique_times, baseline_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, weights, method="nelson-aalen"
    )
    
    # Check output types and shapes
    assert isinstance(unique_times, np.ndarray)
    assert isinstance(baseline_hazard, np.ndarray)
    assert len(unique_times) == len(baseline_hazard)
    
    # Check that unique_times contains only event times
    assert np.all(unique_times == np.array([1, 2, 4, 6, 7, 9, 10]))
    
    # Check hazard values are non-negative
    assert np.all(baseline_hazard >= 0)
    
    # Check hazard values are reasonable
    # At time 1: 1 event / 10 at risk
    assert np.isclose(baseline_hazard[0], 0.1)
    
    # At time 2: 1 event / 9 at risk
    assert np.isclose(baseline_hazard[1], 1/9)

def test_estimate_baseline_hazard_weights(sample_data):
    """Test weighted hazard estimation."""
    times, events, _ = sample_data
    weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Estimate baseline hazard with weights
    unique_times, baseline_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, weights, method="nelson-aalen"
    )
    
    # Check that weighted hazard is different from unweighted
    _, unweighted_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, None, method="nelson-aalen"
    )
    
    assert not np.allclose(baseline_hazard, unweighted_hazard)
    
    # Check first hazard value with weight 2.0
    assert np.isclose(baseline_hazard[0], 2.0/11.0)

def test_estimate_cumulative_hazard(sample_data):
    """Test cumulative hazard estimation."""
    times, events, weights = sample_data
    
    # Get baseline hazard
    unique_times, baseline_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, weights, method="nelson-aalen"
    )
    
    # Estimate cumulative hazard
    pred_times, cumulative_hazard = HazardEstimator.estimate_cumulative_hazard(
        unique_times, baseline_hazard
    )
    
    # Check output types and shapes
    assert isinstance(pred_times, np.ndarray)
    assert isinstance(cumulative_hazard, np.ndarray)
    assert len(pred_times) == len(cumulative_hazard)
    
    # Check cumulative hazard is non-decreasing
    assert np.all(np.diff(cumulative_hazard) >= 0)
    
    # Check cumulative hazard at first time point
    print(f"Cumulative hazard[0]: {cumulative_hazard[0]}")
    print(f"Baseline hazard[0]: {baseline_hazard[0]}")
    assert np.isclose(cumulative_hazard[0], baseline_hazard[0])
    
    # Check cumulative hazard at last time point
    assert np.isclose(cumulative_hazard[-1], np.sum(baseline_hazard))

def test_transform_hazard():
    """Test hazard transformations."""
    # Create sample cumulative hazard
    cumulative_hazard = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Test survival transformation
    survival = HazardEstimator.transform_hazard(cumulative_hazard, transform="exp")
    assert np.allclose(survival, np.exp(-cumulative_hazard))
    
    # Test CIF transformation
    cif = HazardEstimator.transform_hazard(cumulative_hazard, transform="cif")
    assert np.allclose(cif, 1 - np.exp(-cumulative_hazard))
    
    # Test invalid transformation
    with pytest.raises(ValueError):
        HazardEstimator.transform_hazard(cumulative_hazard, transform="invalid")

def test_estimate_baseline_hazard_invalid_method(sample_data):
    """Test invalid hazard estimation method."""
    times, events, weights = sample_data
    
    with pytest.raises(ValueError):
        HazardEstimator.estimate_baseline_hazard(
            times, events, weights, method="invalid"
        ) 