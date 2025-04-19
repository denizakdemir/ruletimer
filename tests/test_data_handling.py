"""
Tests for data handling and preprocessing
"""

import numpy as np
import pandas as pd
import pytest
from ruletimer import Survival, CompetingRisks, MultiState
from ruletimer.data import DataConverter, DataValidator

def test_survival_data_validation():
    """Test validation of survival data"""
    # Valid data
    time = np.array([1, 2, 3])
    event = np.array([1, 0, 1])
    validator = DataValidator()
    assert validator.validate_survival(time, event)
    
    # Invalid data - different lengths
    with pytest.raises(ValueError):
        validator.validate_survival(np.array([1, 2]), np.array([1, 0, 1]))
    
    # Invalid data - negative times
    with pytest.raises(ValueError):
        validator.validate_survival(np.array([-1, 2, 3]), np.array([1, 0, 1]))
    
    # Invalid data - invalid event indicators
    with pytest.raises(ValueError):
        validator.validate_survival(np.array([1, 2, 3]), np.array([1, 2, 1]))
    
    # Test with DataFrame input
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'event': [1, 0, 1]
    })
    assert validator.validate_survival(df['time'], df['event'])

def test_competing_risks_data_validation():
    """Test validation of competing risks data"""
    # Valid data
    time = np.array([1, 2, 3])
    event = np.array([0, 1, 2])
    validator = DataValidator()
    assert validator.validate_competing_risks(time, event)
    
    # Invalid data - negative event types
    with pytest.raises(ValueError):
        validator.validate_competing_risks(time, np.array([-1, 1, 2]))
    
    # Test with DataFrame input
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'event': [0, 1, 2]
    })
    assert validator.validate_competing_risks(df['time'], df['event'])

def test_multi_state_data_validation():
    """Test validation of multi-state data"""
    # Valid data
    start_time = np.array([0, 0, 0])
    end_time = np.array([1, 2, 3])
    start_state = np.array([0, 0, 0])
    end_state = np.array([1, 2, 1])
    validator = DataValidator()
    assert validator.validate_multi_state(start_time, end_time, start_state, end_state)
    
    # Invalid data - end time before start time
    with pytest.raises(ValueError):
        validator.validate_multi_state(
            np.array([1, 2, 3]),
            np.array([0, 1, 2]),
            start_state,
            end_state
        )
    
    # Invalid data - invalid state transitions
    with pytest.raises(ValueError):
        validator.validate_multi_state(
            start_time,
            end_time,
            np.array([0, 1, 2]),
            np.array([2, 0, 1])
        )

def test_data_conversion():
    """Test conversion between different data formats"""
    converter = DataConverter()
    
    # Test survival data conversion
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'event': [1, 0, 1],
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [1, 2, 3]
    })
    
    X, y = converter.convert_survival(df, time_col='time', event_col='event')
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, Survival)
    assert X.shape == (3, 2)
    assert len(y.time) == 3
    
    # Test competing risks conversion
    df_cr = pd.DataFrame({
        'time': [1, 2, 3],
        'event': [0, 1, 2],
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [1, 2, 3]
    })
    
    X_cr, y_cr = converter.convert_competing_risks(df_cr, time_col='time', event_col='event')
    assert isinstance(X_cr, pd.DataFrame)
    assert isinstance(y_cr, CompetingRisks)
    assert X_cr.shape == (3, 2)
    assert len(y_cr.time) == 3
    
    # Test multi-state conversion
    df_ms = pd.DataFrame({
        'start_time': [0, 0, 0],
        'end_time': [1, 2, 3],
        'start_state': [0, 0, 0],
        'end_state': [1, 2, 1],
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [1, 2, 3]
    })
    
    X_ms, y_ms = converter.convert_multi_state(
        df_ms,
        start_time_col='start_time',
        end_time_col='end_time',
        start_state_col='start_state',
        end_state_col='end_state'
    )
    assert isinstance(X_ms, pd.DataFrame)
    assert isinstance(y_ms, MultiState)
    assert X_ms.shape == (3, 2)
    assert len(y_ms.start_time) == 3

def test_missing_value_handling():
    """Test handling of missing values"""
    # Create data with missing values
    df = pd.DataFrame({
        'time': [1, 2, 3, np.nan],
        'event': [1, 0, 1, 1],
        'feature1': [0.1, np.nan, 0.3, 0.4],
        'feature2': [1, 2, np.nan, 4]
    })
    
    converter = DataConverter()
    
    # Test different missing value strategies
    strategies = ['drop', 'mean', 'median', 'most_frequent']
    for strategy in strategies:
        X, y = converter.convert_survival(
            df,
            time_col='time',
            event_col='event',
            missing_value_strategy=strategy
        )
        
        if strategy == 'drop':
            assert X.shape[0] == 2  # Should drop rows with missing values
        else:
            assert X.shape[0] == 4  # Should impute missing values
            assert not X.isna().any().any()

def test_time_dependent_covariates():
    """Test handling of time-dependent covariates"""
    # Create data with time-dependent features
    n_samples = 100
    n_time_points = 10
    n_features = 5
    
    # Create time-dependent features
    time_dep_features = np.random.randn(n_samples, n_features, n_time_points)
    time_points = np.linspace(0, 5, n_time_points)
    
    # Create event times and indicators
    event_times = np.random.exponential(scale=1, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    
    converter = DataConverter()
    
    # Test conversion to time-dependent format
    X_td, y = converter.convert_time_dependent(
        time_dep_features,
        time_points,
        event_times,
        events
    )
    
    assert isinstance(X_td, dict)
    assert len(X_td) == n_samples
    for i in range(n_samples):
        assert X_td[i].shape[1] == n_features
        assert X_td[i].shape[0] <= n_time_points

def test_data_splitting():
    """Test train-test splitting for time-to-event data"""
    # Create sample data
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(scale=1, size=n_samples)
    event = np.random.binomial(1, 0.7, size=n_samples)
    
    converter = DataConverter()
    
    # Test different splitting strategies
    strategies = ['random', 'time', 'event']
    for strategy in strategies:
        X_train, X_test, y_train, y_test = converter.train_test_split(
            X,
            Survival(time, event),
            test_size=0.2,
            strategy=strategy
        )
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert len(y_train.time) == 80
        assert len(y_test.time) == 20
        
        if strategy == 'time':
            # For time-based splitting, test set should have later times
            assert np.min(y_test.time) >= np.max(y_train.time)
        elif strategy == 'event':
            # For event-based splitting, test set should have similar event rate
            assert abs(np.mean(y_test.event) - np.mean(y_train.event)) < 0.1

def test_feature_preprocessing():
    """Test feature preprocessing methods"""
    # Create sample data with different feature types
    df = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'categorical': ['A', 'B', 'A', 'C', 'B'],
        'binary': [0, 1, 0, 1, 0]
    })
    
    converter = DataConverter()
    
    # Test different preprocessing options
    X_processed = converter.preprocess_features(
        df,
        numeric_columns=['numeric1', 'numeric2'],
        categorical_columns=['categorical'],
        binary_columns=['binary'],
        scale_numeric=True,
        one_hot_encode=True
    )
    
    # Verify preprocessing
    assert X_processed.shape[1] > df.shape[1]  # Should have more columns due to one-hot encoding
    assert np.allclose(np.mean(X_processed, axis=0), 0, atol=1e-10)  # Should be centered
    assert np.allclose(np.std(X_processed, axis=0), 1, atol=1e-10)  # Should be scaled 