"""
Time and prediction point management utilities for time-to-event models.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from ruletimer.data import MultiState

def create_prediction_grid(
    data,
    n_points: int = 100,
    max_time: Optional[float] = None,
    quantile_max: float = 0.95
) -> np.ndarray:
    """Create a standard time grid for predictions based on observed data.
    
    Parameters
    ----------
    data : MultiState, Survival, or CompetingRisks
        Time-to-event data
    n_points : int, default=100
        Number of time points
    max_time : float, optional
        Maximum time value (if None, uses quantile_max of observed times)
    quantile_max : float, default=0.95
        Quantile of observed times to use as maximum if max_time is None
        
    Returns
    -------
    np.ndarray
        Array of time points for prediction
    """
    if hasattr(data, 'time'):
        times = data.time
    elif hasattr(data, 'end_time'):
        times = data.end_time
    else:
        raise ValueError("Data object must have time or end_time attribute")
        
    if max_time is None:
        max_time = np.quantile(times, quantile_max)
        
    return np.linspace(0, max_time, n_points)

def to_multi_state_format(
    data,
    data_type: str = 'survival'
) -> MultiState:
    """Convert any time-to-event data to multi-state format.
    
    Parameters
    ----------
    data : Survival, CompetingRisks, or related data
        Time-to-event data
    data_type : str, default='survival'
        Type of data ('survival', 'competing_risks')
        
    Returns
    -------
    MultiState
        Data converted to multi-state format
    """
    if data_type == 'survival':
        # Convert survival data to multi-state (state 1 → state 2)
        patient_id = np.arange(len(data.time))
        start_time = np.zeros_like(data.time)
        end_time = data.time
        start_state = np.ones_like(data.time)
        end_state = np.where(data.event == 1, 2, 0)  # 2=event, 0=censored
        
        return MultiState(
            start_time=start_time,
            end_time=end_time,
            start_state=start_state,
            end_state=end_state,
            patient_id=patient_id
        )
        
    elif data_type == 'competing_risks':
        # Convert competing risks to multi-state (state 1 → state 2, 3, etc.)
        patient_id = np.arange(len(data.time))
        start_time = np.zeros_like(data.time)
        end_time = data.time
        start_state = np.ones_like(data.time)
        end_state = np.where(data.event == 0, 0, data.event + 1)  # 0=censored, 2,3,...=events
        
        return MultiState(
            start_time=start_time,
            end_time=end_time,
            start_state=start_state,
            end_state=end_state,
            patient_id=patient_id
        )
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

def bootstrap_confidence_intervals(
    model,
    X: np.ndarray,
    y,
    times: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    prediction_method: str = 'predict_state_occupation'
) -> Dict:
    """Calculate bootstrap confidence intervals for predictions.
    
    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted model
    X : array-like
        Feature matrix
    y : MultiState, Survival, or CompetingRisks
        Time-to-event data
    times : array-like
        Time points for prediction
    n_bootstrap : int, default=100
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals
    prediction_method : str, default='predict_state_occupation'
        Method to use for prediction ('predict_state_occupation', 
        'predict_transition_probability', 'predict_survival')
        
    Returns
    -------
    dict
        Dictionary containing mean predictions and confidence intervals
    """
    # Ensure we have the right method
    pred_method = getattr(model, prediction_method)
    
    # Get base prediction for reference
    base_pred = pred_method(X, times)
    
    # Initialize bootstrap results
    if isinstance(base_pred, dict):
        # Multi-state predictions
        bootstrap_results = {
            k: np.zeros((n_bootstrap, X.shape[0], len(times)))
            for k in base_pred.keys()
        }
    else:
        # Single outcome predictions
        bootstrap_results = np.zeros((n_bootstrap, X.shape[0], len(times)))
    
    # Perform bootstrap
    n_samples = X.shape[0]
    for i in range(n_bootstrap):
        # Create bootstrap sample
        sample_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[sample_idx]
        
        # Handle different data types
        if hasattr(y, 'time'):
            y_boot = type(y)(y.time[sample_idx], y.event[sample_idx])
        else:
            # For MultiState, need to filter transitions by patient
            patient_idx = np.unique(y.patient_id[sample_idx])
            mask = np.isin(y.patient_id, patient_idx)
            y_boot = type(y)(
                start_time=y.start_time[mask],
                end_time=y.end_time[mask],
                start_state=y.start_state[mask],
                end_state=y.end_state[mask],
                patient_id=y.patient_id[mask]
            )
            
        # Fit model on bootstrap sample
        model_boot = model.__class__(**model.get_params())
        model_boot.fit(X_boot, y_boot)
        
        # Make predictions
        pred_boot = pred_method(X_boot, times)
        
        # Store results
        if isinstance(pred_boot, dict):
            for k in pred_boot.keys():
                bootstrap_results[k][i] = pred_boot[k]
        else:
            bootstrap_results[i] = pred_boot
    
    # Calculate confidence intervals
    low_quantile = alpha / 2
    high_quantile = 1 - low_quantile
    
    if isinstance(base_pred, dict):
        # For dictionary results
        result = {}
        for k in base_pred.keys():
            result[k] = {
                'mean': np.mean(bootstrap_results[k], axis=0),
                'lower': np.quantile(bootstrap_results[k], low_quantile, axis=0),
                'upper': np.quantile(bootstrap_results[k], high_quantile, axis=0)
            }
    else:
        # For array results
        result = {
            'mean': np.mean(bootstrap_results, axis=0),
            'lower': np.quantile(bootstrap_results, low_quantile, axis=0),
            'upper': np.quantile(bootstrap_results, high_quantile, axis=0)
        }
        
    return result 