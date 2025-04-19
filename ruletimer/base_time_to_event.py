from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional
from ruletimer.data import Survival, CompetingRisks, MultiState
from ruletimer.time_handler import TimeHandler

class BaseTimeToEventModel(ABC):
    """Abstract base class for all time-to-event models"""
    
    def __init__(self):
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, 'pd.DataFrame'], 
            y: Union[Survival, CompetingRisks, MultiState]) -> 'BaseTimeToEventModel':
        """
        Fit the model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Survival, CompetingRisks, or MultiState
            Target values
            
        Returns
        -------
        self : BaseTimeToEventModel
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict_survival(self, X: Union[np.ndarray, 'pd.DataFrame'], 
                        times: np.ndarray) -> np.ndarray:
        """
        Predict survival probabilities
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
        times : array-like
            Times at which to predict survival
            
        Returns
        -------
        survival : array-like of shape (n_samples, n_times)
            Predicted survival probabilities
        """
        pass
    
    @abstractmethod
    def predict_risk(self, X: Union[np.ndarray, 'pd.DataFrame']) -> np.ndarray:
        """
        Predict risk scores
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict for
            
        Returns
        -------
        risk : array-like of shape (n_samples,)
            Predicted risk scores
        """
        pass
    
    @property
    @abstractmethod
    def feature_importances_(self) -> np.ndarray:
        """
        Get feature importances
        
        Returns
        -------
        importances : array-like of shape (n_features,)
            Feature importances
        """
        pass
    
    def _validate_data(self, X: Union[np.ndarray, 'pd.DataFrame'],
                      y: Union[Survival, CompetingRisks, MultiState]) -> None:
        """
        Validate input data
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Survival, CompetingRisks, or MultiState
            Target values
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
            
        if not isinstance(y, (Survival, CompetingRisks, MultiState)):
            raise ValueError("y must be a Survival, CompetingRisks, or MultiState object")
            
        if len(y) != n_samples:
            raise ValueError("X and y must have the same number of samples")
    
    def _check_is_fitted(self) -> None:
        """
        Check if the model is fitted
        
        Raises
        ------
        ValueError
            If model is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call 'fit' before using this method.")
    
    def _validate_times(self, times: np.ndarray) -> np.ndarray:
        """
        Validate prediction times
        
        Parameters
        ----------
        times : array-like
            Times to validate
            
        Returns
        -------
        times : np.ndarray
            Validated times
            
        Raises
        ------
        ValueError
            If times are invalid
        """
        return TimeHandler.validate_times(times)
    
    def _get_time_points(self, data: object, n_points: Optional[int] = None) -> np.ndarray:
        """
        Get time points for prediction
        
        Parameters
        ----------
        data : object
            Data object with time attribute
        n_points : int, optional
            Number of time points to generate
            
        Returns
        -------
        times : np.ndarray
            Time points for prediction
        """
        return TimeHandler.get_time_points(data, n_points) 