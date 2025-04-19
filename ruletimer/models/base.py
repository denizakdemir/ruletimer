"""
Base classes for rule-based models
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

class BaseRuleEnsemble(BaseEstimator, ABC):
    """Base class for rule-based ensemble models"""
    
    def __init__(self):
        self.is_fitted_ = False
        self._rules_tuples = []
        self.rule_weights_ = None
        self.state_structure = None
        self._y = None
        self.baseline_hazards_ = {}
    
    @abstractmethod
    def _fit_weights(self, rule_values, y):
        """Fit weights for the rules"""
        pass
    
    @abstractmethod
    def _compute_feature_importances(self):
        """Compute feature importances"""
        pass
    
    @abstractmethod
    def _evaluate_rules(self, X):
        """Evaluate rules on input data"""
        pass
    
    def predict_cumulative_incidence(self, X, times, event_types):
        """Predict cumulative incidence for each event type"""
        raise NotImplementedError
    
    def predict_state_occupation(self, X, times):
        """Predict state occupation probabilities"""
        raise NotImplementedError
    
    @property
    def rules_(self):
        """Get the list of rules"""
        return self._rules_tuples
    
    def get_rules(self):
        """Get the list of rules"""
        return self._rules_tuples
    
    @property
    def feature_importances_(self):
        """Get feature importances"""
        if not hasattr(self, '_feature_importances'):
            self._compute_feature_importances()
        return self._feature_importances 