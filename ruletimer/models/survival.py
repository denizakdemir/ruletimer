"""
Specialized survival model implemented as a two-state model.
"""
from typing import Optional, Union, List, Tuple
import numpy as np
from sklearn.linear_model import ElasticNet
from ruletimer.models.base_multi_state import BaseMultiStateModel
from ruletimer.data import Survival

class RuleSurvival(BaseMultiStateModel):
    """
    Survival model implemented as a special case of multi-state model.
    
    This model represents standard survival analysis as a two-state model:
    Alive (0) -> Dead (1)
    """
    
    def __init__(self, hazard_method: str = "nelson-aalen"):
        """
        Initialize survival model.
        
        Parameters
        ----------
        hazard_method : str
            Method for hazard estimation: "nelson-aalen" or "parametric"
        """
        # Initialize with two states and one transition
        super().__init__(
            states=["Alive", "Dead"],
            transitions=[(0, 1)],
            hazard_method=hazard_method
        )
    
    def fit(self, X: np.ndarray, y: Survival) -> 'RuleSurvival':
        """
        Fit the survival model.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : Survival
            Target survival data
            
        Returns
        -------
        self : RuleSurvival
            Fitted model
        """
        # Validate input data
        if not isinstance(y, Survival):
            raise ValueError("y must be a Survival object")
            
        # Prepare transition-specific data
        transition = (0, 1)  # Alive -> Dead
        transition_times = {transition: y.time}
        transition_events = {transition: y.event}
        
        # Estimate baseline hazards
        self._estimate_baseline_hazards(transition_times, transition_events)
        
        # Fit transition-specific model (to be implemented by subclass)
        self._fit_transition_model(X, y, transition)
        
        self.is_fitted_ = True
        return self
    
    def predict_survival(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict survival probabilities.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like, optional
            Times at which to predict survival
            
        Returns
        -------
        np.ndarray
            Predicted survival probabilities
        """
        if times is None:
            times = self.baseline_hazards_[(0, 1)][0]
            
        # Survival probability is 1 - transition probability
        return 1 - self.predict_transition_probability(X, times, 0, 1)
    
    def predict_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict hazard function.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like, optional
            Times at which to predict hazard
            
        Returns
        -------
        np.ndarray
            Predicted hazard values
        """
        if times is None:
            times = self.baseline_hazards_[(0, 1)][0]
            
        return self.predict_transition_hazard(X, times, 0, 1)
    
    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
        from_state: Optional[Union[str, int]] = None,
        to_state: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """
        Predict cumulative hazard function.
        
        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like, optional
            Times at which to predict cumulative hazard
        from_state : str or int, optional
            Starting state (ignored in survival analysis)
        to_state : str or int, optional
            Target state (ignored in survival analysis)
            
        Returns
        -------
        np.ndarray
            Predicted cumulative hazard values
        """
        if times is None:
            times = self.baseline_hazards_[(0, 1)][0]
            
        # In survival analysis, we only have one transition (0 -> 1)
        return super().predict_cumulative_hazard(X, times, 0, 1)
    
    def _fit_transition_model(
        self,
        X: np.ndarray,
        y: Survival,
        transition: tuple
    ) -> None:
        """
        Fit transition-specific model using random forest for rule generation
        and elastic net for fitting.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : Survival
            Target survival data
        transition : tuple
            Transition to fit model for
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Initialize random forest
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Fit forest on survival times
        forest.fit(X, y.time)
        
        # Extract rules from forest
        self.rules_ = self._extract_rules_from_forest(forest)
        
        # Evaluate rules on data
        rule_matrix = self._evaluate_rules(X)
        
        # Fit elastic net on rule matrix
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )
        model.fit(rule_matrix, y.time)
        
        # Store rule weights and model
        self.rule_weights_ = model.coef_
        self.transition_models_[transition] = model
        
        # Compute feature importances
        self.feature_importances_ = self._compute_feature_importances()
        
    def _extract_rules_from_forest(self, forest):
        """Extract rules from random forest."""
        rules = []
        for tree in forest.estimators_:
            rules.extend(self._extract_rules_from_tree(tree.tree_))
        return rules[:self.max_rules]
    
    def _extract_rules_from_tree(self, tree):
        """Extract rules from a single tree."""
        rules = []
        
        def recurse(node, rule):
            if tree.feature[node] != -2:  # Not a leaf
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                
                # Left branch: feature <= threshold
                left_rule = rule + [(feature, "<=", threshold)]
                recurse(tree.children_left[node], left_rule)
                
                # Right branch: feature > threshold
                right_rule = rule + [(feature, ">", threshold)]
                recurse(tree.children_right[node], right_rule)
            else:
                rules.append(rule)
        
        recurse(0, [])
        return rules
    
    def _evaluate_rules(self, X):
        """Evaluate rules on data."""
        rule_matrix = np.zeros((X.shape[0], len(self.rules_)))
        for i, rule in enumerate(self.rules_):
            mask = np.ones(X.shape[0], dtype=bool)
            for feature, op, threshold in rule:
                if op == "<=":
                    mask &= (X[:, feature] <= threshold)
                else:  # op == ">"
                    mask &= (X[:, feature] > threshold)
            rule_matrix[:, i] = mask
        return rule_matrix
    
    def _compute_feature_importances(self):
        """Compute feature importances from rules."""
        # Get number of features from the first rule's first condition's feature index
        n_features = max(feature for rule in self.rules_ for feature, _, _ in rule) + 1 if self.rules_ else 0
        importances = np.zeros(n_features)
        
        for rule, weight in zip(self.rules_, self.rule_weights_):
            for feature, _, _ in rule:
                importances[feature] += abs(weight)
                
        # Normalize importances
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
            
        return importances 

class RuleSurvivalCox(RuleSurvival):
    """
    Rule-based Cox model for survival analysis.
    """
    
    def __init__(
        self,
        max_rules: int = 32,
        max_depth: int = 4,
        n_estimators: int = 200,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        hazard_method: str = "nelson-aalen",
        random_state: Optional[int] = None
    ):
        """
        Initialize RuleSurvivalCox model.
        
        Parameters
        ----------
        max_rules : int
            Maximum number of rules to generate
        max_depth : int
            Maximum depth of trees
        n_estimators : int
            Number of trees in random forest
        alpha : float
            L1 + L2 regularization strength
        l1_ratio : float
            L1 ratio for elastic net (1 = lasso, 0 = ridge)
        hazard_method : str
            Method for hazard estimation
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(hazard_method=hazard_method)
        self.max_rules = max_rules
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        
    def predict_transition_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict transition-specific hazard.
        
        Parameters
        ----------
        X : array-like
            Covariate values
        times : array-like
            Times at which to predict hazard
        from_state : str or int
            Starting state
        to_state : str or int
            Target state
            
        Returns
        -------
        np.ndarray
            Predicted hazard values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Convert states to internal indices
        from_idx = self.state_manager.to_internal_index(from_state)
        to_idx = self.state_manager.to_internal_index(to_state)
        transition = (from_idx, to_idx)
        
        if transition not in self.transition_models_:
            raise ValueError(f"No model for transition {transition}")
            
        # Evaluate rules on input data
        rule_matrix = self._evaluate_rules(X)
        
        # Get relative hazard from model
        relative_hazard = self.transition_models_[transition].predict(rule_matrix)
        
        # Get baseline hazard for each time point
        baseline_hazard = np.zeros(len(times))
        baseline_times, baseline_values = self.baseline_hazards_[transition]
        
        for i, t in enumerate(times):
            # Find the closest baseline time
            idx = np.searchsorted(baseline_times, t)
            if idx == 0:
                baseline_hazard[i] = baseline_values[0]
            elif idx == len(baseline_times):
                baseline_hazard[i] = baseline_values[-1]
            else:
                # Linear interpolation
                t0, t1 = baseline_times[idx-1:idx+1]
                h0, h1 = baseline_values[idx-1:idx+1]
                baseline_hazard[i] = h0 + (h1 - h0) * (t - t0) / (t1 - t0)
                
        # Compute hazard for each sample and time point
        hazard = np.exp(relative_hazard[:, np.newaxis]) * baseline_hazard[np.newaxis, :]
        return hazard
        
    def _fit_transition_model(
        self,
        X: np.ndarray,
        y: Survival,
        transition: tuple
    ) -> None:
        """
        Fit transition-specific model using random forest for rule generation
        and elastic net for fitting.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : Survival
            Target survival data
        transition : tuple
            Transition to fit model for
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Initialize random forest
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Fit forest on survival times
        forest.fit(X, y.time)
        
        # Extract rules from forest
        self.rules_ = self._extract_rules_from_forest(forest)
        
        # Evaluate rules on data
        rule_matrix = self._evaluate_rules(X)
        
        # Fit elastic net on rule matrix
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )
        model.fit(rule_matrix, y.time)
        
        # Store rule weights and model
        self.rule_weights_ = model.coef_
        self.transition_models_[transition] = model
        
        # Compute feature importances
        self.feature_importances_ = self._compute_feature_importances()
        
    def _extract_rules_from_forest(self, forest):
        """Extract rules from random forest."""
        rules = []
        for tree in forest.estimators_:
            rules.extend(self._extract_rules_from_tree(tree.tree_))
        return rules[:self.max_rules]
        
    def _extract_rules_from_tree(self, tree):
        """Extract rules from a single tree."""
        rules = []
        
        def recurse(node, rule):
            if tree.feature[node] != -2:  # Not a leaf
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                
                # Left branch: feature <= threshold
                left_rule = rule + [(feature, "<=", threshold)]
                recurse(tree.children_left[node], left_rule)
                
                # Right branch: feature > threshold
                right_rule = rule + [(feature, ">", threshold)]
                recurse(tree.children_right[node], right_rule)
            else:
                rules.append(rule)
        
        recurse(0, [])
        return rules
        
    def _evaluate_rules(self, X):
        """Evaluate rules on data."""
        rule_matrix = np.zeros((X.shape[0], len(self.rules_)))
        for i, rule in enumerate(self.rules_):
            mask = np.ones(X.shape[0], dtype=bool)
            for feature, op, threshold in rule:
                if op == "<=":
                    mask &= (X[:, feature] <= threshold)
                else:  # op == ">"
                    mask &= (X[:, feature] > threshold)
            rule_matrix[:, i] = mask
        return rule_matrix
        
    def _compute_feature_importances(self):
        """Compute feature importances from rules."""
        # Get number of features from the first rule's first condition's feature index
        n_features = max(feature for rule in self.rules_ for feature, _, _ in rule) + 1 if self.rules_ else 0
        importances = np.zeros(n_features)
        
        for rule, weight in zip(self.rules_, self.rule_weights_):
            for feature, _, _ in rule:
                importances[feature] += abs(weight)
                
        # Normalize importances
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
            
        return importances 