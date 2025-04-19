"""
Rule-based multi-state time-to-event model implementation.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from ruletimer.models.base_multi_state import BaseMultiStateModel
from ruletimer.utils import StateStructure
from ruletimer.utils.hazard_estimation import HazardEstimator
from ruletimer.state_manager import StateManager
from ruletimer.time_handler import TimeHandler

class RuleMultiState(BaseMultiStateModel):
    """
    Rule-based multi-state time-to-event model.
    
    This implementation extends the base multi-state model with rule-based feature
    extraction and selection capabilities. It uses decision trees to generate rules
    and then selects the most important rules using elastic net regularization.
    
    Parameters
    ----------
    max_rules : int, optional (default=100)
        Maximum number of rules to extract
    alpha : float, optional (default=0.1)
        Regularization strength for rule selection
    state_structure : StateStructure, optional
        Pre-defined state structure for the model
    max_depth : int, optional (default=3)
        Maximum depth of decision trees used for rule extraction
    min_samples_leaf : int, optional (default=10)
        Minimum number of samples required to be at a leaf node
    n_estimators : int, optional (default=100)
        Number of trees in the random forest
    tree_type : str, optional (default='classification')
        Type of trees to grow: 'classification' or 'regression'
    tree_growing_strategy : str, optional (default='forest')
        Strategy for growing trees: 'forest' or 'single'
    prune_rules : bool, optional (default=True)
        Whether to prune redundant rules
    l1_ratio : float, optional (default=0.5)
        Ratio of L1 to L2 regularization in elastic net
    random_state : int, optional
        Random seed for reproducibility
    hazard_method : str, optional (default="nelson-aalen")
        Method for hazard estimation: "nelson-aalen" or "parametric"
    """
    
    def __init__(
        self,
        max_rules: int = 100,
        alpha: float = 0.1,
        state_structure: StateStructure = None,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        n_estimators: int = 100,
        tree_type: str = 'classification',
        tree_growing_strategy: str = 'forest',
        prune_rules: bool = True,
        l1_ratio: float = 0.5,
        random_state: int = None,
        hazard_method: str = "nelson-aalen"
    ):
        if tree_type not in ['classification', 'regression']:
            raise ValueError("tree_type must be 'classification' or 'regression'")
        if tree_growing_strategy not in ['forest', 'single']:
            raise ValueError("tree_growing_strategy must be 'forest' or 'single'")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError("l1_ratio must be between 0 and 1")
            
        super().__init__(
            states=state_structure.states if state_structure else [],
            transitions=state_structure.transitions if state_structure else [],
            hazard_method=hazard_method
        )
        
        self.max_rules = max_rules
        self.alpha = alpha
        self.state_structure = state_structure
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.tree_type = tree_type
        self.tree_growing_strategy = tree_growing_strategy
        self.prune_rules = prune_rules
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        
        # Initialize rule-specific attributes
        self.rules_: Dict[Tuple[int, int], List[str]] = {}
        self.rule_importances_: Dict[Tuple[int, int], np.ndarray] = {}
        self.rule_coefficients_: Dict[Tuple[int, int], np.ndarray] = {}
        
    def _generate_rules(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """Generate rules using decision trees."""
        if self.tree_growing_strategy == 'forest':
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            forest.fit(X, y)
            rules = []
            for tree in forest.estimators_:
                rules.extend(self._extract_rules_from_tree(tree))
        else:
            # Single tree strategy
            tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            ).fit(X, y)
            rules = self._extract_rules_from_tree(tree.estimators_[0])
            
        return list(set(rules))[:self.max_rules]  # Remove duplicates and limit number
    
    def _extract_rules_from_tree(self, tree) -> List[str]:
        """Extract rules from a decision tree."""
        rules = []
        n_nodes = tree.tree_.node_count
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        
        def recurse(node, path):
            if children_left[node] != children_right[node]:  # Internal node
                name = f"feature_{feature[node]}"
                if children_left[node] != -1:
                    left_path = path + [f"{name} <= {threshold[node]:.3f}"]
                    recurse(children_left[node], left_path)
                if children_right[node] != -1:
                    right_path = path + [f"{name} > {threshold[node]:.3f}"]
                    recurse(children_right[node], right_path)
            else:  # Leaf node
                if path:  # Only add non-empty paths
                    rules.append(" AND ".join(path))
        
        recurse(0, [])
        return rules
    
    def _evaluate_rules(self, X: np.ndarray, rules: List[str]) -> np.ndarray:
        """Evaluate rules on the input data."""
        n_samples = X.shape[0]
        n_rules = len(rules)
        rule_matrix = np.zeros((n_samples, n_rules))
        
        for i, rule in enumerate(rules):
            conditions = rule.split(" AND ")
            mask = np.ones(n_samples, dtype=bool)
            for condition in conditions:
                feature_name, op, value = condition.split()
                feature_idx = int(feature_name.split("_")[1])
                value = float(value)
                if op == "<=":
                    mask &= (X[:, feature_idx] <= value)
                else:  # op == ">"
                    mask &= (X[:, feature_idx] > value)
            rule_matrix[:, i] = mask
        
        return rule_matrix
    
    def fit(self, X, multi_state):
        """Fit the model to the data."""
        # Initialize elastic net for rule selection
        elastic_net = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )
        
        # Process each transition separately
        for transition in self.state_structure.transitions:
            # Extract transition-specific data
            times = multi_state[transition]['times']
            events = multi_state[transition]['events']
            
            # Generate rules for this transition
            rules = self._generate_rules(X, events)
            self.rules_[transition] = rules
            
            # Evaluate rules on the data
            rule_matrix = self._evaluate_rules(X, rules)
            
            # Fit elastic net to select important rules
            elastic_net.fit(rule_matrix, events)
            
            # Store rule coefficients and importances
            self.rule_coefficients_[transition] = elastic_net.coef_
            importances = np.abs(elastic_net.coef_)
            self.rule_importances_[transition] = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
        self.is_fitted_ = True
        return self
    
    def get_feature_importances(self, transition: Tuple[int, int]) -> np.ndarray:
        """Get feature importances for a specific transition."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importances")
        
        if transition not in self.rule_importances_:
            raise ValueError(f"No feature importances available for transition {transition}")
        
        return self.rule_importances_[transition]
    
    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        target_state: Union[str, int]
    ) -> np.ndarray:
        """Predict cumulative incidence for a target state."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert target state to internal index if necessary
        if isinstance(target_state, str):
            target_state = self.state_manager.to_internal_index(target_state)
        
        # Get all transitions leading to the target state
        transitions_to_target = [
            t for t in self.state_structure.transitions
            if t[1] == target_state
        ]
        
        if not transitions_to_target:
            raise ValueError(f"No transitions found leading to state {target_state}")
        
        # Initialize cumulative incidence matrix
        n_samples = X.shape[0]
        n_times = len(times)
        cumulative_incidence = np.zeros((n_samples, n_times))
        
        # For each transition leading to the target state
        for transition in transitions_to_target:
            # Evaluate rules for this transition
            rules = self.rules_[transition]
            rule_matrix = self._evaluate_rules(X, rules)
            
            # Apply rule coefficients
            transition_risk = rule_matrix @ self.rule_coefficients_[transition]
            
            # Add to cumulative incidence
            cumulative_incidence += transition_risk.reshape(-1, 1) * np.ones((1, n_times))
        
        return np.clip(cumulative_incidence, 0, 1) 