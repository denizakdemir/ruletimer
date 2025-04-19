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
from sklearn.tree import DecisionTreeClassifier
from ruletimer.models.base import BaseRuleEnsemble

class RuleMultiState(BaseMultiStateModel, BaseRuleEnsemble):
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
    min_support : float, optional (default=0.05)
        Minimum support threshold for rule pruning
    min_confidence : float, optional (default=0.5)
        Minimum confidence threshold for rule pruning
    max_impurity : float, optional (default=0.1)
        Maximum impurity reduction for rule pruning
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
        hazard_method: str = "nelson-aalen",
        min_support: float = 0.05,
        min_confidence: float = 0.5,
        max_impurity: float = 0.1
    ):
        if tree_type not in ['classification', 'regression']:
            raise ValueError("tree_type must be 'classification' or 'regression'")
        if tree_growing_strategy not in ['forest', 'single']:
            raise ValueError("tree_growing_strategy must be 'forest' or 'single'")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError("l1_ratio must be between 0 and 1")
            
        # Initialize with empty states and transitions if no state structure provided
        states = []
        transitions = []
        state_names = []
        if state_structure is not None:
            states = state_structure.states
            transitions = state_structure.transitions
            state_names = state_structure.state_names
            
        # Initialize both parent classes
        BaseMultiStateModel.__init__(
            self,
            states=state_names,  # Pass state names instead of states
            transitions=transitions,
            hazard_method=hazard_method
        )
        BaseRuleEnsemble.__init__(self)
        
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
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_impurity = max_impurity
        
        # Initialize rule-specific attributes
        self._rules_dict: Dict[Tuple[int, int], List[str]] = {}
        self.rule_coefficients_: Dict[Tuple[int, int], np.ndarray] = {}
        self.rule_importances_: Dict[Tuple[int, int], np.ndarray] = {}
        
    def _generate_rules(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """Generate rules using decision trees.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        List[str]
            List of generated rules
        """
        # Ensure we have enough samples of each class
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or np.min(counts) < self.min_samples_leaf:
            return []  # Not enough samples to generate meaningful rules
        
        if self.tree_growing_strategy == 'forest':
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                min_samples_split=2  # Ensure splits can occur
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
                random_state=self.random_state,
                min_samples_split=2  # Ensure splits can occur
            ).fit(X, y).estimators_[0]
            rules = self._extract_rules_from_tree(tree)
        
        # Remove duplicates and limit number
        unique_rules = list(dict.fromkeys(rules))
        return unique_rules[:self.max_rules]
    
    def _extract_rules_from_tree(self, tree):
        """Extract rules from a decision tree.

        Parameters
        ----------
        tree : DecisionTreeClassifier
            Fitted decision tree.

        Returns
        -------
        list
            List of extracted rules.
        """
        rules = []
        n_nodes = tree.tree_.node_count
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        value = tree.tree_.value
        
        def recurse(node, path):
            if feature[node] != -2:  # Internal node
                name = feature[node]
                threshold_val = threshold[node]
                
                # Left child
                left_path = path + [(name, '<=', threshold_val)]
                recurse(children_left[node], left_path)
                
                # Right child
                right_path = path + [(name, '>', threshold_val)]
                recurse(children_right[node], right_path)
            else:  # Leaf node
                # Only add rules for leaves with significant samples
                if np.sum(value[node]) >= self.min_samples_leaf:
                    # Convert path to rule string
                    rule_parts = []
                    for feat_idx, op, thresh in path:
                        rule_parts.append(f"X{feat_idx} {op} {thresh:.4f}")
                    rule = " and ".join(rule_parts)
                    rules.append(rule)
        
        recurse(0, [])  # Start from root node
        return rules
    
    def _evaluate_rules(self, X: np.ndarray, rules: Union[List[str], Dict[Tuple[int, int], List[str]]] = None) -> np.ndarray:
        """
        Evaluate rules on input data.

        Parameters
        ----------
        X : array-like
            Input data to evaluate rules on
        rules : list of str or dict, optional
            Rules to evaluate. If None, uses self._rules_dict

        Returns
        -------
        array-like
            Boolean matrix indicating which rules apply to each sample
        """
        if rules is None:
            rules = self._rules_dict
            
        if len(rules) == 0:
            return np.zeros((X.shape[0], 0))

        # Handle dictionary format
        if isinstance(rules, dict):
            # Flatten rules from all transitions
            all_rules = []
            for transition_rules in rules.values():
                all_rules.extend(transition_rules)
            rules = all_rules

        rule_matrix = np.zeros((X.shape[0], len(rules)))
        for i, rule in enumerate(rules):
            if not isinstance(rule, str):
                continue  # Skip non-string rules
                
            conditions = rule.split(' and ')
            rule_result = np.ones(X.shape[0], dtype=bool)
            
            for condition in conditions:
                # Extract feature index and value
                if '<=' in condition:
                    feature_str = condition.split('<=')[0].strip()
                    value = float(condition.split('<=')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] <= value
                elif '>=' in condition:
                    feature_str = condition.split('>=')[0].strip()
                    value = float(condition.split('>=')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] >= value
                elif '<' in condition:
                    feature_str = condition.split('<')[0].strip()
                    value = float(condition.split('<')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] < value
                elif '>' in condition:
                    feature_str = condition.split('>')[0].strip()
                    value = float(condition.split('>')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] > value
                elif '==' in condition:
                    feature_str = condition.split('==')[0].strip()
                    value = float(condition.split('==')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] == value
                elif '!=' in condition:
                    feature_str = condition.split('!=')[0].strip()
                    value = float(condition.split('!=')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] != value
                    
            rule_matrix[:, i] = rule_result
            
        return rule_matrix
    
    def fit(self, X, multi_state):
        """Fit the model to multi-state data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        multi_state : MultiState or dict
            Multi-state data object or dictionary with transition data
            in the format {(from_state, to_state): {'times': array, 'events': array}}

        Returns
        -------
        self : object
            Returns self.
        """
        # Initialize dictionaries for baseline hazards and rules
        transition_times = {}
        transition_events = {}
        self._rules_dict = {}
        self.rule_coefficients_ = {}
        self.rule_importances_ = {}
        
        # Handle different input formats
        if isinstance(multi_state, dict):
            # Process dictionary format
            for transition in self.state_structure.transitions:
                if transition not in multi_state:
                    continue
                    
                # Extract transition data
                times = multi_state[transition]['times']
                events = multi_state[transition]['events']
                
                # Store in dictionaries
                transition_times[transition] = times
                transition_events[transition] = events
                
                # Generate rules for current transition
                rules = self._generate_rules(X, events)
                
                # Store rules
                self._rules_dict[transition] = rules
                
                # Initialize elastic net for rule selection
                elastic_net = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    random_state=self.random_state
                )
                
                # Evaluate rules on the data
                rule_matrix = self._evaluate_rules(X, rules)
                
                # Fit elastic net to select important rules
                elastic_net.fit(rule_matrix, events)
                
                # Store rule coefficients and importances
                if len(rules) > 0:
                    self.rule_coefficients_[transition] = elastic_net.coef_
                    importances = np.abs(elastic_net.coef_)
                    self.rule_importances_[transition] = importances / np.sum(importances) if np.sum(importances) > 0 else importances
                else:
                    # If no rules, use a single constant feature
                    self.rule_coefficients_[transition] = np.array([1.0])
                    self.rule_importances_[transition] = np.array([1.0])
                    
                # Store transition models
                self.transition_models_[transition] = elastic_net
        else:
            # Original implementation for MultiState objects
            for transition in self.state_structure.transitions:
                # Get mask for current transition
                mask = (multi_state.start_state == transition[0])
                
                if not np.any(mask):
                    continue
                    
                # Get times and events for current transition
                times = multi_state.end_time[mask] - multi_state.start_time[mask]
                events = (multi_state.end_state[mask] == transition[1]).astype(int)
                
                # Store in dictionaries
                transition_times[transition] = times
                transition_events[transition] = events
                
                # Generate rules for current transition
                rules = self._generate_rules(X[mask], events)
                
                # Store rules
                self._rules_dict[transition] = rules
                
                # Initialize elastic net for rule selection
                elastic_net = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    random_state=self.random_state
                )
                
                # Evaluate rules on the data
                rule_matrix = self._evaluate_rules(X[mask], rules)
                
                # Fit elastic net to select important rules
                elastic_net.fit(rule_matrix, events)
                
                # Store rule coefficients and importances
                if len(rules) > 0:
                    self.rule_coefficients_[transition] = elastic_net.coef_
                    importances = np.abs(elastic_net.coef_)
                    self.rule_importances_[transition] = importances / np.sum(importances) if np.sum(importances) > 0 else importances
                else:
                    # If no rules, use a single constant feature
                    self.rule_coefficients_[transition] = np.array([1.0])
                    self.rule_importances_[transition] = np.array([1.0])
                    
                # Store transition models
                self.transition_models_[transition] = elastic_net
        
        # Estimate baseline hazards for all transitions
        self._estimate_baseline_hazards(
            transition_times,
            transition_events,
            None  # No weights for now
        )
        
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
            rules = self._rules_dict[transition]
            rule_matrix = self._evaluate_rules(X, rules)
            
            # Apply rule coefficients
            transition_risk = rule_matrix @ self.rule_coefficients_[transition]
            
            # Add to cumulative incidence
            cumulative_incidence += transition_risk.reshape(-1, 1) * np.ones((1, n_times))
        
        return np.clip(cumulative_incidence, 0, 1)
    
    def _prune_rules(self, rules: List[str], X: np.ndarray, y: np.ndarray) -> List[str]:
        """Prune rules using advanced techniques.
        
        This method implements several rule pruning strategies:
        1. Redundancy removal using rule coverage
        2. Statistical significance testing
        3. Rule interaction analysis
        4. Feature importance-based pruning
        
        Parameters
        ----------
        rules : List[str]
            List of rules to prune
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        List[str]
            Pruned list of rules
        """
        if not self.prune_rules:
            return rules
            
        # Evaluate all rules
        rule_matrix = self._evaluate_rules(X)
        
        # Calculate rule statistics
        rule_stats = []
        for i, rule in enumerate(rules):
            # Rule coverage
            coverage = np.mean(rule_matrix[:, i])
            
            # Rule accuracy
            rule_mask = rule_matrix[:, i]
            if np.sum(rule_mask) > 0:
                accuracy = np.mean(y[rule_mask] == np.argmax(np.bincount(y[rule_mask])))
            else:
                accuracy = 0
                
            # Rule significance (chi-square test)
            contingency = np.zeros((2, 2))
            contingency[0, 0] = np.sum((rule_mask) & (y == 0))
            contingency[0, 1] = np.sum((rule_mask) & (y == 1))
            contingency[1, 0] = np.sum((~rule_mask) & (y == 0))
            contingency[1, 1] = np.sum((~rule_mask) & (y == 1))
            
            from scipy.stats import chi2_contingency
            _, p_value, _, _ = chi2_contingency(contingency)
            
            # Feature importance
            features = set()
            for condition in rule.split(" AND "):
                feature_idx = int(condition.split("_")[1].split()[0])
                features.add(feature_idx)
            feature_importance = np.mean([self.rule_importances_[f] for f in features])
            
            rule_stats.append({
                'rule': rule,
                'coverage': coverage,
                'accuracy': accuracy,
                'p_value': p_value,
                'feature_importance': feature_importance
            })
        
        # Sort rules by quality metrics
        rule_stats.sort(key=lambda x: (
            x['accuracy'],
            x['coverage'],
            -x['p_value'],
            x['feature_importance']
        ), reverse=True)
        
        # Remove redundant rules
        pruned_rules = []
        covered_samples = np.zeros(X.shape[0], dtype=bool)
        
        for stat in rule_stats:
            rule_mask = self._evaluate_rules(X)
            new_coverage = np.sum(~covered_samples & rule_mask[:, i])
            
            # Keep rule if it adds significant new coverage
            if new_coverage / X.shape[0] >= self.min_support:
                pruned_rules.append(stat['rule'])
                covered_samples |= rule_mask[:, i]
        
        return pruned_rules
    
    def predict_transition_hazard(self, X, times, from_state, to_state):
        """Predict transition hazard for given samples and times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        times : array-like of shape (n_times,)
            Time points at which to evaluate the hazard.
        from_state : int
            Starting state.
        to_state : int
            Target state.

        Returns
        -------
        array-like of shape (n_samples, n_times)
            Predicted hazard values.
        """
        transition = (from_state, to_state)
        
        if transition not in self._rules_dict:
            raise ValueError(f"No model for transition {transition}")
        
        # Get rules and coefficients for this transition
        rules = self._rules_dict[transition]
        coefficients = self.rule_coefficients_[transition]
        
        # Evaluate rules on input data
        rule_matrix = self._evaluate_rules(X, rules)
        
        # Calculate linear predictor
        linear_pred = rule_matrix @ coefficients
        
        # Expand to match times dimension
        linear_pred = np.tile(linear_pred[:, np.newaxis], (1, len(times)))
        
        # Apply exponential function to get hazard
        hazard = np.exp(linear_pred)
        
        return hazard

    def _compute_feature_importances(self):
        """Compute feature importances"""
        # Combine importances from all transitions
        all_importances = []
        for transition in self.state_structure.transitions:
            if transition in self.rule_importances_:
                all_importances.extend(self.rule_importances_[transition])
        
        if all_importances:
            self._feature_importances = np.array(all_importances)
        else:
            self._feature_importances = np.array([])
            
    def _fit_weights(self, rule_values, y):
        """Fit weights for the rules"""
        # This is handled in the fit method using elastic net
        pass
        
    @property
    def rules_(self):
        """Get the rules dictionary"""
        return self._rules_dict 