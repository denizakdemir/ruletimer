"""Tests for the base rule ensemble model"""

import numpy as np
import pandas as pd
import pytest
from ruletimer.models.base import BaseRuleEnsemble
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

class SimpleRuleEnsemble(BaseRuleEnsemble):
    """Simple implementation of BaseRuleEnsemble for testing"""
    
    def _extract_rules(self, X):
        """Simple rule extraction for testing"""
        # Create a simple rule for testing
        if isinstance(X, pd.DataFrame):
            n_features = len(X.columns)
        else:
            n_features = X.shape[1]
        
        # Create one simple rule per feature
        rules = []
        for i in range(min(n_features, self.max_rules)):
            rules.append([(i, "<=", 0.5)])
        return rules
    
    def _get_tree_model(self):
        """Get tree model for testing"""
        # Test regression tree
        if self.tree_type == 'regression':
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        
        # Test classification tree
        elif self.tree_type == 'classification':
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        
        # Test random forest regression
        elif self.tree_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        
        # Test invalid tree type
        else:
            raise ValueError(f"Unsupported tree_type: {self.tree_type}")
    
    def _fit_weights(self, rule_values, y):
        """Simple weight fitting for testing"""
        self.rule_weights_ = np.ones(rule_values.shape[1])
    
    def _compute_feature_importances(self):
        """Simple feature importance computation for testing"""
        self.feature_importances_ = np.ones(self._X.shape[1])

def test_base_rule_ensemble_initialization():
    """Test initialization of BaseRuleEnsemble"""
    model = SimpleRuleEnsemble(max_rules=50, alpha=0.1, l1_ratio=0.7, random_state=42)
    assert model.max_rules == 50
    assert model.alpha == 0.1
    assert model.l1_ratio == 0.7
    assert model.random_state == 42
    assert model.rules_ is None
    assert model.rule_weights_ is None
    assert model.feature_importances_ is None

def test_base_rule_ensemble_with_numpy():
    """Test BaseRuleEnsemble with numpy arrays"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Create time attribute for y (needed for rule extraction)
    y = np.rec.fromarrays([y], names=['time'])
    
    # Fit model
    model = SimpleRuleEnsemble(max_rules=10, random_state=42)
    model.fit(X, y)
    
    # Check attributes
    assert len(model.rules_) <= 10
    assert len(model.rule_weights_) == len(model.rules_)
    assert len(model.feature_importances_) == X.shape[1]
    
    # Test rule evaluation
    rule_values = model._evaluate_rules(X)
    assert rule_values.shape == (X.shape[0], len(model.rules_))
    assert np.all((rule_values == 0) | (rule_values == 1))
    
    # Test rule and weight getters
    assert model.get_rules() == model.rules_
    assert np.array_equal(model.get_rule_weights(), model.rule_weights_)

def test_base_rule_ensemble_with_pandas():
    """Test BaseRuleEnsemble with pandas DataFrames"""
    # Generate synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = np.random.randn(100)
    
    # Create time attribute for y (needed for rule extraction)
    y = np.rec.fromarrays([y], names=['time'])
    
    # Fit model
    model = SimpleRuleEnsemble(max_rules=10, random_state=42)
    model.fit(X, y)
    
    # Check attributes
    assert len(model.rules_) <= 10
    assert len(model.rule_weights_) == len(model.rules_)
    assert len(model.feature_importances_) == X.shape[1]
    
    # Test rule evaluation with DataFrame
    rule_values = model._evaluate_rules(X)
    assert rule_values.shape == (X.shape[0], len(model.rules_))
    assert np.all((rule_values == 0) | (rule_values == 1))

def test_base_rule_ensemble_errors():
    """Test error handling in BaseRuleEnsemble"""
    model = BaseRuleEnsemble()

    with pytest.raises(NotImplementedError):
        X = np.random.randn(10, 5)
        y = np.rec.fromarrays([np.random.randn(10)], names=['time'])
        model._fit_weights(X, y)

    with pytest.raises(NotImplementedError):
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        model.fit(X, y)  # This should raise NotImplementedError from _fit_weights

    with pytest.raises(NotImplementedError):
        model._compute_feature_importances()

def test_get_tree_model():
    """Test _get_tree_model method with different configurations"""
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    y = np.rec.fromarrays([y], names=['time'])
    
    # Test regression tree with all parameters
    model = SimpleRuleEnsemble(
        tree_type='regression',
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )
    model._X = X
    model._y = y
    tree = model._get_tree_model()
    assert isinstance(tree, DecisionTreeRegressor)
    assert tree.max_depth == 4
    assert tree.min_samples_leaf == 5
    assert tree.random_state == 42
    
    # Test classification tree with all parameters
    model = SimpleRuleEnsemble(
        tree_type='classification',
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    )
    model._X = X
    model._y = y
    tree = model._get_tree_model()
    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.max_depth == 3
    assert tree.min_samples_leaf == 10
    assert tree.random_state == 42
    
    # Test random forest with all parameters
    model = SimpleRuleEnsemble(
        tree_type='random_forest',
        max_depth=5,
        min_samples_leaf=8,
        n_estimators=50,
        random_state=42
    )
    model._X = X
    model._y = y
    forest = model._get_tree_model()
    assert isinstance(forest, RandomForestRegressor)
    assert forest.max_depth == 5
    assert forest.min_samples_leaf == 8
    assert forest.n_estimators == 50
    assert forest.random_state == 42
    
    # Test invalid tree type
    model = SimpleRuleEnsemble(tree_type='invalid')
    model._X = X
    model._y = y
    with pytest.raises(ValueError, match="Unsupported tree_type: invalid"):
        model._get_tree_model()

def test_prune_rules():
    """Test _prune_rules method with different scenarios"""
    # Create test data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    y = np.rec.fromarrays([y], names=['time'])
    
    # Initialize model
    model = SimpleRuleEnsemble(prune_rules=True, prune_threshold=0.5)
    model._X = X
    model._y = y
    
    # Test with no rules
    rules = []
    pruned_rules = model._prune_rules(rules)
    assert len(pruned_rules) == 0
    
    # Test with one rule
    rules = [[(0, "<=", 0.5)]]
    pruned_rules = model._prune_rules(rules)
    assert len(pruned_rules) == 1
    
    # Test with identical rules
    rules = [[(0, "<=", 0.5)], [(0, "<=", 0.5)]]
    pruned_rules = model._prune_rules(rules)
    assert len(pruned_rules) == 1
    
    # Test with different rules
    rules = [[(0, "<=", 0.5)], [(1, "<=", 0.5)]]
    pruned_rules = model._prune_rules(rules)
    assert len(pruned_rules) == 2
    
    # Test with pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    model._X = X_df
    rules = [[(0, "<=", 0.5)], [(1, "<=", 0.5)]]
    pruned_rules = model._prune_rules(rules)
    assert len(pruned_rules) == 2

def test_time_dependent_covariates():
    """Test handling of time-dependent covariates"""
    # Create test data with time-dependent features
    X = pd.DataFrame({
        ('feature_1', 'static'): np.random.randn(100),
        ('feature_2', 'time'): np.random.randn(100),
        ('feature_3', 'time'): np.random.randn(100)
    })
    y = np.random.randn(100)
    y = np.rec.fromarrays([y], names=['time'])
    
    # Test with time-dependent covariates enabled
    model = SimpleRuleEnsemble(support_time_dependent=True)
    model._y = y
    X_processed = model._handle_time_dependent_covariates(X)
    
    # Check that time interactions were added
    expected_time_cols = [
        "('feature_2', 'time')_time",
        "('feature_3', 'time')_time"
    ]
    for col in expected_time_cols:
        assert col in X_processed.columns
    
    # Test with time-dependent covariates disabled
    model = SimpleRuleEnsemble(support_time_dependent=False)
    X_processed = model._handle_time_dependent_covariates(X)
    assert X_processed.equals(X)
    
    # Test with numpy array (should return unchanged)
    X_np = np.random.randn(100, 3)
    X_processed = model._handle_time_dependent_covariates(X_np)
    assert np.array_equal(X_processed, X_np)

def test_evaluate_rules():
    """Test rule evaluation with different inputs"""
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    y = np.rec.fromarrays([y], names=['time'])
    
    # Initialize model and extract rules
    model = SimpleRuleEnsemble(max_rules=3)
    model._X = X
    model._y = y
    model.rules_ = [
        [(0, "<=", 0.5)],  # Simple rule
        [(1, ">", 0.0), (2, "<=", 1.0)],  # Compound rule
        [(3, ">", -1.0), (4, ">", 0.0), (0, "<=", 1.0)]  # Complex rule
    ]
    
    # Test with numpy array
    rule_values = model._evaluate_rules(X)
    assert rule_values.shape == (100, 3)
    assert np.all((rule_values == 0) | (rule_values == 1))
    
    # Verify first rule (simple)
    expected_rule1 = (X[:, 0] <= 0.5).astype(int)
    assert np.array_equal(rule_values[:, 0], expected_rule1)
    
    # Verify second rule (compound)
    expected_rule2 = ((X[:, 1] > 0.0) & (X[:, 2] <= 1.0)).astype(int)
    assert np.array_equal(rule_values[:, 1], expected_rule2)
    
    # Verify third rule (complex)
    expected_rule3 = ((X[:, 3] > -1.0) & (X[:, 4] > 0.0) & (X[:, 0] <= 1.0)).astype(int)
    assert np.array_equal(rule_values[:, 2], expected_rule3)
    
    # Test with pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    rule_values_df = model._evaluate_rules(X_df)
    assert np.array_equal(rule_values, rule_values_df)
    
    # Test with time-dependent covariates
    X_time = pd.DataFrame({
        ('feature_1', 'static'): X[:, 0],
        ('feature_2', 'time'): X[:, 1],
        ('feature_3', 'time'): X[:, 2]
    })
    model.support_time_dependent = True
    rule_values_time = model._evaluate_rules(X_time)
    assert rule_values_time.shape[0] == 100
    assert np.all((rule_values_time == 0) | (rule_values_time == 1)) 