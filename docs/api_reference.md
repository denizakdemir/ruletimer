# RuleTimeR API Reference

## Core Classes

### BaseTimeToEventModel

Abstract base class for all time-to-event models.

```python
class BaseTimeToEventModel(ABC)
```

#### Methods

##### fit
```python
fit(X: Union[np.ndarray, pd.DataFrame], y: Union[Survival, CompetingRisks, MultiState]) -> BaseTimeToEventModel
```
Fit the model to the training data.

**Parameters:**
- `X`: array-like of shape (n_samples, n_features) - Training data
- `y`: Survival, CompetingRisks, or MultiState - Target values

**Returns:**
- `self`: BaseTimeToEventModel - Fitted model

##### predict_survival
```python
predict_survival(X: Union[np.ndarray, pd.DataFrame], times: np.ndarray) -> np.ndarray
```
Predict survival probabilities for given data points at specified times.

**Parameters:**
- `X`: array-like of shape (n_samples, n_features) - Data to predict for
- `times`: array-like - Times at which to predict survival

**Returns:**
- `survival`: array-like of shape (n_samples, n_times) - Predicted survival probabilities

##### predict_risk
```python
predict_risk(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```
Predict risk scores for given data points.

**Parameters:**
- `X`: array-like of shape (n_samples, n_features) - Data to predict for

**Returns:**
- `risk`: array-like of shape (n_samples,) - Predicted risk scores

##### feature_importances_
```python
@property
feature_importances_() -> np.ndarray
```
Get feature importances from the fitted model.

**Returns:**
- `importances`: array-like of shape (n_features,) - Feature importances

### StateManager

Class for managing state indices and transitions in multi-state models.

```python
class StateManager
```

#### Methods

##### __init__
```python
__init__(states: List[str], transitions: List[Tuple[int, int]])
```
Initialize state manager with states and valid transitions.

**Parameters:**
- `states`: list of str - List of state names
- `transitions`: list of tuple - List of valid transitions as (from_state, to_state) pairs

##### to_internal_index
```python
to_internal_index(state: Union[str, int]) -> int
```
Convert external state representation to internal index.

**Parameters:**
- `state`: str or int - State name or index

**Returns:**
- `int`: Internal state index

**Raises:**
- `ValueError`: If state is invalid

##### to_external_state
```python
to_external_state(idx: int) -> str
```
Convert internal index to external state representation.

**Parameters:**
- `idx`: int - Internal state index

**Returns:**
- `str`: State name

**Raises:**
- `ValueError`: If index is invalid

##### get_possible_transitions
```python
get_possible_transitions(from_state: Optional[Union[str, int]] = None) -> List[Union[int, Tuple[int, int]]]
```
Get possible transitions from a state or all transitions.

**Parameters:**
- `from_state`: str or int, optional - State to get transitions from. If None, returns all transitions.

**Returns:**
- `list`: If from_state is None: List of (from_state, to_state) tuples
         If from_state is provided: List of possible target state indices

##### validate_transition
```python
validate_transition(from_state: Union[str, int], to_state: Union[str, int]) -> bool
```
Validate if a transition between two states is valid.

**Parameters:**
- `from_state`: str or int - Starting state
- `to_state`: str or int - Target state

**Returns:**
- `bool`: True if transition is valid, False otherwise

##### is_absorbing_state
```python
is_absorbing_state(state: Union[str, int]) -> bool
```
Check if a state is absorbing (has no outgoing transitions).

**Parameters:**
- `state`: str or int - State to check

**Returns:**
- `bool`: True if state is absorbing, False otherwise

##### get_absorbing_states
```python
get_absorbing_states() -> List[int]
```
Get all absorbing states in the system.

**Returns:**
- `list of int`: List of absorbing state indices

##### validate_state_sequence
```python
validate_state_sequence(sequence: List[Union[str, int]]) -> bool
```
Validate a sequence of states.

**Parameters:**
- `sequence`: list - Sequence of states

**Returns:**
- `bool`: True if sequence is valid

## Model Implementations

### BaseMultiStateModel

Base class for all multi-state time-to-event models.

```python
class BaseMultiStateModel(ABC)
```

#### Parameters
- `states`: list of str - List of state names
- `transitions`: list of tuple - List of valid transitions as (from_state, to_state) pairs
- `hazard_method`: str, default="nelson-aalen" - Method for hazard estimation: "nelson-aalen" or "parametric"

#### Methods

##### predict_transition_hazard
```python
predict_transition_hazard(
    X: np.ndarray,
    times: np.ndarray,
    from_state: Union[str, int],
    to_state: Union[str, int]
) -> np.ndarray
```
Predict transition-specific hazard.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `from_state`: str or int - Starting state
- `to_state`: str or int - Target state

**Returns:**
- `hazard`: array-like - Predicted transition hazards

##### predict_cumulative_hazard
```python
predict_cumulative_hazard(
    X: np.ndarray,
    times: np.ndarray,
    from_state: Union[str, int],
    to_state: Union[str, int]
) -> np.ndarray
```
Predict cumulative hazard for a specific transition.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `from_state`: str or int - Starting state
- `to_state`: str or int - Target state

**Returns:**
- `cumulative_hazard`: array-like - Predicted cumulative hazards

##### predict_transition_probability
```python
predict_transition_probability(
    X: np.ndarray,
    times: np.ndarray,
    from_state: Union[str, int],
    to_state: Union[str, int]
) -> np.ndarray
```
Predict transition probabilities.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `from_state`: str or int - Starting state
- `to_state`: str or int - Target state

**Returns:**
- `probabilities`: array-like - Predicted transition probabilities

##### predict_state_occupation
```python
predict_state_occupation(
    X: np.ndarray,
    times: np.ndarray,
    initial_state: Union[str, int]
) -> Dict[int, np.ndarray]
```
Predict state occupation probabilities.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `initial_state`: str or int - Initial state

**Returns:**
- `probabilities`: dict - Dictionary mapping states to occupation probabilities

### RuleMultiState

A rule-based implementation of the multi-state model.

```python
class RuleMultiState(BaseMultiStateModel)
```

#### Parameters
- `max_rules`: int, default=100 - Maximum number of rules to generate
- `alpha`: float, default=0.1 - Elastic net mixing parameter
- `state_structure`: StateStructure, optional - Structure of the multi-state model
- `max_depth`: int, default=3 - Maximum depth of decision trees
- `min_samples_leaf`: int, default=10 - Minimum samples in leaf nodes
- `n_estimators`: int, default=100 - Number of trees in the forest
- `tree_type`: str, default='classification' - Type of decision trees
- `tree_growing_strategy`: str, default='forest' - Strategy for growing trees
- `prune_rules`: bool, default=True - Whether to prune rules
- `l1_ratio`: float, default=0.5 - Ratio of L1 to L2 regularization
- `random_state`: int, optional - Random state for reproducibility
- `hazard_method`: str, default="nelson-aalen" - Method for hazard estimation

#### Methods

##### get_feature_importances
```python
get_feature_importances(transition: Tuple[int, int]) -> np.ndarray
```
Get feature importances for a specific transition.

**Parameters:**
- `transition`: tuple - Transition to get importances for

**Returns:**
- `importances`: array-like - Feature importances

##### predict_cumulative_incidence
```python
predict_cumulative_incidence(
    X: np.ndarray,
    times: np.ndarray,
    target_state: Union[str, int]
) -> np.ndarray
```
Predict cumulative incidence for a target state.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `target_state`: str or int - Target state

**Returns:**
- `incidence`: array-like - Predicted cumulative incidence

### RuleSurvival

A rule-based survival model that implements standard survival analysis as a two-state model.

```python
class RuleSurvival(BaseMultiStateModel)
```

#### Parameters
- `hazard_method`: str, default="nelson-aalen" - Method for hazard estimation: "nelson-aalen" or "parametric"

#### Methods

##### fit
```python
fit(X: np.ndarray, y: Survival) -> RuleSurvival
```
Fit the survival model.

**Parameters:**
- `X`: array-like - Training data
- `y`: Survival - Target survival data

**Returns:**
- `self`: RuleSurvival - Fitted model

##### predict_survival
```python
predict_survival(X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray
```
Predict survival probabilities.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like, optional - Times at which to predict survival

**Returns:**
- `survival`: array-like - Predicted survival probabilities

##### predict_hazard
```python
predict_hazard(X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray
```
Predict hazard rates.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like, optional - Times at which to predict hazard

**Returns:**
- `hazard`: array-like - Predicted hazard rates

### RuleCompetingRisks

A rule-based competing risks model that handles multiple event types.

```python
class RuleCompetingRisks(BaseMultiStateModel)
```

#### Parameters
- `n_rules`: int, default=100 - Maximum number of rules to generate for each event type
- `min_support`: float, default=0.1 - Minimum proportion of samples that must satisfy a rule
- `alpha`: float, default=0.5 - Elastic net mixing parameter
- `l1_ratio`: float, default=0.5 - Ratio of L1 to L2 regularization
- `max_iter`: int, default=1000 - Maximum number of iterations
- `tol`: float, default=1e-4 - Tolerance for optimization
- `random_state`: int or RandomState, default=None - Random state for reproducibility

#### Methods

##### fit
```python
fit(X: np.ndarray, y: CompetingRisks) -> RuleCompetingRisks
```
Fit the competing risks model.

**Parameters:**
- `X`: array-like - Training data
- `y`: CompetingRisks - Target competing risks data

**Returns:**
- `self`: RuleCompetingRisks - Fitted model

##### predict_cumulative_incidence
```python
predict_cumulative_incidence(
    X: np.ndarray,
    times: np.ndarray,
    event_type: Optional[Union[str, int]] = None,
    event_types: Optional[List[Union[str, int]]] = None
) -> Union[np.ndarray, Dict[Union[str, int], np.ndarray]]
```
Predict cumulative incidence functions.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `event_type`: str or int, optional - Specific event type to predict for
- `event_types`: list of str or int, optional - List of event types to predict for

**Returns:**
- `cif`: array-like or dict - Predicted cumulative incidence functions

##### predict_cause_specific_hazard
```python
predict_cause_specific_hazard(
    X: np.ndarray,
    times: np.ndarray,
    event_type: Union[str, int]
) -> np.ndarray
```
Predict cause-specific hazard rates.

**Parameters:**
- `X`: array-like - Data to predict for
- `times`: array-like - Times at which to predict
- `event_type`: str or int - Event type to predict for

**Returns:**
- `hazard`: array-like - Predicted cause-specific hazard rates

## Data Types

### Survival
Class for handling survival data.

### CompetingRisks
Class for handling competing risks data.

### MultiState
Class for handling multi-state data.

## Utility Classes

### TimeHandler
Class for handling time-related operations in time-to-event models.

## Visualization Functions

### plot_rule_importance
```python
plot_rule_importance(
    rules: Union[Dict[Tuple[int, int], List[str]], BaseRuleEnsemble],
    importances: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    top_n: int = 10,
    figsize: tuple = (10, 6)
) -> matplotlib.figure.Figure
```
Plot rule importance as a horizontal bar chart.

**Parameters:**
- `rules`: dict or BaseRuleEnsemble - Either a dictionary mapping transitions to rules or a fitted model
- `importances`: dict, optional - Dictionary mapping transitions to rule importances
- `top_n`: int, default=10 - Number of top rules to plot
- `figsize`: tuple, default=(10, 6) - Figure size

**Returns:**
- `matplotlib.figure.Figure` - The figure object

### plot_cumulative_incidence
```python
plot_cumulative_incidence(
    model: BaseRuleEnsemble,
    X: Union[np.ndarray, pd.DataFrame],
    event_types: List[int],
    times: Optional[np.ndarray] = None,
    figsize: tuple = (10, 6)
) -> matplotlib.figure.Figure
```
Plot cumulative incidence functions with confidence intervals.

**Parameters:**
- `model`: BaseRuleEnsemble - Fitted competing risks model
- `X`: array-like of shape (n_samples, n_features) - Data to predict for
- `event_types`: list of int - Event types to plot
- `times`: array-like, optional - Times at which to plot cumulative incidence
- `figsize`: tuple, default=(10, 6) - Figure size

**Returns:**
- `matplotlib.figure.Figure` - The figure object

### plot_state_transitions
```python
plot_state_transitions(
    model: BaseRuleEnsemble,
    X: Union[np.ndarray, pd.DataFrame],
    time: float,
    initial_state: int = 0,
    figsize: tuple = (10, 6)
) -> None
```
Plot state transition diagram with probabilities.

**Parameters:**
- `model`: BaseRuleEnsemble - Fitted model
- `X`: array-like of shape (n_samples, n_features) - Data to predict for
- `time`: float - Time at which to plot state occupation probabilities
- `initial_state`: int, default=0 - Initial state for prediction
- `figsize`: tuple, default=(10, 6) - Figure size

### plot_state_occupation
```python
plot_state_occupation(
    times: np.ndarray,
    state_probs: Dict[int, np.ndarray],
    state_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.1,
    title: str = "State Occupation Probabilities",
    xlabel: str = "Time",
    ylabel: str = "Probability",
    legend_loc: str = "best"
) -> None
```
Plot state occupation probabilities over time.

**Parameters:**
- `times`: array-like - Time points
- `state_probs`: dict - Dictionary mapping state indices to occupation probabilities
- `state_names`: list of str, optional - Names of states
- `figsize`: tuple, default=(10, 6) - Figure size
- `alpha`: float, default=0.1 - Transparency for confidence intervals
- `title`: str, default="State Occupation Probabilities" - Plot title
- `xlabel`: str, default="Time" - X-axis label
- `ylabel`: str, default="Probability" - Y-axis label
- `legend_loc`: str, default="best" - Legend location

### plot_importance_comparison
```python
plot_importance_comparison(
    model: RuleCompetingRisks,
    top_n: int = 10,
    figsize: tuple = (12, 8)
) -> None
```
Plot comparison of feature importances across different event types.

**Parameters:**
- `model`: RuleCompetingRisks - Fitted competing risks model
- `top_n`: int, default=10 - Number of top features to plot
- `figsize`: tuple, default=(12, 8) - Figure size

### plot_importance_heatmap
```python
plot_importance_heatmap(
    model: RuleCompetingRisks,
    top_n: int = 10,
    figsize: tuple = (12, 8)
) -> None
```
Plot feature importance heatmap across different event types.

**Parameters:**
- `model`: RuleCompetingRisks - Fitted competing risks model
- `top_n`: int, default=10 - Number of top features to plot
- `figsize`: tuple, default=(12, 8) - Figure size

## Example Usage

### Basic Survival Analysis
```python
from ruletimer import CoxPH
from ruletimer.data import Survival

# Create model
model = CoxPH()

# Prepare data
X = ...  # Features
y = Survival(time, event)  # Time and event indicators

# Fit model
model.fit(X, y)

# Predict survival
times = np.linspace(0, 100, 100)
survival_probs = model.predict_survival(X_new, times)
```

### Accessing Feature Importances

#### For Standard Survival Models
```python
from ruletimer import RuleSurvival
from ruletimer.data import Survival

# Create and fit model
model = RuleSurvival()
model.fit(X, y)

# Access feature importances as property
feature_importances = model.feature_importances_
```

#### For Competing Risks Models
```python
from ruletimer import RuleCompetingRisks
from ruletimer.data import CompetingRisks

# Create and fit model
model = RuleCompetingRisks()
model.fit(X, y)

# Get importances for a specific event type
importances_event1 = model.get_feature_importances(event_type=1)
```

#### For Multi-State Models
```python
from ruletimer import RuleMultiState
from ruletimer.data import MultiState

# Create and fit model
model = RuleMultiState(state_structure=structure)
model.fit(X, y)

# Get importances for a specific transition
importances_transition = model.get_feature_importances(transition=(0, 1))
```

### Competing Risks Analysis
```python
from ruletimer import RuleCompetingRisks
from ruletimer.data import CompetingRisks

# Create model
model = RuleCompetingRisks(n_rules=50, min_support=0.2)

# Prepare data
X = ...  # Features
y = CompetingRisks(time=times, event=events)  # Time and event indicators

# Fit model
model.fit(X, y)

# Predict cumulative incidence
times = np.linspace(0, 100, 100)
cif = model.predict_cumulative_incidence(X_new, times)
```

### Multi-State Analysis
```python
from ruletimer import RuleMultiState
from ruletimer.data import MultiState

# Define states and transitions
states = ['healthy', 'disease', 'death']
transitions = [(0, 1), (0, 2), (1, 2)]  # healthy->disease, healthy->death, disease->death

# Create model
model = RuleMultiState(
    states=states,
    transitions=transitions,
    max_rules=100,
    max_depth=3
)

# Prepare data
X = ...  # Features
y = MultiState(...)  # Multi-state data

# Fit model
model.fit(X, y)

# Predict state occupation probabilities
times = np.linspace(0, 100, 100)
probs = model.predict_state_occupation(X_new, times, initial_state='healthy')
```

### Visualizing Rule Importance
```python
from ruletimer.visualization import plot_rule_importance

# Plot rule importance from a fitted model
plot_rule_importance(model, top_n=15)

# Plot rule importance from a dictionary
rules = {(0, 1): ["feature1 > 0.5", "feature2 < 0.3"]}
importances = {(0, 1): [0.8, 0.5]}
plot_rule_importance(rules, importances)
```

### Visualizing Cumulative Incidence
```python
from ruletimer.visualization import plot_cumulative_incidence

# Plot cumulative incidence for specific event types
plot_cumulative_incidence(
    model=model,
    X=X_test,
    event_types=[1, 2],
    times=np.linspace(0, 100, 100)
)
```

### Visualizing State Transitions
```python
from ruletimer.visualization import plot_state_transitions

# Plot state transition diagram at a specific time
plot_state_transitions(
    model=model,
    X=X_test,
    time=50,
    initial_state=0
)
```

### Visualizing State Occupation
```python
from ruletimer.visualization import plot_state_occupation

# Get state occupation probabilities
times = np.linspace(0, 100, 100)
state_probs = model.predict_state_occupation(X_test, times, initial_state=0)

# Plot state occupation probabilities
plot_state_occupation(
    times=times,
    state_probs=state_probs,
    state_names=['Healthy', 'Disease', 'Death']
)
```

### Visualizing Feature Importance
```python
from ruletimer.visualization import plot_importance_comparison, plot_importance_heatmap

# Plot feature importance comparison
plot_importance_comparison(model, top_n=15)

# Plot feature importance heatmap
plot_importance_heatmap(model, top_n=15)
```

## Utility Functions and Classes

### State Management

#### `StateStructure` Class
Class for managing state structure in multi-state models.

```python
from ruletimer.utils import StateStructure
```

**Parameters:**
- `states` (list): List of state indices
- `transitions` (list of tuples): List of valid transitions as (from_state_idx, to_state_idx) pairs
- `state_names` (list, optional): List of state names. If not provided, state indices will be used as names.

**Methods:**
- `get_state_index(state)`: Get index of a state
- `get_state_name(state)`: Get name of a state
- `is_valid_transition(from_state, to_state)`: Check if a transition is valid
- `get_next_states(state)`: Get all possible next states from a given state
- `get_previous_states(state)`: Get all possible previous states to a given state
- `transitions_from_state(state)`: Get all transitions from a given state
- `transitions_to_state(state)`: Get all transitions to a given state
- `get_state_names()`: Get list of all state names
- `get_transition_names()`: Get list of all transitions with state names

### Time and Prediction Utilities

#### `create_prediction_grid`
Create a standard time grid for predictions based on observed data.

```python
from ruletimer.utils.time_utils import create_prediction_grid
```

**Parameters:**
- `data` (MultiState, Survival, or CompetingRisks): Time-to-event data
- `n_points` (int, default=100): Number of time points
- `max_time` (float, optional): Maximum time value (if None, uses quantile_max of observed times)
- `quantile_max` (float, default=0.95): Quantile of observed times to use as maximum if max_time is None

**Returns:**
- `np.ndarray`: Array of time points for prediction

#### `to_multi_state_format`
Convert any time-to-event data to multi-state format.

```python
from ruletimer.utils.time_utils import to_multi_state_format
```

**Parameters:**
- `data` (Survival, CompetingRisks, or related data): Time-to-event data
- `data_type` (str, default='survival'): Type of data ('survival', 'competing_risks')

**Returns:**
- `MultiState`: Data converted to multi-state format

#### `bootstrap_confidence_intervals`
```python
bootstrap_confidence_intervals(
    model,
    X: np.ndarray,
    y,
    times: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    prediction_method: str = 'predict_state_occupation'
) -> Dict
```
Calculate bootstrap confidence intervals for predictions.

**Parameters:**
- `model`: BaseRuleEnsemble - Fitted model
- `X`: array-like - Feature matrix
- `y`: MultiState, Survival, or CompetingRisks - Time-to-event data
- `times`: array-like - Time points for prediction
- `n_bootstrap`: int, default=100 - Number of bootstrap samples
- `alpha`: float, default=0.05 - Significance level for confidence intervals
- `prediction_method`: str, default='predict_state_occupation' - Method to use for prediction

**Returns:**
- `dict`: Dictionary containing:
  - For multi-state predictions: Dictionary mapping states to arrays of shape (n_bootstrap, n_samples, n_times)
  - For single outcome predictions: Array of shape (n_bootstrap, n_samples, n_times)

### Importance Analysis

#### `ImportanceAnalyzer` Class
Analyze variable importance and dependencies in rule ensemble models.

```python
from ruletimer.utils.importance import ImportanceAnalyzer
```

**Methods:**
- `calculate_permutation_importance`: Calculate permutation-based feature importance
- `calculate_dependence_matrix`: Calculate feature dependency matrix using partial dependence

**Parameters for `calculate_permutation_importance`:**
- `model` (BaseRuleEnsemble): Fitted model
- `X` (array-like): Feature matrix
- `y` (MultiState, Survival, or CompetingRisks): Time-to-event data
- `prediction_func` (callable, optional): Function to generate predictions
- `n_repeats` (int, default=10): Number of permutation repeats
- `random_state` (int, optional): Random state for reproducibility

**Parameters for `calculate_dependence_matrix`:**
- `model` (BaseRuleEnsemble): Fitted model
- `X` (array-like): Feature matrix
- `prediction_func` (callable, optional): Function to generate predictions
- `threshold` (float, default=0.05): Threshold for significance of dependency
- `n_points` (int, default=10): Number of points for evaluating each feature

### Prediction Utilities

#### `UnifiedTransitionCalculator` Class
Calculate transition probabilities for multi-state models.

```python
from ruletimer.utils.prediction_utils import UnifiedTransitionCalculator
```

**Methods:**
- `calculate_transition_matrix`: Calculate transition probability matrix P(s,t) for each time point
- `_calculate_transition_matrix_ci`: Calculate confidence intervals for transition matrix using bootstrap

**Parameters for `calculate_transition_matrix`:**
- `model` (RuleMultiState): Fitted multi-state model
- `X` (array-like): Feature matrix
- `times` (array-like): Time points for prediction
- `include_ci` (bool, default=True): Whether to include confidence intervals
- `alpha` (float, default=0.05): Significance level for confidence intervals
- `n_bootstrap` (int, default=100): Number of bootstrap samples

#### `UnifiedPredictionCalculator` Class
Unified calculator for various predictions from multi-state models.

```python
from ruletimer.utils.prediction_utils import UnifiedPredictionCalculator
```

**Methods:**
- `survival_from_multistate`: Calculate survival probabilities from multi-state model
- `cumulative_incidence_from_multistate`: Calculate cumulative incidence functions from multi-state model

**Parameters for `survival_from_multistate`:**
- `model` (RuleMultiState): Fitted multi-state model
- `X` (array-like): Feature matrix
- `times` (array-like): Time points for prediction
- `initial_state` (int, default=1): Initial state
- `death_state` (int, default=2): Absorbing state representing death
- `include_ci` (bool, default=True): Whether to include confidence intervals
- `alpha` (float, default=0.05): Significance level for confidence intervals
- `n_bootstrap` (int, default=100): Number of bootstrap samples

### Hazard Estimation

#### `HazardEstimator` Class
Unified hazard estimation for all time-to-event models.

```python
from ruletimer.utils.hazard_estimation import HazardEstimator
```

**Methods:**
- `estimate_baseline_hazard`: Estimate baseline hazard using various methods
- `estimate_cumulative_hazard`: Estimate cumulative hazard using the trapezoidal rule
- `transform_hazard`: Transform cumulative hazard to survival or CIF

**Parameters for `estimate_baseline_hazard`:**
- `times` (np.ndarray): Event/censoring times
- `events` (np.ndarray): Event indicators (1 if event, 0 if censored)
- `weights` (np.ndarray, optional): Case weights for hazard estimation
- `method` (str): Estimation method: "nelson-aalen" or "parametric"

**Parameters for `estimate_cumulative_hazard`:**
- `times` (np.ndarray): Array of unique event times
- `baseline_hazard` (np.ndarray): Array of baseline hazard values corresponding to times

**Parameters for `transform_hazard`:**
- `cumulative_hazard` (np.ndarray): Cumulative hazard values
- `transform` (str): Transformation type: "exp" for survival, "cif" for CIF

## Data Handling

### Core Data Classes

#### `Survival` Class
Class for handling survival data.

```python
from ruletimer.data import Survival
```

**Parameters:**
- `time` (array-like): Event or censoring times
- `event` (array-like): Event indicators (1 if event, 0 if censored)
- `weights` (array-like, optional): Case weights

**Methods:**
- `get_event_times()`: Get unique event times
- `get_risk_set(time)`: Get risk set at a given time
- `get_event_set(time)`: Get event set at a given time

#### `CompetingRisks` Class
Class for handling competing risks data.

```python
from ruletimer.data import CompetingRisks
```

**Parameters:**
- `time` (array-like): Event or censoring times
- `event` (array-like): Event type indicators (0 for censoring, >0 for different event types)
- `weights` (array-like, optional): Case weights

**Methods:**
- `get_event_types()`: Get unique event types
- `get_event_times(event_type)`: Get unique event times for a specific event type
- `get_risk_set(time)`: Get risk set at a given time
- `get_event_set(time, event_type)`: Get event set at a given time for a specific event type

#### `MultiState` Class
Class for handling multi-state data.

```python
from ruletimer.data import MultiState
```

**Parameters:**
- `time` (array-like): Transition or censoring times
- `from_state` (array-like): Starting states
- `to_state` (array-like): Target states
- `weights` (array-like, optional): Case weights

**Methods:**
- `get_states()`: Get unique states
- `get_transitions()`: Get unique transitions
- `get_transition_times(from_state, to_state)`: Get unique transition times for a specific transition
- `get_risk_set(time, from_state)`: Get risk set at a given time for a specific state
- `get_transition_set(time, from_state, to_state)`: Get transition set at a given time for a specific transition

### Data Conversion

#### `DataConverter` Class
Class for converting between different data formats.

```python
from ruletimer.data import DataConverter
```

**Methods:**
- `to_survival_format(data)`: Convert data to survival format
- `to_competing_risks_format(data)`: Convert data to competing risks format
- `to_multi_state_format(data)`: Convert data to multi-state format
- `validate_data(data, data_type)`: Validate data format

### Data Validation

#### `DataValidator` Class
Class for validating time-to-event data.

```python
from ruletimer.data import DataValidator
```

**Methods:**
- `validate_survival_data(data)`: Validate survival data
- `validate_competing_risks_data(data)`: Validate competing risks data
- `validate_multi_state_data(data)`: Validate multi-state data
- `check_time_consistency(data)`: Check time consistency in data

## Model Evaluation

### `ModelEvaluator` Class
Class for evaluating time-to-event models.

```python
from ruletimer.evaluation import ModelEvaluator
```

**Methods:**

#### `transition_concordance`
```python
transition_concordance(model, X, y)
```
Calculate concordance index for transitions in multi-state models.

**Parameters:**
- `model`: BaseRuleEnsemble - Fitted model
- `X`: array-like - Feature matrix
- `y`: MultiState - True multi-state outcomes

**Returns:**
- `float`: Concordance index

#### `compare_models`
```python
compare_models(
    models: Dict[str, Union[RuleSurvival, RuleCompetingRisks, RuleMultiState]],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[Survival, CompetingRisks, MultiState],
    times: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]
```
Compare multiple models using various metrics.

**Parameters:**
- `models`: dict - Dictionary mapping model names to fitted models
- `X`: array-like - Feature matrix
- `y`: Survival, CompetingRisks, or MultiState - True outcomes
- `times`: array-like, optional - Time points for time-dependent metrics

**Returns:**
- `dict`: Dictionary containing evaluation metrics for each model

#### `time_dependent_auc`
```python
time_dependent_auc(model, X, y, times)
```
Calculate time-dependent AUC for survival models.

**Parameters:**
- `model`: RuleSurvival - Fitted survival model
- `X`: array-like - Feature matrix
- `y`: Survival - True survival outcomes
- `times`: array-like - Time points for evaluation

**Returns:**
- `array-like`: Time-dependent AUC values

#### `time_dependent_auc_competing_risks`
```python
time_dependent_auc_competing_risks(model, X, y, times)
```
Calculate time-dependent AUC for competing risks models.

**Parameters:**
- `model`: RuleCompetingRisks - Fitted competing risks model
- `X`: array-like - Feature matrix
- `y`: CompetingRisks - True competing risks outcomes
- `times`: array-like - Time points for evaluation

**Returns:**
- `dict`: Dictionary mapping event types to time-dependent AUC values

#### `time_dependent_auc_multi_state`
```python
time_dependent_auc_multi_state(model, X, y, times)
```
Calculate time-dependent AUC for multi-state models.

**Parameters:**
- `model`: RuleMultiState - Fitted multi-state model
- `X`: array-like - Feature matrix
- `y`: MultiState - True multi-state outcomes
- `times`: array-like - Time points for evaluation

**Returns:**
- `dict`: Dictionary mapping transitions to time-dependent AUC values

### Time and Prediction Utilities

#### `bootstrap_confidence_intervals`
```python
bootstrap_confidence_intervals(
    model,
    X: np.ndarray,
    y,
    times: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    prediction_method: str = 'predict_state_occupation'
) -> Dict
```
Calculate bootstrap confidence intervals for predictions.

**Parameters:**
- `model`: BaseRuleEnsemble - Fitted model
- `X`: array-like - Feature matrix
- `y`: MultiState, Survival, or CompetingRisks - Time-to-event data
- `times`: array-like - Time points for prediction
- `n_bootstrap`: int, default=100 - Number of bootstrap samples
- `alpha`: float, default=0.05 - Significance level for confidence intervals
- `prediction_method`: str, default='predict_state_occupation' - Method to use for prediction

**Returns:**
- `dict`: Dictionary containing:
  - For multi-state predictions: Dictionary mapping states to arrays of shape (n_bootstrap, n_samples, n_times)
  - For single outcome predictions: Array of shape (n_bootstrap, n_samples, n_times)

## Time Handling

### `TimeHandler` Class
Class for handling time-related operations in time-to-event models.

```python
from ruletimer import TimeHandler
```

**Methods:**
- `create_time_grid(data, n_points=100)`: Create a time grid for predictions
- `align_times(times1, times2)`: Align two sets of time points
- `interpolate_values(times, values, new_times)`: Interpolate values at new time points
- `calculate_time_differences(times1, times2)`: Calculate time differences between two sets of time points

## State Management

### `StateManager` Class
Class for managing state indices and transitions in multi-state models.

```python
from ruletimer import StateManager
```

**Methods:**
- `to_internal_index(state)`: Convert external state representation to internal index
- `to_external_state(idx)`: Convert internal index to external state representation
- `get_possible_transitions(from_state)`: Get possible transitions from a state
- `validate_transition(from_state, to_state)`: Validate if a transition is valid
- `is_absorbing_state(state)`: Check if a state is absorbing
- `get_absorbing_states()`: Get all absorbing states
- `validate_state_sequence(sequence)`: Validate a sequence of states

## Visualization

### Additional Visualization Functions

#### `plot_survival_curves`
```python
plot_survival_curves(
    model: BaseTimeToEventModel,
    X: Union[np.ndarray, pd.DataFrame],
    times: Optional[np.ndarray] = None,
    figsize: tuple = (10, 6)
) -> matplotlib.figure.Figure
```
Plot survival curves with confidence intervals.

**Parameters:**
- `model`: BaseTimeToEventModel - Fitted model
- `X`: array-like - Data to predict for
- `times`: array-like, optional - Times at which to plot survival
- `figsize`: tuple, default=(10, 6) - Figure size

#### `plot_hazard_curves`
```python
plot_hazard_curves(
    model: BaseTimeToEventModel,
    X: Union[np.ndarray, pd.DataFrame],
    times: Optional[np.ndarray] = None,
    figsize: tuple = (10, 6)
) -> matplotlib.figure.Figure
```
Plot hazard curves with confidence intervals.

**Parameters:**
- `model`: BaseTimeToEventModel - Fitted model
- `X`: array-like - Data to predict for
- `times`: array-like, optional - Times at which to plot hazard
- `figsize`: tuple, default=(10, 6) - Figure size

#### `plot_calibration`
```python
plot_calibration(
    model: BaseTimeToEventModel,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[Survival, CompetingRisks, MultiState],
    times: np.ndarray,
    figsize: tuple = (10, 6)
) -> matplotlib.figure.Figure
```
Plot calibration curves for model predictions.

**Parameters:**
- `model`: BaseTimeToEventModel - Fitted model
- `X`: array-like - Data to predict for
- `y`: Survival, CompetingRisks, or MultiState - True outcomes
- `times`: array-like - Times at which to evaluate calibration
- `figsize`: tuple, default=(10, 6) - Figure size 