# RuleTimeR Statistical Models Documentation

## Overview

RuleTimeR implements three main types of rule ensemble models for time-to-event analysis:

1. Standard Survival Analysis
2. Competing Risks Analysis
3. Multi-state Modeling

Each model type combines decision rules with statistical methods to provide interpretable predictions while maintaining predictive power.

## Standard Survival Analysis

### RuleSurvival Class

The `RuleSurvival` class implements a rule ensemble model for standard survival analysis, where the outcome is a single event of interest.

#### Key Methods

- `fit(X, y)`: Fits the model to training data
  - `X`: Feature matrix (n_samples, n_features)
  - `y`: Survival data object containing time and event indicator

- `predict_survival(X, times)`: Predicts survival probabilities
  - `X`: Feature matrix for prediction
  - `times`: Array of time points for prediction
  - Returns: Array of survival probabilities (n_samples, n_times)

- `predict_hazard(X, times)`: Predicts hazard rates
  - `X`: Feature matrix for prediction
  - `times`: Array of time points for prediction
  - Returns: Array of hazard rates (n_samples, n_times)

- `feature_importances_`: Property returning feature importance scores

#### Implementation Details

1. Rule Extraction:
   - Uses decision trees to generate rules
   - Each rule is a conjunction of conditions on features
   - Rules capture nonlinear relationships and interactions

2. Model Fitting:
   - Combines rules using regularized Cox regression
   - Uses elastic net penalty for rule selection
   - Computes baseline hazard function

3. Prediction:
   - Computes risk scores using rule evaluations
   - Applies baseline hazard to get absolute risks
   - Converts to survival probabilities

## Competing Risks Analysis

### RuleCompetingRisks Class

The `RuleCompetingRisks` class handles multiple competing events, where subjects can experience one of several distinct event types.

#### Key Methods

- `fit(X, y)`: Fits the model to training data
  - `X`: Feature matrix (n_samples, n_features)
  - `y`: Competing risks data object containing time and event type

- `predict_cumulative_incidence(X, times)`: Predicts cumulative incidence functions
  - `X`: Feature matrix for prediction
  - `times`: Array of time points for prediction
  - Returns: Dictionary of CIFs for each event type

- `predict_subdistribution_hazard(X, times)`: Predicts subdistribution hazards
  - `X`: Feature matrix for prediction
  - `times`: Array of time points for prediction
  - Returns: Dictionary of hazards for each event type

#### Implementation Details

1. Rule Extraction:
   - Generates separate rule sets for each event type
   - Uses Fine-Gray subdistribution hazard approach
   - Handles censoring appropriately for each event type

2. Model Fitting:
   - Fits separate models for each competing event
   - Uses regularized regression for rule selection
   - Computes baseline subdistribution hazards

3. Prediction:
   - Computes event-specific risk scores
   - Applies baseline hazards for each event type
   - Converts to cumulative incidence functions

## Multi-state Modeling

### RuleMultiState Class

The `RuleMultiState` class handles transitions between multiple states over time, such as disease progression stages.

#### Key Methods

- `fit(X, y)`: Fits the model to training data
  - `X`: Feature matrix (n_samples, n_features)
  - `y`: Multi-state data object containing transition information

- `predict_state_occupation(X, times)`: Predicts state occupation probabilities
  - `X`: Feature matrix for prediction
  - `times`: Array of time points for prediction
  - Returns: Dictionary of state occupation probabilities

- `predict_transition_probability(X, times, from_state, to_state)`: Predicts transition probabilities
  - `X`: Feature matrix for prediction
  - `times`: Array of time points for prediction
  - `from_state`: Starting state
  - `to_state`: Target state
  - Returns: Array of transition probabilities

#### Implementation Details

1. State Structure:
   - Defines allowed transitions between states
   - Handles absorbing states (e.g., death)
   - Supports complex transition patterns

2. Rule Extraction:
   - Generates rules for each possible transition
   - Captures state-specific risk factors
   - Handles time-dependent transitions

3. Model Fitting:
   - Fits separate models for each transition
   - Uses regularized regression for rule selection
   - Computes transition-specific baseline hazards

4. Prediction:
   - Computes transition probabilities
   - Updates state occupation probabilities
   - Handles competing transitions appropriately

## Common Features

All models share these common features:

1. Rule Generation:
   - Maximum number of rules controlled by `max_rules` parameter
   - Rule complexity controlled by tree parameters
   - Support for both numerical and categorical features

2. Regularization:
   - Elastic net penalty for rule selection
   - L1 ratio parameter controls sparsity
   - Alpha parameter controls overall regularization strength

3. Model Evaluation:
   - Concordance index for predictive accuracy
   - Time-dependent AUC for discrimination
   - Calibration assessment for predicted probabilities

4. Interpretation:
   - Rule importance scores
   - Feature importance based on rule usage
   - Partial dependence plots for key rules

## Usage Examples

### Standard Survival Analysis
```python
from ruletimer import Survival, RuleSurvival

# Create and fit model
model = RuleSurvival(max_rules=100, alpha=0.05)
model.fit(X_train, y_train)

# Predict survival probabilities
times = [6, 12, 24]  # months
survival_probs = model.predict_survival(X_test, times)
```

### Competing Risks Analysis
```python
from ruletimer import CompetingRisks, RuleCompetingRisks

# Create and fit model
model = RuleCompetingRisks(max_rules=150, alpha=0.05)
model.fit(X_train, y_train)

# Predict cumulative incidence functions
times = [30, 60, 90]  # days
cif = model.predict_cumulative_incidence(X_test, times)
```

### Multi-state Modeling
```python
from ruletimer import MultiState, RuleMultiState, StateStructure
import numpy as np
import pandas as pd

# Define state structure
states = ["Healthy", "Mild", "Moderate", "Severe", "Death"]
transitions = [(0,1), (1,2), (2,3), (1,3), (2,4), (3,4)]
structure = StateStructure(states, transitions)

# Create and fit model
model = RuleMultiState(structure, max_rules=200)
model.fit(X_train, y_train)

# Predict state occupation probabilities
times = [1, 3, 5, 10]  # years
state_probs = {}

# Generate example data for 50 patients with multiple transitions
n_patients = 50
max_transitions = 3  # Maximum number of transitions per patient

# Initialize lists to store data
patient_ids = []
start_times = []
end_times = []
start_states = []
end_states = []
censored = []

# Generate data for each patient
for patient_id in range(n_patients):
    current_time = 0
    current_state = 0  # Start in Healthy state
    
    # Generate transitions for this patient
    n_transitions = np.random.randint(1, max_transitions + 1)
    
    for _ in range(n_transitions):
        # Add patient ID
        patient_ids.append(patient_id)
        
        # Set start time and state
        start_times.append(current_time)
        start_states.append(current_state)
        
        # Generate time until next transition
        time_to_next = np.random.exponential(scale=1.0)
        current_time += time_to_next
        
        # Determine next state based on current state
        if current_state == 0:  # Healthy
            next_state = np.random.choice([1, 3], p=[0.8, 0.2])  # 80% to Mild, 20% to Severe
        elif current_state == 1:  # Mild
            next_state = np.random.choice([2, 3], p=[0.7, 0.3])  # 70% to Moderate, 30% to Severe
        elif current_state == 2:  # Moderate
            next_state = np.random.choice([3, 4], p=[0.6, 0.4])  # 60% to Severe, 40% to Death
        elif current_state == 3:  # Severe
            next_state = 4  # Always to Death
        else:  # Death (absorbing state)
            break
            
        # Set end time and state
        end_times.append(current_time)
        end_states.append(next_state)
        
        # Determine if this transition is censored
        is_censored = np.random.binomial(1, 0.1)  # 10% chance of censoring
        censored.append(is_censored)
        
        # Update current state
        current_state = next_state
        
        # If reached death state, stop generating transitions
        if current_state == 4:
            break

# Convert to numpy arrays
patient_ids = np.array(patient_ids)
start_times = np.array(start_times)
end_times = np.array(end_times)
start_states = np.array(start_states)
end_states = np.array(end_states)
censored = np.array(censored)

# Create feature matrix (example with time-invariant features)
n_features = 5
X = np.random.randn(n_patients, n_features)  # One row per patient

# Create multi-state data object
y = MultiState(
    patient_id=patient_ids,
    start_time=start_times,
    end_time=end_times,
    start_state=start_states,
    end_state=end_states,
    censored=censored
)

# Example of how to handle time-varying features
# Create a DataFrame with patient IDs and features
feature_data = pd.DataFrame({
    'patient_id': patient_ids,
    'feature1': np.random.randn(len(patient_ids)),
    'feature2': np.random.randn(len(patient_ids)),
    'time': start_times  # Time point when feature was measured
})

# You can then use this data to create time-dependent features
# during model fitting and prediction

# Example of how to handle time-varying features
# Create a DataFrame with patient IDs and features
feature_data = pd.DataFrame({
    'patient_id': patient_ids,
    'feature1': np.random.randn(len(patient_ids)),
    'feature2': np.random.randn(len(patient_ids)),
    'time': start_times  # Time point when feature was measured
})

# You can then use this data to create time-dependent features
# during model fitting and prediction
```

## Data Format Requirements

### Standard Survival Analysis

The standard survival model requires two main components:

1. Feature Matrix (`X`):
   - Shape: (n_samples, n_features)
   - Can contain both numerical and categorical features
   - Missing values should be handled before fitting

2. Survival Data (`y`):
   - Created using `Survival` class
   - Requires two arrays:
     - `time`: Array of event/censoring times
     - `event`: Binary array indicating event occurrence (1) or censoring (0)

Example:
```python
import numpy as np
from ruletimer import Survival

# Generate example data
n_samples = 100
X = np.random.randn(n_samples, 5)  # 5 features
time = np.random.exponential(scale=1.0, size=n_samples)
event = np.random.binomial(1, 0.7, size=n_samples)  # 70% events

# Create survival data object
y = Survival(time=time, event=event)
```

### Competing Risks Analysis

The competing risks model requires:

1. Feature Matrix (`X`):
   - Same format as standard survival analysis
   - Shape: (n_samples, n_features)

2. Competing Risks Data (`y`):
   - Created using `CompetingRisks` class
   - Requires two arrays:
     - `time`: Array of event/censoring times
     - `event`: Array of event types (0 for censoring, positive integers for different event types)

Example:
```python
from ruletimer import CompetingRisks

# Generate example data
n_samples = 100
X = np.random.randn(n_samples, 5)
time = np.random.exponential(scale=1.0, size=n_samples)
event = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.4, 0.4])  # 20% censored, 40% each event type

# Create competing risks data object
y = CompetingRisks(time=time, event=event)
```

### Multi-state Modeling

The multi-state model has more complex data requirements:

1. Feature Matrix (`X`):
   - Shape: (n_samples, n_features)
   - Same format as other models
   - Features can be time-varying or time-invariant

2. Multi-state Data (`y`):
   - Created using `MultiState` class
   - Requires five arrays:
     - `patient_id`: Unique identifier for each patient
     - `start_time`: Time when observation period begins
     - `end_time`: Time when transition occurs or observation ends
     - `start_state`: State at the beginning of the period
     - `end_state`: State at the end of the period
   - Optional `censored` array indicating censored observations

3. State Structure:
   - Created using `StateStructure` class
   - Requires:
     - List of state names
     - List of allowed transitions as (from_state, to_state) tuples

Example with patient IDs and multiple transitions:
```python
from ruletimer import MultiState, StateStructure
import numpy as np
import pandas as pd

# Define states and transitions
states = ["Healthy", "Mild", "Moderate", "Severe", "Death"]
transitions = [
    (0, 1),  # Healthy -> Mild
    (1, 2),  # Mild -> Moderate
    (2, 3),  # Moderate -> Severe
    (1, 3),  # Mild -> Severe
    (2, 4),  # Moderate -> Death
    (3, 4)   # Severe -> Death
]
structure = StateStructure(states, transitions)

# Generate example data for 50 patients with multiple transitions
n_patients = 50
max_transitions = 3  # Maximum number of transitions per patient

# Initialize lists to store data
patient_ids = []
start_times = []
end_times = []
start_states = []
end_states = []
censored = []

# Generate data for each patient
for patient_id in range(n_patients):
    current_time = 0
    current_state = 0  # Start in Healthy state
    
    # Generate transitions for this patient
    n_transitions = np.random.randint(1, max_transitions + 1)
    
    for _ in range(n_transitions):
        # Add patient ID
        patient_ids.append(patient_id)
        
        # Set start time and state
        start_times.append(current_time)
        start_states.append(current_state)
        
        # Generate time until next transition
        time_to_next = np.random.exponential(scale=1.0)
        current_time += time_to_next
        
        # Determine next state based on current state
        if current_state == 0:  # Healthy
            next_state = np.random.choice([1, 3], p=[0.8, 0.2])  # 80% to Mild, 20% to Severe
        elif current_state == 1:  # Mild
            next_state = np.random.choice([2, 3], p=[0.7, 0.3])  # 70% to Moderate, 30% to Severe
        elif current_state == 2:  # Moderate
            next_state = np.random.choice([3, 4], p=[0.6, 0.4])  # 60% to Severe, 40% to Death
        elif current_state == 3:  # Severe
            next_state = 4  # Always to Death
        else:  # Death (absorbing state)
            break
            
        # Set end time and state
        end_times.append(current_time)
        end_states.append(next_state)
        
        # Determine if this transition is censored
        is_censored = np.random.binomial(1, 0.1)  # 10% chance of censoring
        censored.append(is_censored)
        
        # Update current state
        current_state = next_state
        
        # If reached death state, stop generating transitions
        if current_state == 4:
            break

# Convert to numpy arrays
patient_ids = np.array(patient_ids)
start_times = np.array(start_times)
end_times = np.array(end_times)
start_states = np.array(start_states)
end_states = np.array(end_states)
censored = np.array(censored)

# Create feature matrix (example with time-invariant features)
n_features = 5
X = np.random.randn(n_patients, n_features)  # One row per patient

# Create multi-state data object
y = MultiState(
    patient_id=patient_ids,
    start_time=start_times,
    end_time=end_times,
    start_state=start_states,
    end_state=end_states,
    censored=censored
)

# Example of how to handle time-varying features
# Create a DataFrame with patient IDs and features
feature_data = pd.DataFrame({
    'patient_id': patient_ids,
    'feature1': np.random.randn(len(patient_ids)),
    'feature2': np.random.randn(len(patient_ids)),
    'time': start_times  # Time point when feature was measured
})

# You can then use this data to create time-dependent features
# during model fitting and prediction
```

### Handling Time-Varying Features

For time-varying features, you can:

1. Create a feature matrix for each time point:
```python
# Example of creating time-dependent features
def create_time_dependent_features(feature_data, prediction_time):
    # Select features measured before prediction_time
    relevant_features = feature_data[feature_data['time'] <= prediction_time]
    
    # Aggregate features (e.g., take most recent measurement)
    latest_features = relevant_features.groupby('patient_id').last()
    
    return latest_features
```

2. Use the feature matrix in predictions:
```python
# Predict state occupation probabilities
times = [1, 3, 5, 10]  # years
state_probs = {}

for t in times:
    # Get features at time t
    X_t = create_time_dependent_features(feature_data, t)
    
    # Predict state occupation probabilities
    state_probs[t] = model.predict_state_occupation(X_t, [t])
```

### Data Validation

All models include built-in data validation to check for common issues:

1. Time Validation:
   - Ensures end times are after start times
   - Checks for negative times
   - Validates time ordering in multi-state transitions

2. State Validation:
   - Verifies state indices are valid
   - Checks for allowed transitions
   - Validates absorbing states

3. Censoring Validation:
   - Ensures proper censoring indicators
   - Validates censoring times
   - Checks for competing events

4. Feature Validation:
   - Checks for missing values
   - Validates feature types
   - Ensures proper dimensionality

Example of data validation:
```python
# The following will raise ValueError if data is invalid
model.fit(X, y)  # Includes validation checks
```

## References

1. Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. The Annals of Applied Statistics, 2(3), 916-954.
2. Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. Journal of the American Statistical Association, 94(446), 496-509.
3. Andersen, P. K., & Keiding, N. (2002). Multi-state models for event history analysis. Statistical Methods in Medical Research, 11(2), 91-115. 