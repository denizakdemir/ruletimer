# RuleTimeR Quick Start Guide

## Installation

```bash
pip install ruletimer
```

## Basic Usage

### 1. Standard Survival Analysis

```python
import numpy as np
from ruletimer.models import RuleSurvival
from ruletimer.data import Survival

# Generate example data
X = np.random.randn(100, 5)  # Features
times = np.random.exponential(scale=5, size=100)  # Survival times
events = np.random.binomial(1, 0.7, size=100)  # Event indicators (1=event, 0=censored)

# Create survival data object
y = Survival(time=times, event=events)

# Initialize and fit model
model = RuleSurvival()
model.fit(X, y)

# Make predictions
test_times = np.linspace(0, 10, 100)
survival_probs = model.predict_survival(X, test_times)
```

### 2. Competing Risks Analysis

```python
from ruletimer.models import RuleCompetingRisks
from ruletimer.data import CompetingRisks

# Generate example data with two competing events
X = np.random.randn(100, 5)
times = np.random.exponential(scale=5, size=100)
events = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.4, 0.3])  # 0=censored, 1=event1, 2=event2

# Create competing risks data object
y = CompetingRisks(time=times, event=events)

# Initialize and fit model
model = RuleCompetingRisks()
model.fit(X, y)

# Predict cumulative incidence
test_times = np.linspace(0, 10, 100)
incidence = model.predict_cumulative_incidence(X, test_times, event_types=[1, 2])
```

### 3. Multi-state Modeling

```python
from ruletimer.models import RuleMultiState
from ruletimer.data import MultiState
from ruletimer.utils import StateStructure

# Define state structure
states = [1, 2, 3]  # States (1-based indexing)
transitions = [(1, 2), (2, 3)]  # Valid transitions
state_names = ["Healthy", "Mild", "Severe"]  # Human-readable names

# Create state structure
state_structure = StateStructure(
    states=states,
    transitions=transitions,
    state_names=state_names
)

# Initialize model
model = RuleMultiState(state_structure=state_structure)

# Fit model (assuming you have appropriate data)
# model.fit(X, y)

# Predict state occupation probabilities
test_times = np.linspace(0, 10, 100)
state_probs = model.predict_state_occupation(X, test_times)
```

## Key Features

1. **Interpretable Models**: Rule-based approach provides transparent decision rules
2. **Flexible State Structures**: Support for standard survival, competing risks, and multi-state models
3. **Visualization Tools**: Built-in plotting functions for model interpretation
4. **Regularization**: Automatic rule selection using elastic net regularization

## Next Steps

1. Check out the [API Documentation](models.md) for detailed information about each model
2. Explore the [examples](examples/) directory for more complex use cases
3. Read about [advanced features](advanced.md) for customization options

## Getting Help

- Documentation: [https://ruletimer.readthedocs.io/](https://ruletimer.readthedocs.io/)
- GitHub Issues: [https://github.com/yourusername/ruletimer/issues](https://github.com/yourusername/ruletimer/issues)
- Email: support@ruletimer.org 