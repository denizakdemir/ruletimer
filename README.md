# RuleTimeR

RuleTimeR is a Python module implementing rule ensemble methods for time-to-event (survival) analysis, including support for competing risks and multi-state modeling. The module combines the interpretability of rule-based methods with the predictive power of ensemble techniques for time-to-event data.

## Features

- Standard survival analysis with rule ensembles
- Competing risks analysis with rule ensembles
- Multi-state modeling with rule ensembles
- Interpretable rule-based predictions
- Visualization tools for model interpretation
- Compatible with scikit-learn API

## Installation

```bash
pip install ruletimer
```

## Usage

### Standard Survival Analysis

```python
import ruletimer as rt
import pandas as pd

# Load and prepare data
data = pd.read_csv("patient_data.csv")
X = data.drop(columns=["time", "event"])
y = rt.Survival(data["time"], data["event"])

# Create and train model
model = rt.RuleSurvival(max_rules=100, alpha=0.05)
model.fit(X, y)

# Predict survival probability at specific time points
times = [6, 12, 24]  # months
survival_probs = model.predict_survival(X_new, times)

# Get rule importance
importance = model.rule_importance()
rt.plot_rule_importance(model)
```

### Competing Risks Analysis

```python
import ruletimer as rt
import pandas as pd

# Load and prepare data
data = pd.read_csv("failure_data.csv")
X = data.drop(columns=["time", "event_type"])
y = rt.CompetingRisks(data["time"], data["event_type"])

# Create and train model
model = rt.RuleCompetingRisks(max_rules=150, alpha=0.05)
model.fit(X, y)

# Predict cumulative incidence for each event type
times = [30, 60, 90]  # days
event_types = [1, 2]  # failure types
cif = model.predict_cumulative_incidence(X_new, times, event_types)

# Plot cumulative incidence functions
rt.plot_cumulative_incidence(model, X_new.iloc[0], event_types)
```

### Multi-state Modeling

```python
import ruletimer as rt
import pandas as pd

# Load and prepare data
data = pd.read_csv("disease_progression.csv")
X = data.drop(columns=["start_time", "end_time", "start_state", "end_state"])
y = rt.MultiState(data["start_time"], data["end_time"], 
                 data["start_state"], data["end_state"])

# Define state transition structure
states = ["Healthy", "Mild", "Moderate", "Severe", "Death"]
transitions = [(0,1), (1,2), (2,3), (1,3), (2,4), (3,4)]
structure = rt.StateStructure(states, transitions)

# Create and train model
model = rt.RuleMultiState(structure, max_rules=200)
model.fit(X, y)

# Predict state occupation probabilities
times = [1, 3, 5, 10]  # years
state_probs = model.predict_state_occupation(X_new, times)

# Plot state transition diagram with predicted probabilities
rt.plot_state_transitions(model, X_new.iloc[0], time=5)
```

## Documentation

For detailed documentation, including API reference and examples, please visit [documentation link].

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use RuleTimeR in your research, please cite:

[Citation information to be added] 