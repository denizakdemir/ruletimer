"""
Example using the EBMT dataset to demonstrate RuleTimeR's capabilities for
multi-state modeling with competing risks at each state.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pymsm.datasets import load_ebmt
from ruletimer.models.survival import RuleSurvivalCox
from ruletimer.models.competing_risks import RuleCompetingRisks
from ruletimer.data import CompetingRisks, Survival
from ruletimer.visualization.visualization import plot_cumulative_incidence

# Load the EBMT dataset
df = load_ebmt()
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# State definitions
states = {
    1: "Alive",
    2: "Platelet Recovery",
    3: "GvHD",
    4: "Relapse",
    5: "Death",
    6: "Death"  # States 5 and 6 both represent death
}

print("\nState transitions:")
for _, row in df[['from', 'to']].drop_duplicates().sort_values(['from', 'to']).iterrows():
    print(f"From {states[row['from']]} to {states[row['to']]}")

# Example 1: Survival analysis from initial state
# Focus on transitions from state 1 (Alive) to death (states 5 or 6)
initial_state = df[df['from'] == 1].copy()
death_mask = initial_state['to'].isin([5, 6])
initial_state['status'] = death_mask.astype(int)
initial_state['time'] = initial_state['Tstop']

# Prepare features
features = ['match', 'proph', 'year', 'agecl']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse=False), features)
    ])

# Prepare features and target
X = initial_state[features].copy()
y = Survival(time=initial_state['time'], event=initial_state['status'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit preprocessor and transform features
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after transformation
feature_names = []
for feature, categories in zip(features, 
                             preprocessor.named_transformers_['cat'].categories_):
    feature_names.extend([f"{feature}_{cat}" for cat in categories[1:]])

# Fit survival model
print("\nFitting survival model for initial state transitions...")
surv_model = RuleSurvivalCox(
    max_depth=3,
    min_samples_leaf=100,
    max_rules=10
)
surv_model.fit(X_train_transformed, y_train)

# Print top rules
print("\nTop survival rules:")
for rule in surv_model.rules_[(0, 1)][:3]:
    print(f"- {rule}")

# Example 2: Competing risks analysis from Platelet Recovery
# Focus on transitions from state 2 (Platelet Recovery)
platelet_state = df[df['from'] == 2].copy()
platelet_state['time'] = platelet_state['Tstop'] - platelet_state['Tstart']

# Create event indicators for each competing risk
platelet_state['event_relapse'] = (platelet_state['to'] == 4).astype(int)
platelet_state['event_death'] = (platelet_state['to'].isin([5, 6])).astype(int)

# Prepare features and target data
X_cr = platelet_state[features].copy()
times = platelet_state['time']
# Map events to model's event types: 1 for Relapse (Event1), 2 for Death (Event2)
events = np.where(platelet_state['event_relapse'], 1,
                 np.where(platelet_state['event_death'], 2, 0))

# Print event distribution
print("\nEvent distribution in competing risks data:")
for event_type, event_name in {0: "Censored", 1: "Relapse", 2: "Death"}.items():
    count = np.sum(events == event_type)
    percentage = (count/len(events))*100
    print(f"{event_name}: {count} ({percentage:.1f}%)")

# Split data
X_train_cr, X_test_cr, times_train, times_test, events_train, events_test = train_test_split(
    X_cr, times, events, test_size=0.2, random_state=42
)

# Create CompetingRisks objects after splitting
y_train_cr = CompetingRisks(time=times_train, event=events_train)
y_test_cr = CompetingRisks(time=times_test, event=events_test)

# Transform features
X_train_cr_transformed = preprocessor.fit_transform(X_train_cr)
X_test_cr_transformed = preprocessor.transform(X_test_cr)

# Fit competing risks model
print("\nFitting competing risks model for transitions from Platelet Recovery...")
cr_model = RuleCompetingRisks(
    n_rules=10,
    min_support=0.1,
    alpha=0.5,
    l1_ratio=0.5,
    max_iter=1000,
    tol=1e-4,
    random_state=42
)

# Print event counts for each transition
print("\nEvent counts for transitions:")
for event_type in [1, 2]:  # 1: Relapse, 2: Death
    count = np.sum(events_train == event_type)
    print(f"Event {event_type}: {count} events")

# Fit the model
cr_model.fit(X_train_cr_transformed, y_train_cr)

# Print baseline hazard information
print("\nBaseline hazard information:")
for transition in cr_model.baseline_hazards_:
    times, hazards = cr_model.baseline_hazards_[transition]
    print(f"Transition {transition}:")
    print(f"  Number of time points: {len(times)}")
    print(f"  Time range: [{times.min():.1f}, {times.max():.1f}]")
    print(f"  Hazard range: [{hazards.min():.1e}, {hazards.max():.1e}]")

# Generate time points for prediction (ensure we cover the full range of times)
max_time = platelet_state['time'].max()
time_points = np.linspace(0, max_time, 100)

# Convert pandas Series to numpy arrays if needed
times_test_np = times_test.to_numpy() if hasattr(times_test, 'to_numpy') else times_test
events_test_np = events_test

# Compute Aalen-Johansen estimator for cumulative incidence functions
def compute_aalen_johansen_cif(times, events, eval_times, event_of_interest):
    n_samples = len(times)
    n_eval = len(eval_times)
    cif = np.zeros(n_eval)
    
    # Sort times and get corresponding event indicators
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_events = events[sort_idx]
    
    # Initialize at-risk process and CIF
    at_risk = np.zeros(len(sorted_times))
    for i in range(len(sorted_times)):
        at_risk[i] = np.sum(sorted_times >= sorted_times[i])
    
    # Compute Nelson-Aalen estimator for cause-specific hazard
    cause_specific_hazard = np.zeros(len(sorted_times))
    overall_hazard = np.zeros(len(sorted_times))
    
    for i in range(len(sorted_times)):
        if at_risk[i] > 0:
            # Cause-specific hazard for event of interest
            if sorted_events[i] == event_of_interest:
                cause_specific_hazard[i] = 1 / at_risk[i]
            # Overall hazard for any event
            if sorted_events[i] > 0:
                overall_hazard[i] = 1 / at_risk[i]
    
    # Compute survival function
    survival = np.exp(-np.cumsum(overall_hazard))
    
    # Compute CIF at evaluation times
    for i, t in enumerate(eval_times):
        # Find index of largest time less than or equal to t
        idx = np.searchsorted(sorted_times, t, side='right') - 1
        if idx >= 0:
            # Compute CIF using Aalen-Johansen formula
            cif[i] = np.sum(survival[:idx] * cause_specific_hazard[:idx])
    
    return cif

# Compute CIFs for test data
cif_relapse = compute_aalen_johansen_cif(times_test_np, events_test_np, time_points, 1)
cif_death = compute_aalen_johansen_cif(times_test_np, events_test_np, time_points, 2)

# Plot cumulative incidence functions
plt.figure(figsize=(10, 6))
plt.plot(time_points, cif_relapse, label='Relapse', color='blue')
plt.plot(time_points, cif_death, label='Death', color='red')

# Add confidence intervals using bootstrap
n_bootstrap = 100
cif_relapse_boot = np.zeros((n_bootstrap, len(time_points)))
cif_death_boot = np.zeros((n_bootstrap, len(time_points)))

for i in range(n_bootstrap):
    # Bootstrap sample indices
    boot_idx = np.random.choice(len(times_test_np), size=len(times_test_np), replace=True)
    
    # Compute bootstrap CIFs
    cif_relapse_boot[i] = compute_aalen_johansen_cif(
        times_test_np[boot_idx], 
        events_test_np[boot_idx],
        time_points,
        1
    )
    cif_death_boot[i] = compute_aalen_johansen_cif(
        times_test_np[boot_idx],
        events_test_np[boot_idx],
        time_points,
        2
    )

# Add confidence intervals
plt.fill_between(time_points, 
                np.percentile(cif_relapse_boot, 2.5, axis=0),
                np.percentile(cif_relapse_boot, 97.5, axis=0),
                alpha=0.2, color='blue')
plt.fill_between(time_points,
                np.percentile(cif_death_boot, 2.5, axis=0),
                np.percentile(cif_death_boot, 97.5, axis=0),
                alpha=0.2, color='red')

plt.xlabel('Time (days)')
plt.ylabel('Cumulative Incidence')
plt.title('Cumulative Incidence Functions (Aalen-Johansen Estimator)')
plt.legend()
plt.grid(True)
plt.savefig('ebmt_competing_risks.png')
plt.close()

print("\nAnalysis complete. The example demonstrates:")
print("1. Survival analysis for transitions from initial state to death")
print("2. Competing risks analysis for transitions from Platelet Recovery")
print("   - Competing events: Relapse and Death") 