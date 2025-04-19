"""
Example of competing risks analysis with RuleTimeR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler
from ruletimer import RuleCompetingRisks, CompetingRisks
from ruletimer.visualization import plot_cumulative_incidence, plot_rule_importance, plot_importance_comparison, plot_importance_heatmap
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
n_features = 5

# Generate correlated features
age = np.random.normal(50, 10, n_samples)
bmi = 25 + 0.1 * age + np.random.normal(0, 3, n_samples)
blood_pressure = 120 + 0.3 * age + np.random.normal(0, 10, n_samples)
gender = np.random.binomial(1, 0.5, n_samples)
smoking = np.random.binomial(1, 0.3, n_samples)

# Define feature names
feature_names = ['Age', 'BMI', 'Blood Pressure', 'Gender', 'Smoking']

# Standardize features
X = np.column_stack([age, bmi, blood_pressure, gender, smoking])
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Generate event times using Weibull distribution
# Parameters for cardiovascular events
shape_cardio = 1.2
scale_cardio = 20.0
# Parameters for cancer events
shape_cancer = 1.1
scale_cancer = 25.0

# Calculate baseline hazards
def weibull_hazard(t, shape, scale):
    return (shape / scale) * (t / scale) ** (shape - 1)

# Generate event times
cardio_coef = np.array([0.15, 0.1, 0.1, 0.0, 0.05])
cancer_coef = np.array([0.1, 0.0, 0.0, 0.05, 0.15])

# Calculate risk scores
cardio_risk = np.exp(np.dot(X, cardio_coef))
cancer_risk = np.exp(np.dot(X, cancer_coef))

# Generate event times
u1 = np.random.uniform(0, 1, n_samples)
u2 = np.random.uniform(0, 1, n_samples)

t1 = weibull_min.ppf(1 - np.exp(-u1 * cardio_risk), shape_cardio, scale=scale_cardio)
t2 = weibull_min.ppf(1 - np.exp(-u2 * cancer_risk), shape_cancer, scale=scale_cancer)

# Generate censoring times
c_times = np.random.uniform(0, 30, n_samples)

# Determine event times and types
time = np.minimum(np.minimum(t1, t2), c_times)
event = np.zeros(n_samples)
event[t1 < t2] = 1
event[t2 < t1] = 2
event[c_times < np.minimum(t1, t2)] = 0

# Create and fit the model
model = RuleCompetingRisks(
    max_rules=16,
    max_depth=3,
    n_estimators=100
)

# Create competing risks data object
y = CompetingRisks(time, event)
model.fit(X, y)

# Calculate cumulative incidence
time_points = np.linspace(0, 30, 100)
cif = model.predict_cumulative_incidence(X, time_points, event_types=[1, 2])

# Print summary statistics
results = []
results.append("Data Summary:")
results.append(f"Total samples: {n_samples}")
results.append(f"Event 1 (Cardiovascular) count: {np.sum(event == 1)}")
results.append(f"Event 2 (Cancer) count: {np.sum(event == 2)}")
results.append(f"Censored count: {np.sum(event == 0)}")
results.append(f"Censoring rate: {np.sum(event == 0) / n_samples:.3f}\n")

results.append("Time summary:")
results.append(f"Mean follow-up time: {np.mean(time):.3f}")
results.append(f"Median follow-up time: {np.median(time):.3f}")
results.append(f"Max follow-up time: {np.max(time):.3f}\n")

results.append("Model Results:")
results.append("Feature descriptions:")
results.append("Feature 0: Age (standardized)")
results.append("Feature 1: BMI (correlated with age)")
results.append("Feature 2: Blood pressure")
results.append("Feature 3: Gender (binary)")
results.append("Feature 4: Smoking status (binary)\n")

results.append(f"Feature importances: {model.feature_importances_}\n")

results.append("Cumulative incidence summary:\n")
for event_type, cif_values in cif.items():
    results.append(f"Event type {event_type} ({'Cardiovascular' if event_type == 1 else 'Cancer'}):")
    results.append("Mean cumulative incidence at different times:")
    for i, t in enumerate(time_points[::10]):
        results.append(f"Time {t:.1f}: {np.mean(cif_values[::10], axis=0)[i]:.3f}")
    results.append("")

results.append("Model evaluation:")
results.append(f"Number of rules: {len(model.rules_)}")
results.append(f"Test set censoring rate: {np.sum(event == 0) / len(event):.3f}\n")

results.append("Weights for Cardiovascular Events:")
results.append(str(model.rule_weights_[1]))
results.append("\nWeights for Cancer Events:")
results.append(str(model.rule_weights_[2]))

results.append("\nRule descriptions for Cardiovascular Events:")
for i, (rule, weight) in enumerate(zip(model.rules_, model.rule_weights_[1]), 1):
    if weight != 0:
        results.append(f"Rule {i}: {rule} (weight: {weight:.4f})")

results.append("\nRule descriptions for Cancer Events:")
for i, (rule, weight) in enumerate(zip(model.rules_, model.rule_weights_[2]), 1):
    if weight != 0:
        results.append(f"Rule {i}: {rule} (weight: {weight:.4f})")

# Calculate risk scores for each event type
risk_scores = {}
for event_type in model.event_types_:
    # First evaluate rules on the data
    rule_values = model._evaluate_rules(X)
    # Then calculate risk scores using rule values and weights
    risk_scores[event_type] = np.exp(np.dot(rule_values, model.rule_weights_[event_type]))

# Print results
print("\nResults:")
print(f"Number of rules: {len(model.rules_)}")
print(f"Rule weights for event 1: {model.rule_weights_[1]}")
print(f"Rule weights for event 2: {model.rule_weights_[2]}")
print(f"Risk scores for event 1: {risk_scores[1][:5]}")  # Print first 5 risk scores
print(f"Risk scores for event 2: {risk_scores[2][:5]}")  # Print first 5 risk scores

# Save results to file
with open('examples/competing_risks_results.txt', 'w') as f:
    f.write(f"Number of rules: {len(model.rules_)}\n")
    f.write(f"Rule weights for event 1: {model.rule_weights_[1]}\n")
    f.write(f"Rule weights for event 2: {model.rule_weights_[2]}\n")
    f.write(f"Risk scores for event 1: {risk_scores[1][:5]}\n")
    f.write(f"Risk scores for event 2: {risk_scores[2][:5]}\n")

print("\nResults and plots have been saved to:")
print("examples/competing_risks_results.txt")
print("examples/plots/")

# Plot cumulative incidence
sns.set_style("whitegrid")
plt.ioff()  # Set to non-interactive mode

# Create directory for plots if it doesn't exist
os.makedirs('examples/plots', exist_ok=True)

# Plot cumulative incidence
fig1 = plt.figure(figsize=(12, 8))
plot_cumulative_incidence(model, X, event_types=[1, 2])
plt.savefig('examples/plots/cumulative_incidence.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# Plot feature importances
fig2 = plot_rule_importance(model, figsize=(10, 6))
plt.savefig('examples/plots/competing_risks_feature_importances.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# Plot comparison of global and event-specific importance
fig3 = plt.figure(figsize=(12, 8))
plot_importance_comparison(model, top_n=10)
plt.savefig('examples/plots/competing_risks_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# Plot importance heatmap
fig4 = plt.figure(figsize=(12, 8))
plot_importance_heatmap(model, top_n=10)
plt.savefig('examples/plots/competing_risks_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

# Plot 5: Event-specific hazard rates
fig5 = plt.figure(figsize=(12, 8))
hazard_rates = model.predict_hazard(X, time_points)
colors = ['#FF6B6B', '#4ECDC4']  # Red for cardiovascular, teal for cancer

for event_type, color in zip([1, 2], colors):
    mean_hazard = np.mean(hazard_rates[event_type], axis=0)
    std_hazard = np.std(hazard_rates[event_type], axis=0)
    ci_lower = np.maximum(0, mean_hazard - 1.96 * std_hazard / np.sqrt(n_samples))
    ci_upper = mean_hazard + 1.96 * std_hazard / np.sqrt(n_samples)
    
    plt.plot(time_points, mean_hazard, 
             label=f"{'Cardiovascular' if event_type == 1 else 'Cancer'} (n={np.sum(event == event_type)})",
             color=color, linewidth=2)
    plt.fill_between(time_points, ci_lower, ci_upper, color=color, alpha=0.2)

plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('Hazard Rate', fontsize=10)
plt.title('Event-Specific Hazard Rates with 95% Confidence Intervals', fontsize=12, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('examples/plots/event_specific_hazards.png', dpi=300, bbox_inches='tight')
plt.close(fig5)

# Plot 6: Risk group stratification
fig6 = plt.figure(figsize=(12, 8))
# Use the risk scores we already calculated earlier
# Define risk groups based on combined risk scores
combined_risk = risk_scores[1] + risk_scores[2]
risk_quantiles = np.percentile(combined_risk, [33, 66])
high_risk = combined_risk > risk_quantiles[1]
medium_risk = (combined_risk > risk_quantiles[0]) & (combined_risk <= risk_quantiles[1])
low_risk = combined_risk <= risk_quantiles[0]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
labels = ['High Risk', 'Medium Risk', 'Low Risk']
risk_groups = [high_risk, medium_risk, low_risk]

for event_type, color in zip([1, 2], ['#FF6B6B', '#4ECDC4']):
    for group, group_color, label in zip(risk_groups, colors, labels):
        group_X = X[group]
        if len(group_X) > 0:
            group_cif = model.predict_cumulative_incidence(group_X, time_points, [event_type])[event_type]
            mean_cif = np.mean(group_cif, axis=0)
            std_cif = np.std(group_cif, axis=0)
            ci_lower = np.maximum(0, mean_cif - 1.96 * std_cif / np.sqrt(len(group_X)))
            ci_upper = np.minimum(1, mean_cif + 1.96 * std_cif / np.sqrt(len(group_X)))
            
            plt.plot(time_points, mean_cif, 
                    label=f"{'Cardiovascular' if event_type == 1 else 'Cancer'} - {label} (n={len(group_X)})",
                    color=group_color, linestyle='-' if event_type == 1 else '--', linewidth=2)
            plt.fill_between(time_points, ci_lower, ci_upper, color=group_color, alpha=0.2)

plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('Cumulative Incidence', fontsize=10)
plt.title('Cumulative Incidence by Risk Group and Event Type', fontsize=12, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('examples/plots/risk_group_stratification.png', dpi=300, bbox_inches='tight')
plt.close(fig6)

# Plot 7: Feature distributions by event type
fig7, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    ax = axes[i]
    for event_type, color in zip([0, 1, 2], ['#45B7D1', '#FF6B6B', '#4ECDC4']):
        mask = event == event_type
        if np.unique(X[mask, i]).size <= 2:  # Binary feature
            counts = pd.Series(X[mask, i]).value_counts(normalize=True)
            ax.bar(counts.index + (0.2 * (event_type-1)), counts.values, width=0.2,
                  color=color, alpha=0.7, label=f"{'Censored' if event_type == 0 else 'Cardiovascular' if event_type == 1 else 'Cancer'}")
        else:  # Continuous feature
            sns.kdeplot(data=X[mask, i], ax=ax, color=color, 
                       label=f"{'Censored' if event_type == 0 else 'Cardiovascular' if event_type == 1 else 'Cancer'}",
                       fill=True, alpha=0.3)
    
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('Density' if np.unique(X[:, i]).size > 2 else 'Proportion', fontsize=10)
    ax.set_title(f'Distribution of {feature} by Event Type', fontsize=12, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove the empty subplot
axes[-1].remove()

plt.tight_layout()
plt.savefig('examples/plots/feature_distributions_by_event.png', dpi=300, bbox_inches='tight')
plt.close(fig7)

# Plot 8: Risk score distributions
fig8, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (event_type, color) in enumerate(zip([1, 2], ['#FF6B6B', '#4ECDC4'])):
    ax = axes[i]
    # Use the risk scores we already calculated
    event_risk_scores = risk_scores[event_type]
    
    for event_status, linestyle in zip([0, 1, 2], ['-', '--', ':']):
        mask = event == event_status
        if mask.sum() > 0:
            sns.kdeplot(data=event_risk_scores[mask], ax=ax, color=color, 
                       label=f"{'Censored' if event_status == 0 else 'Cardiovascular' if event_status == 1 else 'Cancer'}",
                       linestyle=linestyle, linewidth=2)
    
    ax.set_xlabel('Risk Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f"{'Cardiovascular' if event_type == 1 else 'Cancer'} Risk Score Distribution", fontsize=12, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('examples/plots/risk_score_distributions.png', dpi=300, bbox_inches='tight')
plt.close(fig8)

# Reset to interactive mode
plt.ion()

# Print importance measures
print("\nGlobal Feature Importance:")
for i, (feature, importance) in enumerate(zip(feature_names, model.get_global_importance())):
    print(f"{feature}: {importance:.4f}")

print("\nEvent-Specific Feature Importance:")
for event_type in model.event_types_:
    print(f"\nEvent {event_type} ({'Cardiovascular' if event_type == 1 else 'Cancer'}):")
    event_importance = model.get_event_specific_importance(event_type)
    for i, (feature, importance) in enumerate(zip(feature_names, event_importance)):
        print(f"{feature}: {importance:.4f}") 