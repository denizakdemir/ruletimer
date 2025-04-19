"""
Example of survival analysis with RuleTimeR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ruletimer.models.survival import RuleSurvival
from ruletimer.data import Survival
from ruletimer.visualization import plot_rule_importance
import os

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 5

# Generate features with more meaningful relationships
X = np.zeros((n_samples, n_features))
X[:, 0] = np.random.normal(65, 10, n_samples)  # Age: strong effect
X[:, 1] = np.random.binomial(1, 0.5, n_samples)  # Gender: moderate effect
X[:, 2] = np.random.normal(25, 5, n_samples)  # BMI: weak effect
X[:, 3] = np.random.normal(120, 20, n_samples)  # Blood pressure: moderate effect
X[:, 4] = np.random.binomial(1, 0.3, n_samples)  # Smoking: very strong effect

# Generate survival times with clear feature effects
# Stronger coefficients for age and smoking
hazard = np.exp(0.01 * X[:, 0] + 0.01 * X[:, 1] + 0.005 * X[:, 2] + 0.01 * X[:, 3] + 0.02 * X[:, 4])  # Reduced coefficients
times = np.random.exponential(scale=5/hazard)  # Scale for ~5 years median survival
events = np.random.binomial(1, 0.7, n_samples)  # 70% event rate

# Create DataFrame for better visualization
feature_names = ['age', 'gender', 'bmi', 'blood_pressure', 'smoking']
X_df = pd.DataFrame(X, columns=feature_names)

# Create Survival object
y = Survival(time=times, event=events)

# Initialize and fit the model with parameters for better rule extraction
model = RuleSurvival(
    max_rules=32,
    max_depth=4,
    n_estimators=200,
    alpha=0.01,  # Reduced regularization to allow stronger feature effects
    l1_ratio=0.5,
    model_type='cox',  # Use Cox regression
    tree_type='regression'  # Use regression trees
)

print("Fitting model...")
model.fit(X, y)  # Use numpy array instead of DataFrame

# Print model parameters
print("\nModel Parameters:")
print(f"Number of rules: {len(model.rules_)}")
print(f"Rule weights range: [{min(model.rule_weights_):.3f}, {max(model.rule_weights_):.3f}]")

# Plot feature importances with better styling
plt.ioff()  # Set to non-interactive mode

# Create directory for plots if it doesn't exist
os.makedirs('examples/plots', exist_ok=True)

# Plot feature importances
fig1 = plot_rule_importance(model, figsize=(10, 6))
plt.savefig('examples/plots/feature_importances.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# Print feature importances
print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=feature_names)
print(importances.sort_values(ascending=False))

# Plot survival curves with improved visualization
sns.set_style("whitegrid")
fig2 = plt.figure(figsize=(12, 8))
eval_times = np.linspace(0, np.percentile(times, 99), 100)  # Use 99th percentile for better visualization

# Define risk groups with simpler criteria (only two conditions each)
high_risk = ((X_df['age'] >= 70) | ((X_df['smoking'] == 1) & (X_df['blood_pressure'] >= 140)))
medium_risk = ((X_df['age'].between(50, 70)) & (X_df['blood_pressure'].between(110, 140)))
low_risk = ((X_df['age'] < 50) & (X_df['blood_pressure'] < 110))

# Get predictions for each risk group
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
labels = ['High Risk', 'Medium Risk', 'Low Risk']
risk_groups = [high_risk, medium_risk, low_risk]

for group, color, label in zip(risk_groups, colors, labels):
    group_X = X_df[group]
    if len(group_X) > 0:
        predictions = model.predict_survival(group_X, eval_times)
        mean_survival = np.mean(predictions, axis=0)
        std_survival = np.std(predictions, axis=0)
        ci_lower = np.maximum(0, mean_survival - 1.96 * std_survival / np.sqrt(len(group_X)))
        ci_upper = np.minimum(1, mean_survival + 1.96 * std_survival / np.sqrt(len(group_X)))
        
        plt.plot(eval_times, mean_survival, label=f"{label} (n={len(group_X)})", 
                color=color, linewidth=2)
        plt.fill_between(eval_times, ci_lower, ci_upper, color=color, alpha=0.2)

plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('Survival Probability', fontsize=10)
plt.title('Survival Curves by Risk Group with 95% Confidence Intervals', fontsize=12, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('examples/plots/survival_curves.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# Plot 3: Distribution of survival times
fig3 = plt.figure(figsize=(12, 8))
sns.histplot(data=pd.DataFrame({'Time': times, 'Event': events}), x='Time', hue='Event', 
             multiple="stack", bins=30, palette=['#FF6B6B', '#4ECDC4'])
plt.axvline(x=np.median(times), color='black', linestyle='--', label=f'Median: {np.median(times):.1f} years')
plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Survival Times by Event Status', fontsize=12, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(title='Event', labels=['Censored', 'Event'], fontsize=10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('examples/plots/survival_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# Plot 4: Hazard rates over time
fig4 = plt.figure(figsize=(12, 8))
eval_times = np.linspace(0, np.percentile(times, 99), 100)
hazard_rates = model.predict_hazard(X, eval_times)

for group, color, label in zip(risk_groups, colors, labels):
    group_X = X[group]
    if len(group_X) > 0:
        group_hazard = model.predict_hazard(group_X, eval_times)
        mean_hazard = np.mean(group_hazard, axis=0)
        std_hazard = np.std(group_hazard, axis=0)
        ci_lower = np.maximum(0, mean_hazard - 1.96 * std_hazard / np.sqrt(len(group_X)))
        ci_upper = mean_hazard + 1.96 * std_hazard / np.sqrt(len(group_X))
        
        plt.plot(eval_times, mean_hazard, label=f"{label} (n={len(group_X)})", 
                color=color, linewidth=2)
        plt.fill_between(eval_times, ci_lower, ci_upper, color=color, alpha=0.2)

plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('Hazard Rate', fontsize=10)
plt.title('Hazard Rates by Risk Group with 95% Confidence Intervals', fontsize=12, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('examples/plots/hazard_rates.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

# Plot 5: Feature distributions by risk group
fig5, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    ax = axes[i]
    for group, color, label in zip(risk_groups, colors, labels):
        if np.unique(X[group, i]).size <= 2:  # Binary feature
            counts = pd.Series(X[group, i]).value_counts(normalize=True)
            ax.bar(counts.index + (0.2 * (i-1)), counts.values, width=0.2,
                  color=color, alpha=0.7, label=label)
        else:  # Continuous feature
            sns.kdeplot(data=X[group, i], ax=ax, color=color, label=label, fill=True, alpha=0.3)
    
    ax.set_xlabel(feature.capitalize(), fontsize=10)
    ax.set_ylabel('Density' if np.unique(X[:, i]).size > 2 else 'Proportion', fontsize=10)
    ax.set_title(f'Distribution of {feature.capitalize()} by Risk Group', fontsize=12, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove the empty subplot
axes[-1].remove()

plt.tight_layout()
plt.savefig('examples/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close(fig5)

# Plot 6: Survival probability heatmap
fig6 = plt.figure(figsize=(12, 8))
survival_probs = model.predict_survival(X, eval_times)
sorted_indices = np.argsort(np.mean(survival_probs, axis=1))
plt.imshow(survival_probs[sorted_indices], aspect='auto', cmap='YlOrRd_r',
           extent=[eval_times[0], eval_times[-1], 0, len(X)])
plt.colorbar(label='Survival Probability')
plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('Patients (sorted by risk)', fontsize=10)
plt.title('Survival Probability Heatmap', fontsize=12, pad=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('examples/plots/survival_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig6)

# Reset to interactive mode
plt.ion()

# Print group sizes
print("\nRisk Group Sizes:")
print(f"High risk: {sum(high_risk)} patients")
print(f"Medium risk: {sum(medium_risk)} patients")
print(f"Low risk: {sum(low_risk)} patients")

# Save dataset statistics
with open('examples/statistics.txt', 'w') as f:
    f.write(f"Dataset Statistics:\n")
    f.write(f"Number of samples: {n_samples}\n")
    f.write(f"Number of events: {events.sum()}\n")
    f.write(f"Censoring rate: {(1 - events.sum()/n_samples)*100:.1f}%\n")
    f.write(f"\nFollow-up Times (years):\n")
    f.write(f"Mean: {times.mean():.3f}\n")
    f.write(f"Median: {np.median(times):.3f}\n")
    f.write(f"Maximum: {times.max():.3f}\n")