# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import RuleTimeR modules
from ruletimer.data import Survival, CompetingRisks, MultiState
from ruletimer.models import RuleSurvivalCox, RuleCompetingRisks, RuleMultiState
from ruletimer.utils import StateStructure
from ruletimer.visualization import (
    plot_rule_importance,
    plot_cumulative_incidence,
    plot_state_transitions,
    plot_state_occupation
)

# ---- Data Loading and Preprocessing ----

# Download the PBC dataset (primary biliary cirrhosis trial from Mayo Clinic)
print("Downloading PBC dataset...")
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv"
response = requests.get(url)
pbc_data = pd.read_csv(StringIO(response.text))

# Basic data cleaning
if 'Unnamed: 0' in pbc_data.columns:
    pbc_data = pbc_data.drop(columns=['Unnamed: 0'])  # Drop index column if it exists
pbc_data = pbc_data.rename(columns=lambda x: x.lower())  # Convert column names to lowercase

print(f"Dataset loaded with {pbc_data.shape[0]} patients and {pbc_data.shape[1]} variables")
print("Data sample:", pbc_data.head(3))

# Convert categorical variables
if 'sex' in pbc_data.columns:
    pbc_data['sex'] = pbc_data['sex'].map({'F': 0, 'M': 1})
if 'stage' in pbc_data.columns:
    # Fill missing values in stage with median
    pbc_data['stage'] = pbc_data['stage'].fillna(pbc_data['stage'].median())
    pbc_data['stage'] = pbc_data['stage'].astype(int)

# Prepare data for analysis
feature_cols = [col for col in pbc_data.columns 
                if col not in ['id', 'time', 'status', 'days']]

# For PBC dataset, status: 0=alive, 1=liver transplant, 2=dead
X = pbc_data[feature_cols].values
time = pbc_data['time'].values
event = (pbc_data['status'] == 2).astype(int)  # 1 if dead, 0 otherwise

# Create train/test split
X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
    X, time, event, test_size=0.3, random_state=42
)

# Handle missing values and scale features
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# First impute missing values
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Then scale the features
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Create Survival data objects
y_train = Survival(time=time_train, event=event_train)
y_test = Survival(time=time_test, event=event_test)

print(f"Data prepared: {len(y_train)} training samples, {len(y_test)} test samples")
print(f"Event rate: {event.mean():.2f}")

# ---- PART 1: STANDARD SURVIVAL ANALYSIS ----

print("\n===== STANDARD SURVIVAL ANALYSIS =====")

# Initialize and fit the survival model
survival_model = RuleSurvivalCox(
    max_rules=50,
    alpha=0.1,
    random_state=42
)

print("Fitting survival model...")
survival_model.fit(X_train_scaled, y_train)

# Make predictions
prediction_times = np.linspace(0, 4000, 100)  # 4000 days is approximately 11 years
survival_probs = survival_model.predict_survival(X_test_scaled, prediction_times)

# Display top rules
print("Top rules for survival prediction:")
for i, (rule, importance, transition) in enumerate(survival_model.get_top_rules(5)):
    print(f"Rule {i+1} (transition {transition}, importance={importance:.3f}): {rule}")

# Calculate rule importances
print("Calculating rule importances...")
survival_model._compute_feature_importances()

# Plot rule importance
fig = plot_rule_importance(survival_model, top_n=10)
plt.savefig('survival_rule_importance.png')
plt.close()

# Plot survival curves
plt.figure(figsize=(10, 6))
for i in range(min(10, len(X_test_scaled))):
    plt.plot(prediction_times, survival_probs[i], alpha=0.5)
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.title('Predicted Survival Curves for 10 Patients')
plt.grid(True, alpha=0.3)
plt.savefig('survival_curves.png')
plt.close()

# ---- PART 2: COMPETING RISKS ANALYSIS ----

print("\n===== COMPETING RISKS ANALYSIS =====")

# For competing risks, we treat both death and transplant as events
competing_status = pbc_data['status'].values  # 0=censored, 1=transplant, 2=death

# Create train/test split for competing risks
_, _, _, _, competing_status_train, competing_status_test = train_test_split(
    X, time, competing_status, test_size=0.3, random_state=42
)

# Create CompetingRisks data objects
y_train_cr = CompetingRisks(time=time_train, event=competing_status_train)
y_test_cr = CompetingRisks(time=time_test, event=competing_status_test)

# Count event types
print("Event counts in training data:")
for event_type in np.unique(competing_status_train):
    count = np.sum(competing_status_train == event_type)
    percent = count / len(competing_status_train) * 100
    event_name = "Censored" if event_type == 0 else "Transplant" if event_type == 1 else "Death"
    print(f"  {event_name}: {count} ({percent:.1f}%)")

# Initialize and fit the competing risks model
cr_model = RuleCompetingRisks(
    n_rules=50,
    alpha=0.1,
    random_state=42
)

print("Fitting competing risks model...")
cr_model.fit(X_train_scaled, y_train_cr)

# Make predictions
cr_times = np.linspace(0, 4000, 100)

# Predict cumulative incidence for each event type
cif_transplant = cr_model.predict_cumulative_incidence(X_test_scaled, cr_times, event_type="Event1")
cif_death = cr_model.predict_cumulative_incidence(X_test_scaled, cr_times, event_type="Event2")

# Calculate average cumulative incidence
avg_cif_transplant = np.mean(cif_transplant, axis=0)
avg_cif_death = np.mean(cif_death, axis=0)

# Plot cumulative incidence functions
plt.figure(figsize=(10, 6))
plt.plot(cr_times, avg_cif_transplant, 'b-', label='Transplant')
plt.plot(cr_times, avg_cif_death, 'r-', label='Death')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Incidence')
plt.title('Average Cumulative Incidence Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('competing_risks_cif.png')
plt.close()

# Plot with package's visualization function
fig = plot_cumulative_incidence(
    cr_model,
    X_test_scaled[:10],  # Use first 10 patients for clarity
    event_types=[1, 2],  # 1=transplant, 2=death
    times=cr_times
)
plt.savefig('competing_risks_visualization.png')
plt.close()

# ---- PART 3: MULTI-STATE MODELING ----

print("\n===== MULTI-STATE MODELING =====")

# For multi-state modeling, we'll use disease stages as states
if 'stage' in pbc_data.columns:
    print("Disease stage distribution:")
    print(pbc_data['stage'].value_counts().sort_index())

    # Define state structure (using disease stages plus death)
    states = [1, 2, 3, 4, 5]  # 1-4 are disease stages, 5 is death
    state_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Death"]
    
    # Define possible transitions
    transitions = [
        (1, 2), (2, 3), (3, 4),  # Disease progression
        (1, 5), (2, 5), (3, 5), (4, 5)  # Death from any stage
    ]
    
    state_structure = StateStructure(
        states=states,
        transitions=transitions,
        state_names=state_names
    )

    # Create synthetic transition data for demonstration
    # Note: In practice, you would have actual longitudinal observations of state transitions
    print("\nCreating synthetic transition data for demonstration purposes...")
    
    transition_data = {}
    
    for transition in transitions:
        from_stage, to_stage = transition
        
        # Select patients in the from_stage
        if from_stage <= 4:
            patient_mask = pbc_data['stage'] == from_stage
            n_patients_in_stage = patient_mask.sum()
            
            if n_patients_in_stage > 0:
                # For disease progression (synthetic)
                if to_stage <= 4:
                    times = np.random.exponential(1000, size=n_patients_in_stage)
                    events = np.random.binomial(1, 0.7, size=n_patients_in_stage)
                # For death (use actual death data)
                else:
                    patients_from_stage = pbc_data[patient_mask]
                    times = patients_from_stage['time'].values
                    events = (patients_from_stage['status'] == 2).astype(int)
                
                transition_data[transition] = {
                    'times': times,
                    'events': events
                }
                print(f"  Transition {from_stage}→{to_stage}: {n_patients_in_stage} patients, {events.sum()} events")
    
    # Create feature matrix for multi-state model
    X_ms = scaler.transform(imputer.transform(pbc_data[feature_cols].values))
    
    # Initialize and fit the multi-state model
    ms_model = RuleMultiState(
        state_structure=state_structure,
        max_rules=30,
        alpha=0.1,
        random_state=42
    )
    
    print("Fitting multi-state model...")
    ms_model.fit(X_ms, transition_data)
    
    # Make predictions
    ms_times = np.linspace(0, 4000, 50)
    
    # Predict state occupation probabilities (starting from stage 1)
    state_probs = ms_model.predict_state_occupation(X_ms[:5], ms_times, initial_state=1)
    
    # Plot state occupation probabilities
    plt.figure(figsize=(12, 8))
    plot_state_occupation(
        ms_times,
        state_probs,
        state_names=state_names,
        title='State Occupation Probabilities (Starting from Stage 1)'
    )
    plt.savefig('multistate_occupation.png')
    plt.close()
    
    # Plot state transitions for a single patient
    plt.figure(figsize=(12, 8))
    plot_state_transitions(
        ms_model,
        X_ms[:1],
        time=1000,  # At time 1000 days
        initial_state=1
    )
    plt.savefig('multistate_transitions.png')
    plt.close()

# ---- PART 4: FEATURE IMPORTANCE ANALYSIS ----

print("\n===== FEATURE IMPORTANCE ANALYSIS =====")

# Extract feature importances for survival model
if hasattr(survival_model, 'feature_importances_'):
    feature_importance_survival = survival_model.feature_importances_
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance_survival
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Feature Importances for Survival Model')
    plt.tight_layout()
    plt.savefig('feature_importance_survival.png')
    plt.close()
    
    print("Top 5 features for survival prediction:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Extract feature importances for competing risks model
for event_type in [1, 2]:  # 1=transplant, 2=death
    try:
        importance_cr = cr_model.get_feature_importances(event_type)
        
        # Create DataFrame for visualization
        importance_df_cr = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_cr
        }).sort_values('Importance', ascending=False)
        
        event_name = "Transplant" if event_type == 1 else "Death"
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df_cr.head(10))
        plt.title(f'Top 10 Feature Importances for {event_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_cr_{event_type}.png')
        plt.close()
        
        print(f"\nTop 5 features for {event_name}:")
        for i, row in importance_df_cr.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    except Exception as e:
        print(f"Could not retrieve feature importances for event type {event_type}: {e}")

# ---- SUMMARY ----

print("\n===== SUMMARY =====")
print("Analysis complete. The following visualizations were generated:")
print("1. survival_curves.png - Survival probability curves for individual patients")
print("2. survival_rule_importance.png - Top rules identified for survival prediction")
print("3. competing_risks_cif.png - Cumulative incidence functions for transplant and death")
print("4. competing_risks_visualization.png - Detailed visualization of competing risks")
if 'stage' in pbc_data.columns:
    print("5. multistate_occupation.png - State occupation probabilities over time")
    print("6. multistate_transitions.png - Transition diagram with probabilities")
print("7. feature_importance_survival.png - Feature importance ranking for survival model")
print("8. feature_importance_cr_1.png - Feature importance ranking for transplant")
print("9. feature_importance_cr_2.png - Feature importance ranking for death")