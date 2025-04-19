"""
Example of multi-state modeling with RuleTimeR
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ruletimer.data import MultiState
from ruletimer.utils import StateStructure
from ruletimer.models.multi_state import RuleMultiState
from ruletimer.visualization import plot_importance_comparison, plot_importance_heatmap, plot_state_occupation, plot_rule_importance
import seaborn as sns

def main():
    # Define the state structure
    states = [1, 2, 3, 4]  # Healthy, Mild, Moderate, Severe
    state_names = ["Healthy", "Mild", "Moderate", "Severe"]
    structure = StateStructure(
        states=states,
        transitions=[(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)],  # All possible transitions
        initial_state=1
    )
    
    # Generate synthetic data
    n_patients = 1000
    n_features = 5
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Generate base features
    X_base = pd.DataFrame(
        np.random.randn(n_patients, n_features),
        columns=feature_names
    )
    
    # Generate transition times and states
    start_times = []
    end_times = []
    start_states = []
    end_states = []
    patient_id_expanded = []
    
    for i in range(n_patients):
        current_state = 1  # Start in Healthy state
        current_time = 0
        max_time = 10
        has_transition = False
        
        while current_time < max_time:
            # Generate next transition time
            next_time = current_time + np.random.exponential(scale=2.0)
            if next_time > max_time:
                break
            
            # Determine next state based on current state
            if current_state == 1:  # Healthy
                next_state = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
            elif current_state == 2:  # Mild
                next_state = np.random.choice([3, 4], p=[0.7, 0.3])
            elif current_state == 3:  # Moderate
                next_state = 4  # Must progress to Severe
            else:  # Severe
                break  # No further transitions
            
            # Add transition record
            patient_id_expanded.append(i)
            start_times.append(current_time)
            end_times.append(next_time)
            start_states.append(current_state)
            end_states.append(next_state)
            has_transition = True
            
            # Update state and time
            current_state = next_state
            current_time = next_time
        else:
            # No more transitions for this patient
            end_times.append(max_time)
            start_states.append(current_state)
            end_states.append(0)  # Censored
            break
        
        # If no transitions occurred, ensure at least one record exists
        if not has_transition:
            if len(patient_id_expanded) == 0 or patient_id_expanded[-1] != i:
                patient_id_expanded.append(i)
                start_times.append(0)
                end_times.append(max_time)
                start_states.append(1)
                end_states.append(0)  # Censored
    
    # Convert lists to arrays
    start_times = np.array(start_times)
    end_times = np.array(end_times)
    start_states = np.array(start_states)
    end_states = np.array(end_states)
    patient_id_expanded = np.array(patient_id_expanded)
    
    # Create X by repeating rows for each patient's records
    X = pd.DataFrame(np.repeat(X_base.values, np.bincount(patient_id_expanded), axis=0),
                    columns=feature_names)
    
    # Create DataFrame with transition data
    data = pd.DataFrame({
        'patient_id': patient_id_expanded,
        'start_time': start_times,
        'end_time': end_times,
        'start_state': start_states,
        'end_state': end_states
    })

    # Create MultiState object
    multi_state = MultiState(
        patient_id=data['patient_id'].values,
        start_time=data['start_time'].values,
        end_time=data['end_time'].values,
        start_state=data['start_state'].values,
        end_state=data['end_state'].values
    )

    # Create and fit model
    model = RuleMultiState(
        max_rules=32,
        alpha=0.01,  # Reduced regularization strength
        state_structure=structure,
        max_depth=2,  # Reduced depth for simpler rules
        min_samples_leaf=10,
        n_estimators=50,
        tree_type='classification',
        tree_growing_strategy='single',  # Changed to single for simpler rules
        prune_rules=False,  # Disabled pruning
        l1_ratio=0.2,  # Reduced L1 ratio to encourage more non-zero weights
        random_state=42
    )
    
    print("\nFitting model...")
    model.fit(X, multi_state)
    
    # Debug prints
    print("\nModel attributes after fitting:")
    print(f"Has rule_weights_: {hasattr(model, 'rule_weights_')}")
    if hasattr(model, 'rule_weights_') and model.rule_weights_ is not None:
        print(f"Number of transitions with rules: {len(model.rule_weights_)}")
        for trans, weights in model.rule_weights_.items():
            print(f"Transition {trans}: {len(weights)} rules")
            if len(weights) > 0:
                print(f"First few weights: {weights[:5]}")
    else:
        print("No rules were generated. Check model parameters and data.")
    
    print("\nModel transitions:")
    print(f"States: {model.states_}")
    print(f"Transitions: {model.transitions_}")
    
    print("\nData summary:")
    for from_state, to_state in model.transitions_:
        trans_mask = (multi_state.start_state == from_state) & (multi_state.end_state == to_state)
        print(f"Transition {from_state} → {to_state}: {np.sum(trans_mask)} events")
    
    # Create plots directory if it doesn't exist
    os.makedirs('examples/plots', exist_ok=True)
    
    # Turn off interactive plotting
    plt.ioff()
    
    # Plot state occupation probabilities
    times = np.linspace(0, 10, 100)
    state_probs = model.predict_state_occupation(X, times)
    fig = plt.figure(figsize=(10, 6))
    plot_state_occupation(times, state_probs, state_names=state_names)
    plt.tight_layout()
    plt.savefig('examples/plots/state_occupation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Cumulative Incidence Functions (CIFs) for all states
    fig = plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'orange', 'red']  # Colors for each state
    
    for state in states:
        cif = model.predict_cumulative_incidence(X, times, state)
        
        # Plot mean CIF
        mean_cif = np.mean(cif, axis=0)
        plt.plot(times, mean_cif, 
                label=f'CIF for {state_names[state-1]}', 
                color=colors[state-1],
                linewidth=2)
        
        # Plot 95% confidence interval
        lower_cif = np.percentile(cif, 2.5, axis=0)
        upper_cif = np.percentile(cif, 97.5, axis=0)
        plt.fill_between(times, lower_cif, upper_cif, 
                        color=colors[state-1],
                        alpha=0.2)
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Cumulative Incidence Functions for All States')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('examples/plots/cumulative_incidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature importances if rules were generated
    if hasattr(model, 'rule_weights_') and model.rule_weights_:
        # Plot rule importance for each transition
        for from_state, to_state in model.transitions_:
            if len(model.rule_weights_[(from_state, to_state)]) > 0:
                fig = plt.figure(figsize=(10, 6))
                rules = model.rules_
                weights = model.rule_weights_[(from_state, to_state)]
                
                # Get top rules
                top_n = min(10, len(rules))
                idx = np.argsort(np.abs(weights))[-top_n:]
                top_rules = [str(rules[i]) for i in idx]
                top_weights = weights[idx]
                
                # Plot
                plt.barh(range(len(top_rules)), np.abs(top_weights))
                plt.yticks(range(len(top_rules)), top_rules)
                plt.xlabel("Absolute Weight")
                plt.title(f"Rule Importance for Transition {from_state} → {to_state}")
                plt.tight_layout()
                plt.savefig(f'examples/plots/rule_importance_{from_state}_{to_state}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot importance comparison
        fig = plt.figure(figsize=(12, 8))
        global_importance = model.get_global_importance()
        transition_importance = model.get_all_transition_importances()
        
        # Get top features
        top_n = min(5, len(feature_names))  # Reduced to 5 for better visibility
        top_idx = np.argsort(global_importance)[-top_n:]
        top_features = [feature_names[i] for i in top_idx]
        
        # Create y positions for bars
        y_pos = np.arange(top_n)
        
        # Plot global importance
        plt.barh(y_pos, global_importance[top_idx], 
                 label='Global', alpha=0.7)
        
        # Plot transition-specific importance
        for i, (trans, importance) in enumerate(transition_importance.items()):
            plt.barh(y_pos, importance[top_idx],
                     label=f"{trans[0]}→{trans[1]}", alpha=0.7,
                     left=global_importance[top_idx] if i == 0 else None)
        
        plt.yticks(y_pos, top_features)
        plt.xlabel("Importance Score")
        plt.title("Feature Importance Comparison")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('examples/plots/multi_state_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot importance heatmap
        fig = plt.figure(figsize=(12, 8))
        importance_matrix = np.zeros((len(transition_importance), top_n))
        labels = [f"{trans[0]}→{trans[1]}" for trans in transition_importance.keys()]
        
        for i, importance in enumerate(transition_importance.values()):
            importance_matrix[i] = importance[top_idx]
        
        sns.heatmap(importance_matrix, 
                    xticklabels=top_features,
                    yticklabels=labels,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'Importance Score'})
        plt.title("Feature Importance Heatmap")
        plt.tight_layout()
        plt.savefig('examples/plots/multi_state_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print mean probabilities for each state
    print("\nResults:")
    print("Mean probabilities for each state:")
    results = []
    results.append("Mean probabilities for each state:")
    for state, probs in state_probs.items():
        state_name = state_names[state-1] if state > 0 else "Censored"
        print(f"{state_name}: {np.mean(probs):.3f}")
        results.append(f"{state_name}: {np.mean(probs):.3f}")
    
    # Print feature importances if available
    if hasattr(model, 'rule_weights_') and model.rule_weights_:
        print("\nGlobal Feature Importance:")
        results.append("\nGlobal Feature Importance:")
        global_importance = model.get_global_importance()
        for feature, importance in zip(feature_names, global_importance):
            print(f"{feature}: {importance:.4f}")
            results.append(f"{feature}: {importance:.4f}")
        
        print("\nTransition-Specific Feature Importance:")
        results.append("\nTransition-Specific Feature Importance:")
        transition_importance = model.get_all_transition_importances()
        for (from_state, to_state), importance in transition_importance.items():
            print(f"\nTransition {from_state} → {to_state}:")
            results.append(f"\nTransition {from_state} → {to_state}:")
            for feature, imp in zip(feature_names, importance):
                print(f"{feature}: {imp:.4f}")
                results.append(f"{feature}: {imp:.4f}")
    
    # Save results to file
    with open('examples/multi_state_results.txt', 'w') as f:
        f.write('\n'.join(results))
    
    print("\nResults and plots have been saved to:")
    print("examples/multi_state_results.txt")
    print("examples/plots/")
    
    # Turn interactive plotting back on
    plt.ion()

if __name__ == '__main__':
    main() 