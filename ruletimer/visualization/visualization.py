"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
from ..models.base import BaseRuleEnsemble
from ..models.competing_risks import RuleCompetingRisks

def plot_rule_importance(model: BaseRuleEnsemble,
                        top_n: int = 10,
                        figsize: tuple = (10, 6)):
    """
    Plot rule importance
    
    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted model
    top_n : int, default=10
        Number of top rules to plot
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    rules = model.get_rules()
    
    # Handle different model types
    if hasattr(model, 'rule_weights_') and isinstance(model.rule_weights_, dict):
        # For competing risks models, average weights across events
        weights = np.zeros(len(rules))
        for event_weights in model.rule_weights_.values():
            weights += np.abs(event_weights)
        weights /= len(model.rule_weights_)
    else:
        # For standard survival models
        weights = model.rule_weights_
    
    # Get top rules
    idx = np.argsort(np.abs(weights))[-top_n:]
    top_rules = [rules[i] for i in idx]
    top_weights = weights[idx]
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    plt.barh(range(len(top_rules)), np.abs(top_weights))
    plt.yticks(range(len(top_rules)), top_rules)
    plt.xlabel("Absolute Weight")
    plt.title("Rule Importance")
    plt.tight_layout()
    return fig

def plot_cumulative_incidence(model: BaseRuleEnsemble,
                            X: Union[np.ndarray, pd.DataFrame],
                            event_types: List[int],
                            times: Optional[np.ndarray] = None,
                            figsize: tuple = (10, 6)):
    """
    Plot cumulative incidence functions with confidence intervals
    
    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted competing risks model
    X : array-like of shape (n_samples, n_features)
        Data to predict for
    event_types : list of int
        Event types to plot
    times : array-like, optional
        Times at which to plot cumulative incidence
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not event_types:
        raise ValueError("Event types list cannot be empty")
        
    if times is not None and len(times) == 0:
        raise ValueError("Times array cannot be empty")
        
    if times is None:
        times = np.linspace(0, model._y.time.max(), 100)
    
    # Get predictions
    cif = model.predict_cumulative_incidence(X, times, event_types)
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    
    # Define colors for different event types
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    
    for i, event_type in enumerate(event_types):
        # Calculate mean and standard deviation
        mean_cif = np.mean(cif[event_type], axis=0)
        std_cif = np.std(cif[event_type], axis=0)
        
        # Plot mean CIF
        plt.plot(times, mean_cif, 
                label=f"Event {event_type}", 
                color=colors[i % len(colors)])
        
        # Add confidence intervals
        plt.fill_between(times, 
                        mean_cif - 1.96 * std_cif,
                        mean_cif + 1.96 * std_cif,
                        alpha=0.2,
                        color=colors[i % len(colors)])
    
    plt.xlabel("Time")
    plt.ylabel("Cumulative Incidence")
    plt.title("Cumulative Incidence Functions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_state_transitions(model: BaseRuleEnsemble,
                         X: Union[np.ndarray, pd.DataFrame],
                         time: float,
                         figsize: tuple = (10, 6)) -> None:
    """
    Plot state transition diagram with probabilities
    
    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted model
    X : array-like of shape (n_samples, n_features)
        Data to predict for
    time : float
        Time at which to plot state occupation probabilities
    figsize : tuple, default=(10, 6)
        Figure size
    """
    if time < 0:
        raise ValueError("Time cannot be negative")
        
    # Get predictions
    state_occupation = model.predict_state_occupation(X, np.array([time]))
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot states
    n_states = len(model.state_structure.states)
    angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    radius = 1.0
    
    for i, state in enumerate(model.state_structure.states):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        
        # Plot state circle
        state_idx = i + 1  # Convert to 1-based indexing
        plt.plot(x, y, 'o', markersize=20,
                label=f"{state} ({state_occupation[state_idx].mean():.2f})")
        
        # Plot state name
        plt.text(x, y, state, ha='center', va='center')
    
    # Plot transitions
    for from_state, to_state in model.state_structure.transitions:
        if (from_state, to_state) in model.rule_weights_:
            # Get transition probability
            risk_scores = np.exp(np.dot(model._evaluate_rules(X),
                                      model.rule_weights_[(from_state, to_state)]))
            hazard_at_t = np.interp(time, model.baseline_hazards_[(from_state, to_state)][0],
                                  model.baseline_hazards_[(from_state, to_state)][1],
                                  left=0, right=0)
            prob = 1 - np.exp(-hazard_at_t * risk_scores.mean())
            
            # Plot arrow
            start_angle = angles[from_state - 1]  # Convert to 0-based indexing
            end_angle = angles[to_state - 1]  # Convert to 0-based indexing
            
            start_x = radius * np.cos(start_angle)
            start_y = radius * np.sin(start_angle)
            end_x = radius * np.cos(end_angle)
            end_y = radius * np.sin(end_angle)
            
            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                     head_width=0.05, head_length=0.1, fc='k', ec='k')
            
            # Plot probability
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            plt.text(mid_x, mid_y, f"{prob:.2f}", ha='center', va='center')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axis('equal')
    plt.axis('off')
    plt.title("State Transition Diagram")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_state_occupation(times: np.ndarray,
                       state_probs: Dict[int, np.ndarray],
                       state_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       alpha: float = 0.1,
                       title: str = "State Occupation Probabilities",
                       xlabel: str = "Time",
                       ylabel: str = "Probability",
                       legend_loc: str = "best") -> None:
    """
    Plot state occupation probabilities over time
    
    Parameters
    ----------
    times : array-like
        Time points
    state_probs : dict
        Dictionary mapping state indices to arrays of probabilities
    state_names : list of str, optional
        Names of states
    figsize : tuple, default=(10, 6)
        Figure size
    alpha : float, default=0.1
        Transparency of confidence intervals
    title : str, default="State Occupation Probabilities"
        Plot title
    xlabel : str, default="Time"
        X-axis label
    ylabel : str, default="Probability"
        Y-axis label
    legend_loc : str, default="best"
        Legend location
    """
    if len(state_probs) < 2:
        raise ValueError("At least two states are required to plot state occupation probabilities.")
    
    plt.figure(figsize=figsize)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    
    for i, (state_idx, probs) in enumerate(state_probs.items()):
        mean_prob = np.mean(probs, axis=0)
        std_prob = np.std(probs, axis=0)
        
        label = f"State {state_idx}" if state_names is None else state_names[state_idx - 1]
        plt.plot(times, mean_prob, label=label, color=colors[i % len(colors)])
        plt.fill_between(times,
                        mean_prob - 1.96 * std_prob,
                        mean_prob + 1.96 * std_prob,
                        alpha=alpha,
                        color=colors[i % len(colors)])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc=legend_loc)
    plt.tight_layout()

def plot_importance_comparison(model: RuleCompetingRisks,
                             top_n: int = 10,
                             figsize: tuple = (12, 8)) -> None:
    """
    Plot rule importance comparison across event types
    
    Parameters
    ----------
    model : RuleCompetingRisks
        Fitted competing risks model
    top_n : int, default=10
        Number of top rules to plot
    figsize : tuple, default=(12, 8)
        Figure size
    """
    rules = model.get_rules()
    
    # Get top rules across all event types
    all_weights = np.zeros(len(rules))
    for event_weights in model.rule_weights_.values():
        all_weights += np.abs(event_weights)
    top_idx = np.argsort(all_weights)[-top_n:]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    x = np.arange(len(top_idx))
    width = 0.8 / len(model.rule_weights_)
    
    for i, (event_type, weights) in enumerate(model.rule_weights_.items()):
        plt.bar(x + i * width,
               weights[top_idx],
               width,
               label=f"Event {event_type}")
    
    plt.xlabel("Rules")
    plt.ylabel("Weight")
    plt.title("Rule Importance by Event Type")
    plt.xticks(x + width * (len(model.rule_weights_) - 1) / 2,
               [rules[i] for i in top_idx],
               rotation=45,
               ha='right')
    plt.legend()
    plt.tight_layout()

def plot_importance_heatmap(model: RuleCompetingRisks,
                          top_n: int = 10,
                          figsize: tuple = (12, 8)) -> None:
    """
    Plot rule importance heatmap
    
    Parameters
    ----------
    model : RuleCompetingRisks
        Fitted competing risks model
    top_n : int, default=10
        Number of top rules to plot
    figsize : tuple, default=(12, 8)
        Figure size
    """
    rules = model.get_rules()
    
    # Get top rules across all event types
    all_weights = np.zeros(len(rules))
    for event_weights in model.rule_weights_.values():
        all_weights += np.abs(event_weights)
    top_idx = np.argsort(all_weights)[-top_n:]
    
    # Create weight matrix
    weight_matrix = np.zeros((len(top_idx), len(model.rule_weights_)))
    for i, (event_type, weights) in enumerate(model.rule_weights_.items()):
        weight_matrix[:, i] = weights[top_idx]
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(weight_matrix,
                xticklabels=[f"Event {k}" for k in model.rule_weights_.keys()],
                yticklabels=[rules[i] for i in top_idx],
                cmap='RdBu',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title("Rule Importance Heatmap")
    plt.tight_layout() 