"""
RuleTimeR: A Rule Ensemble-based Time-to-Event Regression Module
"""

__version__ = "0.1.0"

from .data import Survival, CompetingRisks, MultiState
from .models.survival import RuleSurvival
from .models.competing_risks import RuleCompetingRisks
from .utils import StateStructure
from .visualization import (
    plot_rule_importance,
    plot_cumulative_incidence,
    plot_state_transitions
)

__all__ = [
    "Survival",
    "CompetingRisks",
    "MultiState",
    "RuleSurvival",
    "RuleCompetingRisks",
    "StateStructure",
    "plot_rule_importance",
    "plot_cumulative_incidence",
    "plot_state_transitions"
] 