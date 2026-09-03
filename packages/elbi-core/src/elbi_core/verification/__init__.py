"""The deterministic verification oracle: soundness gates per claim type."""

from ._report import Check, VerificationReport
from .calibration import verify_calibration
from .classification import verify_classification
from .clustering import verify_clusters
from .comparison import verify_comparison
from .correlation import verify_correlation
from .counts import verify_counts
from .did import verify_did
from .distribution import verify_normality
from .effect import verify_effect
from .equivalence import verify_equivalence
from .experiment import verify_experiment
from .extrapolation import verify_extrapolation
from .fairness import verify_fairness
from .forecast import verify_forecast
from .glm import verify_logistic
from .groups import verify_groups
from .hazards import verify_proportional_hazards
from .iv import verify_iv
from .leakage import verify_leakage
from .missingness import verify_missingness
from .multiverse import MultiverseReport, verify_multiverse
from .overlap import verify_overlap
from .powerlaw import verify_powerlaw
from .prediction import verify_prediction
from .proportions import verify_proportions
from .rdd import verify_rdd
from .regression import verify_regression
from .reliability import verify_reliability
from .rtm import verify_rtm
from .screen import verify_screen
from .selection import CompositeReport, applicable, verify_all
from .sensitivity import verify_sensitivity
from .survival import verify_survival
from .timeseries import verify_stationarity
from .trend import verify_trend

__all__ = [
    "Check",
    "CompositeReport",
    "MultiverseReport",
    "VerificationReport",
    "applicable",
    "verify_all",
    "verify_calibration",
    "verify_classification",
    "verify_clusters",
    "verify_comparison",
    "verify_correlation",
    "verify_counts",
    "verify_did",
    "verify_effect",
    "verify_equivalence",
    "verify_experiment",
    "verify_extrapolation",
    "verify_fairness",
    "verify_forecast",
    "verify_groups",
    "verify_iv",
    "verify_leakage",
    "verify_logistic",
    "verify_missingness",
    "verify_multiverse",
    "verify_normality",
    "verify_overlap",
    "verify_powerlaw",
    "verify_prediction",
    "verify_proportional_hazards",
    "verify_proportions",
    "verify_rdd",
    "verify_regression",
    "verify_reliability",
    "verify_rtm",
    "verify_screen",
    "verify_sensitivity",
    "verify_stationarity",
    "verify_survival",
    "verify_trend",
]
