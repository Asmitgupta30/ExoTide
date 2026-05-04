"""
module_2_tidal_rotation
========================
Rotational and Tidal Locking Analysis Module.

Provides:
  - Pseudo-synchronous and resonant rotation state derivation
  - Rossiter-McLaughlin MCMC solver (v sin i, lambda posteriors)
  - Lightcurve-based stellar rotation period extraction (starspot method)
  - Synthetic unlocked-planet RV dataset generation and analysis
  - Publication-quality tidal locking maps and RM anomaly plots
"""

from .rotation_analyser import RotationAnalyser
from .rm_solver import RMSolverMCMC
from .lightcurve_tidal import LightcurveTidalAnalyser
from .synthetic_data import SyntheticTidalDataset

__all__ = [
    "RotationAnalyser",
    "RMSolverMCMC",
    "LightcurveTidalAnalyser",
    "SyntheticTidalDataset",
]
