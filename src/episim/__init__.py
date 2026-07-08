"""Core package for fast SIR and SEIR epidemic simulations."""

from .simulation import run_seir, run_sir
from .utils import summarize_simulation

__all__ = ["run_sir", "run_seir", "summarize_simulation"]
