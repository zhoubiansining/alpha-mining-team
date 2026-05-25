"""WorldQuant Alpha101 baseline (Kakushadze 2015, arXiv:1601.00991).

Selected ~25 of the 101 formulaic alphas, re-implemented as
AlphaFactorTemplate classes consuming the {open, high, low, close, volume,
amount} DataFrame dict used by back_test/engine.py.
"""
from baselines.alpha101.factors import get_library

__all__ = ["get_library"]
