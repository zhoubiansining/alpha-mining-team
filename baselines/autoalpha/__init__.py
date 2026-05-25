"""AutoAlpha-style heuristic / beam-search formulaic alpha miner.

Inspired by Lin et al. "AutoAlpha: an Efficient Hierarchical Evolutionary
Algorithm for Mining Alpha Factors in Quantitative Investment" (2019). We
keep the spirit (search over an operator-primitive grammar, score by
in-sample IC, keep top-K) without the full hierarchical evolutionary
machinery — that lets the baseline run in seconds on a laptop and still be
qualitatively comparable to the agent pipeline's output.
"""
from baselines.autoalpha.miner import mine, get_library

__all__ = ["mine", "get_library"]
