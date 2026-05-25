"""gplearn-style symbolic-regression alpha miner (panel-aware).

The upstream `gplearn` library (https://github.com/trevorstephens/gplearn)
operates on flat (n_samples × n_features) arrays — it does not know about
the (date × stock) panel structure needed for cross-sectional alpha
mining. Rather than force a square peg into a round hole, this module
mirrors gplearn's algorithmic family directly:

    tournament selection + subtree crossover + subtree / point mutation
    + hoist mutation, with multi-generation evolution

operating on a panel-aware AST (the same `Expr` from `baselines.autoalpha`
plus a `ramped_half_and_half` initializer and proper variation operators).

This avoids the gplearn install dep entirely and produces formulas that
plug directly into `AlphaFactorTemplate`.
"""
from baselines.gplearn_gp.miner import GPConfig, evolve, get_library

__all__ = ["GPConfig", "evolve", "get_library"]
