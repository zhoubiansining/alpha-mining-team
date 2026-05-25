"""AlphaGen-style RL alpha miner.

Two integration paths:

1. **In-process miner** (`miner.py`): a numpy-only REINFORCE-style policy
   over AlphaGen's published operator DSL. Lightweight enough to run on
   a laptop, deliberately less powerful than the upstream PPO+transformer
   but a fair like-for-like comparison vs. the agent pipeline's iteration
   budget.

2. **Upstream adapter** (`adapter.py`): bridges back_test/data_api panel
   data to the `StockData` interface used by the official AlphaGen repo
   (https://github.com/RL-MLDM/alphagen, Yu et al., IJCAI 2023), so that
   users with the full PyTorch/Qlib stack can run real PPO training and
   then export discovered formulas through `expressions_to_library()`.
"""
from baselines.alphagen.miner import get_library, train_policy
from baselines.alphagen.adapter import (
    panel_to_alphagen_arrays,
    expressions_to_library,
)

__all__ = [
    "get_library",
    "train_policy",
    "panel_to_alphagen_arrays",
    "expressions_to_library",
]
