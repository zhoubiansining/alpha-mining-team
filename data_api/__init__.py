"""Real-market data API for the alpha-mining pipeline.

Modules 1 and 2 of the alpha pipeline (Data IO + Featurizer base layer).
Backed by akshare; serves OHLCV bars, index constituents, and the
A-share trade calendar through a FastAPI HTTP service.
"""

from data_api.config import settings

__all__ = ["settings"]
