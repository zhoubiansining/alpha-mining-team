"""Reusable daily baseline factor libraries for integration debugging.

Expanded 20-factor library covering 5 major quantitative categories:
Momentum, Mean Reversion, Volatility, Liquidity, and Price-Volume.
"""
from __future__ import annotations
from copy import deepcopy

# ==========================================
# Category 1: Momentum & Trend (趋势动量类)
# ==========================================
MOMENTUM_20D_CODE = '''class Momentum20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return data["close"] / data["close"].shift(self.window) - 1.0
    def get_name(self) -> str: return f"Momentum_{self.window}D"
'''

MOMENTUM_5D_CODE = '''class Momentum5D:
    def __init__(self, window: int = 5, **kwargs): self.window = window
    def compute(self, data: dict): return data["close"] / data["close"].shift(self.window) - 1.0
    def get_name(self) -> str: return f"Momentum_{self.window}D"
'''

PRICE_ACCEL_CODE = '''class PriceAcceleration10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        mom1 = data["close"] / data["close"].shift(self.window) - 1.0
        mom2 = data["close"].shift(self.window) / data["close"].shift(self.window * 2) - 1.0
        return mom1 - mom2
    def get_name(self) -> str: return f"PriceAccel_{self.window}D"
'''

MA_CROSS_CODE = '''class MACrossover:
    def __init__(self, short_w: int = 5, long_w: int = 20, **kwargs): 
        self.sw, self.lw = short_w, long_w
    def compute(self, data: dict):
        ma_s = data["close"].rolling(self.sw).mean()
        ma_l = data["close"].rolling(self.lw).mean().replace(0, 1e-12)
        return ma_s / ma_l - 1.0
    def get_name(self) -> str: return f"MACross_{self.sw}_{self.lw}"
'''

# ==========================================
# Category 2: Mean Reversion (均值回归类)
# ==========================================
MEAN_REV_20D_CODE = '''class MeanReversion20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        mean = data["close"].rolling(self.window).mean()
        std = data["close"].rolling(self.window).std().replace(0, 1e-12)
        return -((data["close"] - mean) / std)
    def get_name(self) -> str: return f"MeanRev_{self.window}D"
'''

BIAS_10D_CODE = '''class Bias10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        ma = data["close"].rolling(self.window).mean().replace(0, 1e-12)
        return -(data["close"] / ma - 1.0)
    def get_name(self) -> str: return f"Bias_{self.window}D"
'''

RSI_14D_CODE = '''class RsiProxy14D:
    def __init__(self, window: int = 14, **kwargs): self.window = window
    def compute(self, data: dict):
        diff = data["close"].diff()
        up = diff.clip(lower=0).rolling(self.window).mean()
        down = (-diff.clip(upper=0)).rolling(self.window).mean().replace(0, 1e-12)
        return -(100 - (100 / (1 + up / down)))
    def get_name(self) -> str: return f"RsiProxy_{self.window}D"
'''

HIGH_52W_CODE = '''class High52WDistance:
    def __init__(self, window: int = 252, **kwargs): self.window = window
    def compute(self, data: dict):
        high_max = data["high"].rolling(self.window).max().replace(0, 1e-12)
        return data["close"] / high_max - 1.0
    def get_name(self) -> str: return f"High52W_{self.window}D"
'''

# ==========================================
# Category 3: Volatility & Risk (波动率类)
# ==========================================
AMPLITUDE_20D_CODE = '''class Amplitude20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        amp = (data["high"].rolling(self.window).max() - data["low"].rolling(self.window).min())
        return -amp / data["close"].replace(0, 1e-12)
    def get_name(self) -> str: return f"Amplitude_{self.window}D"
'''

VOLATILITY_20D_CODE = '''class Volatility20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return -data["close"].pct_change().rolling(self.window).std()
    def get_name(self) -> str: return f"Volatility_{self.window}D"
'''

DOWNSIDE_VOL_CODE = '''class DownsideVol20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        ret = data["close"].pct_change()
        return -ret.clip(upper=0).rolling(self.window).std()
    def get_name(self) -> str: return f"DownsideVol_{self.window}D"
'''

INTRADAY_VOL_CODE = '''class IntradayVol20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        ret = (data["high"] - data["low"]) / data["open"].replace(0, 1e-12)
        return -ret.rolling(self.window).mean()
    def get_name(self) -> str: return f"IntradayVol_{self.window}D"
'''

# ==========================================
# Category 4: Liquidity & Volume (流动性类)
# ==========================================
VOL_RATIO_20D_CODE = '''class VolumeRatio20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        avg = data["volume"].rolling(self.window).mean().replace(0, 1e-12)
        return data["volume"] / avg - 1.0
    def get_name(self) -> str: return f"VolumeRatio_{self.window}D"
'''

TURNOVER_20D_CODE = '''class Turnover20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return -data["volume"].rolling(self.window).mean()
    def get_name(self) -> str: return f"Turnover_{self.window}D"
'''

VOL_STD_20D_CODE = '''class VolumeStd20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        vol = data["volume"]
        return -vol.rolling(self.window).std() / vol.rolling(self.window).mean().replace(0, 1e-12)
    def get_name(self) -> str: return f"VolumeStd_{self.window}D"
'''

AMT_MOM_10D_CODE = '''class AmountMomentum10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        return data["amount"] / data["amount"].shift(self.window).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"AmtMom_{self.window}D"
'''

# ==========================================
# Category 5: Price-Volume (量价微观类)
# ==========================================
VWAP_DEV_10D_CODE = '''class VwapDeviation10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        vwap = data["amount"] / data["volume"].replace(0, 1e-12)
        dev = data["close"] / vwap.replace(0, 1e-12) - 1.0
        return -dev.rolling(self.window).mean()
    def get_name(self) -> str: return f"VwapDev_{self.window}D"
'''

PV_CORR_20D_CODE = '''class PriceVolumeCorr20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return -data["close"].rolling(self.window).corr(data["volume"])
    def get_name(self) -> str: return f"PVCorr_{self.window}D"
'''

OBV_10D_CODE = '''class ObvProxy10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        obv = np.sign(data["close"].diff()) * data["volume"]
        return obv.rolling(self.window).mean()
    def get_name(self) -> str: return f"ObvProxy_{self.window}D"
'''

AMIHUD_20D_CODE = '''class AmihudIlliquidity20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        amihud = np.abs(data["close"].pct_change()) / data["amount"].replace(0, 1e-12)
        return amihud.rolling(self.window).mean()
    def get_name(self) -> str: return f"Amihud_{self.window}D"
'''

# ==========================================
# Registration & Helpers
# ==========================================
_BASE_FACTORS = {
    "momentum_20d": {"id": "base-mom-20", "name": "Momentum 20D", "code": MOMENTUM_20D_CODE, "category": "trend",
                     "parameters": {"window": 20}, "description": "20d"},
    "momentum_5d": {"id": "base-mom-5", "name": "Momentum 5D", "code": MOMENTUM_5D_CODE, "category": "trend",
                    "parameters": {"window": 5}, "description": "5d"},
    "price_accel_10d": {"id": "base-accel-10", "name": "Price Accel 10D", "code": PRICE_ACCEL_CODE, "category": "trend",
                        "parameters": {"window": 10}, "description": "10d"},
    "ma_cross": {"id": "base-macross", "name": "MA Cross", "code": MA_CROSS_CODE, "category": "trend",
                 "parameters": {"short_w": 5, "long_w": 20}, "description": "cross"},

    "mean_reversion_20d": {"id": "base-mr-20", "name": "Mean Rev 20D", "code": MEAN_REV_20D_CODE,
                           "category": "reversion", "parameters": {"window": 20}, "description": "20d"},
    "bias_10d": {"id": "base-bias-10", "name": "Bias 10D", "code": BIAS_10D_CODE, "category": "reversion",
                 "parameters": {"window": 10}, "description": "10d"},
    "rsi_14d": {"id": "base-rsi-14", "name": "RSI Proxy 14D", "code": RSI_14D_CODE, "category": "reversion",
                "parameters": {"window": 14}, "description": "14d"},
    "high_52w": {"id": "base-high52w", "name": "High 52W", "code": HIGH_52W_CODE, "category": "reversion",
                 "parameters": {"window": 252}, "description": "52w"},

    "amplitude_20d": {"id": "base-amp-20", "name": "Amplitude 20D", "code": AMPLITUDE_20D_CODE,
                      "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "volatility_20d": {"id": "base-vol-20", "name": "Volatility 20D", "code": VOLATILITY_20D_CODE,
                       "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "downside_vol": {"id": "base-dvol-20", "name": "Downside Vol 20D", "code": DOWNSIDE_VOL_CODE,
                     "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "intraday_vol": {"id": "base-ivol-20", "name": "Intraday Vol 20D", "code": INTRADAY_VOL_CODE,
                     "category": "volatility", "parameters": {"window": 20}, "description": "20d"},

    "volume_ratio_20d": {"id": "base-volrat-20", "name": "Volume Ratio 20D", "code": VOL_RATIO_20D_CODE,
                         "category": "liquidity", "parameters": {"window": 20}, "description": "20d"},
    "turnover_20d": {"id": "base-turn-20", "name": "Turnover 20D", "code": TURNOVER_20D_CODE, "category": "liquidity",
                     "parameters": {"window": 20}, "description": "20d"},
    "volume_std": {"id": "base-volstd-20", "name": "Volume Std 20D", "code": VOL_STD_20D_CODE, "category": "liquidity",
                   "parameters": {"window": 20}, "description": "20d"},
    "amt_mom_10d": {"id": "base-amtmom-10", "name": "Amount Mom 10D", "code": AMT_MOM_10D_CODE, "category": "liquidity",
                    "parameters": {"window": 10}, "description": "10d"},

    "vwap_dev_10d": {"id": "base-vwap-10", "name": "VWAP Dev 10D", "code": VWAP_DEV_10D_CODE,
                     "category": "price_volume", "parameters": {"window": 10}, "description": "10d"},
    "pv_corr_20d": {"id": "base-pvcorr-20", "name": "PV Corr 20D", "code": PV_CORR_20D_CODE, "category": "price_volume",
                    "parameters": {"window": 20}, "description": "20d"},
    "obv_10d": {"id": "base-obv-10", "name": "OBV Proxy 10D", "code": OBV_10D_CODE, "category": "price_volume",
                "parameters": {"window": 10}, "description": "10d"},
    "amihud_20d": {"id": "base-amihud-20", "name": "Amihud 20D", "code": AMIHUD_20D_CODE, "category": "price_volume",
                   "parameters": {"window": 20}, "description": "20d"},
}


def get_base_factor(name: str) -> dict: return deepcopy(_BASE_FACTORS[name])


def get_base_factor_library(name: str = "momentum_20d") -> list[dict]: return [get_base_factor(name)]


def get_all_base_factor_libraries() -> dict[str, list[dict]]: return {k: [deepcopy(v)] for k, v in
                                                                      _BASE_FACTORS.items()}


def get_combined_base_factor_library() -> list[dict]: return [deepcopy(v) for v in _BASE_FACTORS.values()]


def list_base_factor_names() -> list[str]: return list(_BASE_FACTORS.keys())