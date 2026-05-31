"""Reusable daily baseline factor libraries for integration debugging.

Massive 50-factor library covering 5 major quantitative categories
(10 factors each): Trend/Momentum, Mean Reversion, Volatility, Liquidity, and Price-Volume.
"""
from __future__ import annotations
from copy import deepcopy

# ==========================================
# Category 1: Momentum & Trend (趋势动量类 1-10)
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
    def __init__(self, short_w: int = 5, long_w: int = 20, **kwargs): self.sw, self.lw = short_w, long_w
    def compute(self, data: dict):
        return data["close"].rolling(self.sw).mean() / data["close"].rolling(self.lw).mean().replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"MACross_{self.sw}_{self.lw}"
'''
MOMENTUM_60D_CODE = '''class Momentum60D:
    def __init__(self, window: int = 60, **kwargs): self.window = window
    def compute(self, data: dict): return data["close"] / data["close"].shift(self.window) - 1.0
    def get_name(self) -> str: return f"Momentum_{self.window}D"
'''
MOMENTUM_120D_CODE = '''class Momentum120D:
    def __init__(self, window: int = 120, **kwargs): self.window = window
    def compute(self, data: dict): return data["close"] / data["close"].shift(self.window) - 1.0
    def get_name(self) -> str: return f"Momentum_{self.window}D"
'''
OVERNIGHT_MOM_CODE = '''class OvernightMom:
    def __init__(self, window: int = 1, **kwargs): self.window = window
    def compute(self, data: dict): return data["open"] / data["close"].shift(self.window).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return "OvernightMom"
'''
INTRADAY_MOM_CODE = '''class IntradayMom:
    def __init__(self, window: int = 1, **kwargs): self.window = window
    def compute(self, data: dict): return data["close"] / data["open"].replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return "IntradayMom"
'''
EMA_CROSS_CODE = '''class EmaCrossover:
    def __init__(self, short_w: int = 10, long_w: int = 30, **kwargs): self.sw, self.lw = short_w, long_w
    def compute(self, data: dict):
        return data["close"].ewm(span=self.sw).mean() / data["close"].ewm(span=self.lw).mean().replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"EmaCross_{self.sw}_{self.lw}"
'''
MACD_PROXY_CODE = '''class MacdProxy:
    def __init__(self, short_w: int = 12, long_w: int = 26, **kwargs): self.sw, self.lw = short_w, long_w
    def compute(self, data: dict):
        return data["close"].ewm(span=self.sw).mean() - data["close"].ewm(span=self.lw).mean()
    def get_name(self) -> str: return f"MacdProxy_{self.sw}_{self.lw}"
'''

# ==========================================
# Category 2: Mean Reversion (均值回归类 11-20)
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
        return -(data["close"] / data["close"].rolling(self.window).mean().replace(0, 1e-12) - 1.0)
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
        return data["close"] / data["high"].rolling(self.window).max().replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"High52W_{self.window}D"
'''
SHORT_REV_1D_CODE = '''class ShortRev1D:
    def __init__(self, window: int = 1, **kwargs): self.window = window
    def compute(self, data: dict): return -data["close"].pct_change(1)
    def get_name(self) -> str: return "ShortRev_1D"
'''
SHORT_REV_3D_CODE = '''class ShortRev3D:
    def __init__(self, window: int = 3, **kwargs): self.window = window
    def compute(self, data: dict): return -data["close"].pct_change(3)
    def get_name(self) -> str: return "ShortRev_3D"
'''
DISPARITY_5D_CODE = '''class Disparity5D:
    def __init__(self, window: int = 5, **kwargs): self.window = window
    def compute(self, data: dict):
        return -(data["close"] / data["close"].rolling(self.window).mean().replace(0, 1e-12) - 1.0)
    def get_name(self) -> str: return f"Disparity_{self.window}D"
'''
BOLL_PCT_B_CODE = '''class BollPctB:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        ma = data['close'].rolling(self.window).mean()
        std = data['close'].rolling(self.window).std()
        up = ma + 2 * std
        dn = ma - 2 * std
        return -(data['close'] - dn) / (up - dn).replace(0, 1e-12)
    def get_name(self) -> str: return f"BollPctB_{self.window}D"
'''
STOCH_K_CODE = '''class StochKProxy:
    def __init__(self, window: int = 14, **kwargs): self.window = window
    def compute(self, data: dict):
        low_min = data['low'].rolling(self.window).min()
        high_max = data['high'].rolling(self.window).max()
        return -(data['close'] - low_min) / (high_max - low_min).replace(0, 1e-12)
    def get_name(self) -> str: return f"StochK_{self.window}D"
'''
WILLIAMS_R_CODE = '''class WilliamsRProxy:
    def __init__(self, window: int = 14, **kwargs): self.window = window
    def compute(self, data: dict):
        low_min = data['low'].rolling(self.window).min()
        high_max = data['high'].rolling(self.window).max()
        return (high_max - data['close']) / (high_max - low_min).replace(0, 1e-12)
    def get_name(self) -> str: return f"WilliamsR_{self.window}D"
'''

# ==========================================
# Category 3: Volatility & Risk (波动率类 21-30)
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
    def compute(self, data: dict): return -data["close"].pct_change().rolling(self.window).std()
    def get_name(self) -> str: return f"Volatility_{self.window}D"
'''
DOWNSIDE_VOL_CODE = '''class DownsideVol20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return -data["close"].pct_change().clip(upper=0).rolling(self.window).std()
    def get_name(self) -> str: return f"DownsideVol_{self.window}D"
'''
INTRADAY_VOL_CODE = '''class IntradayVol20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return -((data["high"] - data["low"]) / data["open"].replace(0, 1e-12)).rolling(self.window).mean()
    def get_name(self) -> str: return f"IntradayVol_{self.window}D"
'''
PARKINSON_VOL_CODE = '''class ParkinsonVol:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        return -(np.log(data['high'] / data['low'].replace(0, 1e-12))**2).rolling(self.window).mean()
    def get_name(self) -> str: return f"ParkinsonVol_{self.window}D"
'''
GARMAN_KLASS_VOL_CODE = '''class GarmanKlassVol:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        co = np.log(data['close'] / data['open'].replace(0, 1e-12))**2
        hl = np.log(data['high'] / data['low'].replace(0, 1e-12))**2
        return -(0.5 * hl - (2 * np.log(2) - 1) * co).rolling(self.window).mean()
    def get_name(self) -> str: return f"GarmanKlass_{self.window}D"
'''
VOL_OF_VOL_CODE = '''class VolOfVol:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return -data['close'].pct_change().rolling(self.window).std().rolling(self.window).std()
    def get_name(self) -> str: return f"VolOfVol_{self.window}D"
'''
UPSIDE_VOL_CODE = '''class UpsideVol:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return -data['close'].pct_change().clip(lower=0).rolling(self.window).std()
    def get_name(self) -> str: return f"UpsideVol_{self.window}D"
'''
HL_SPREAD_CODE = '''class HighLowSpread:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        return -((data['high'] - data['low']) / data['close'].replace(0, 1e-12)).rolling(self.window).mean()
    def get_name(self) -> str: return f"HLSpread_{self.window}D"
'''
OC_SPREAD_CODE = '''class OpenCloseSpread:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        return -(np.abs(data['open'] - data['close']) / data['close'].replace(0, 1e-12)).rolling(self.window).mean()
    def get_name(self) -> str: return f"OCSpread_{self.window}D"
'''

# ==========================================
# Category 4: Liquidity & Volume (流动性类 31-40)
# ==========================================
VOL_RATIO_20D_CODE = '''class VolumeRatio20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        return data["volume"] / data["volume"].rolling(self.window).mean().replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"VolumeRatio_{self.window}D"
'''
TURNOVER_20D_CODE = '''class Turnover20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return -data["volume"].rolling(self.window).mean()
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
    def compute(self, data: dict): return data["amount"] / data["amount"].shift(self.window).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"AmtMom_{self.window}D"
'''
TURNOVER_VOL_CODE = '''class TurnoverVolatility:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return -data['volume'].rolling(self.window).std()
    def get_name(self) -> str: return f"TurnoverVol_{self.window}D"
'''
VOL_TREND_CODE = '''class VolumeTrend:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return data['volume'] / data['volume'].shift(self.window).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"VolTrend_{self.window}D"
'''
AMT_TREND_CODE = '''class AmountTrend:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return data['amount'] / data['amount'].shift(self.window).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"AmtTrend_{self.window}D"
'''
LOG_VOL_CODE = '''class LogVolume:
    def __init__(self, window: int = 1, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        return -np.log1p(data['volume'])
    def get_name(self) -> str: return "LogVolume"
'''
LIQ_SHOCK_CODE = '''class LiquidityShock:
    def __init__(self, window: int = 1, **kwargs): self.window = window
    def compute(self, data: dict): return data['volume'] / data['volume'].shift(1).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return "LiqShock_1D"
'''
VOL_OSC_CODE = '''class VolumeOscillator:
    def __init__(self, short_w: int = 5, long_w: int = 20, **kwargs): self.sw, self.lw = short_w, long_w
    def compute(self, data: dict):
        return data['volume'].rolling(self.sw).mean() / data['volume'].rolling(self.lw).mean().replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"VolOsc_{self.sw}_{self.lw}"
'''

# ==========================================
# Category 5: Price-Volume (量价微观类 41-50)
# ==========================================
VWAP_DEV_10D_CODE = '''class VwapDeviation10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        vwap = data["amount"] / data["volume"].replace(0, 1e-12)
        return -(data["close"] / vwap.replace(0, 1e-12) - 1.0).rolling(self.window).mean()
    def get_name(self) -> str: return f"VwapDev_{self.window}D"
'''
PV_CORR_20D_CODE = '''class PriceVolumeCorr20D:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict): return -data["close"].rolling(self.window).corr(data["volume"])
    def get_name(self) -> str: return f"PVCorr_{self.window}D"
'''
OBV_10D_CODE = '''class ObvProxy10D:
    def __init__(self, window: int = 10, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        return (np.sign(data["close"].diff()) * data["volume"]).rolling(self.window).mean()
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
FORCE_INDEX_CODE = '''class ForceIndex:
    def __init__(self, window: int = 13, **kwargs): self.window = window
    def compute(self, data: dict):
        return ((data['close'] - data['close'].shift(1)) * data['volume']).rolling(self.window).mean()
    def get_name(self) -> str: return f"ForceIndex_{self.window}D"
'''
MFI_PROXY_CODE = '''class MFIProxy:
    def __init__(self, window: int = 14, **kwargs): self.window = window
    def compute(self, data: dict):
        tp = (data['high'] + data['low'] + data['close']) / 3
        return -(tp * data['volume']).rolling(self.window).mean()
    def get_name(self) -> str: return f"MFIProxy_{self.window}D"
'''
ILLIQ_VAR_CODE = '''class IlliquidityVariance:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        amihud = np.abs(data['close'].pct_change()) / data['amount'].replace(0, 1e-12)
        return -amihud.rolling(self.window).std()
    def get_name(self) -> str: return f"IlliqVar_{self.window}D"
'''
VWAP_MOM_CODE = '''class VwapMomentum:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        vwap = data['amount'] / data['volume'].replace(0, 1e-12)
        return vwap / vwap.shift(self.window).replace(0, 1e-12) - 1.0
    def get_name(self) -> str: return f"VwapMom_{self.window}D"
'''
PV_TREND_CODE = '''class PriceVolumeTrend:
    def __init__(self, window: int = 20, **kwargs): self.window = window
    def compute(self, data: dict):
        import numpy as np
        return (np.sign(data['close'].pct_change()) * data['volume']).rolling(self.window).sum()
    def get_name(self) -> str: return f"PVTrend_{self.window}D"
'''
VWAP_REV_5D_CODE = '''class VwapReversion5D:
    def __init__(self, window: int = 5, **kwargs): self.window = window
    def compute(self, data: dict):
        vwap = data['amount'] / data['volume'].replace(0, 1e-12)
        return -(data['close'] / vwap.replace(0, 1e-12) - 1.0).rolling(self.window).mean()
    def get_name(self) -> str: return f"VwapRev_{self.window}D"
'''

# ==========================================
# Registration & Helpers
# ==========================================
_BASE_FACTORS = {
    # Trend
    "momentum_20d": {"id": "base-mom-20", "name": "Momentum 20D", "code": MOMENTUM_20D_CODE, "category": "trend",
                     "parameters": {"window": 20}, "description": "20d"},
    "momentum_5d": {"id": "base-mom-5", "name": "Momentum 5D", "code": MOMENTUM_5D_CODE, "category": "trend",
                    "parameters": {"window": 5}, "description": "5d"},
    "price_accel_10d": {"id": "base-accel-10", "name": "Price Accel 10D", "code": PRICE_ACCEL_CODE, "category": "trend",
                        "parameters": {"window": 10}, "description": "10d"},
    "ma_cross": {"id": "base-macross", "name": "MA Cross", "code": MA_CROSS_CODE, "category": "trend",
                 "parameters": {"short_w": 5, "long_w": 20}, "description": "cross"},
    "momentum_60d": {"id": "base-mom-60", "name": "Momentum 60D", "code": MOMENTUM_60D_CODE, "category": "trend",
                     "parameters": {"window": 60}, "description": "60d"},
    "momentum_120d": {"id": "base-mom-120", "name": "Momentum 120D", "code": MOMENTUM_120D_CODE, "category": "trend",
                      "parameters": {"window": 120}, "description": "120d"},
    "overnight_mom": {"id": "base-on-mom", "name": "Overnight Mom", "code": OVERNIGHT_MOM_CODE, "category": "trend",
                      "parameters": {"window": 1}, "description": "1d"},
    "intraday_mom": {"id": "base-in-mom", "name": "Intraday Mom", "code": INTRADAY_MOM_CODE, "category": "trend",
                     "parameters": {"window": 1}, "description": "1d"},
    "ema_cross": {"id": "base-ema-cross", "name": "EMA Cross", "code": EMA_CROSS_CODE, "category": "trend",
                  "parameters": {"short_w": 10, "long_w": 30}, "description": "cross"},
    "macd_proxy": {"id": "base-macd", "name": "MACD Proxy", "code": MACD_PROXY_CODE, "category": "trend",
                   "parameters": {"short_w": 12, "long_w": 26}, "description": "macd"},

    # Reversion
    "mean_reversion_20d": {"id": "base-mr-20", "name": "Mean Rev 20D", "code": MEAN_REV_20D_CODE,
                           "category": "reversion", "parameters": {"window": 20}, "description": "20d"},
    "bias_10d": {"id": "base-bias-10", "name": "Bias 10D", "code": BIAS_10D_CODE, "category": "reversion",
                 "parameters": {"window": 10}, "description": "10d"},
    "rsi_14d": {"id": "base-rsi-14", "name": "RSI Proxy 14D", "code": RSI_14D_CODE, "category": "reversion",
                "parameters": {"window": 14}, "description": "14d"},
    "high_52w": {"id": "base-high52w", "name": "High 52W", "code": HIGH_52W_CODE, "category": "reversion",
                 "parameters": {"window": 252}, "description": "52w"},
    "short_rev_1d": {"id": "base-srev-1", "name": "Short Rev 1D", "code": SHORT_REV_1D_CODE, "category": "reversion",
                     "parameters": {"window": 1}, "description": "1d"},
    "short_rev_3d": {"id": "base-srev-3", "name": "Short Rev 3D", "code": SHORT_REV_3D_CODE, "category": "reversion",
                     "parameters": {"window": 3}, "description": "3d"},
    "disparity_5d": {"id": "base-disp-5", "name": "Disparity 5D", "code": DISPARITY_5D_CODE, "category": "reversion",
                     "parameters": {"window": 5}, "description": "5d"},
    "boll_pct_b": {"id": "base-boll", "name": "Boll %B", "code": BOLL_PCT_B_CODE, "category": "reversion",
                   "parameters": {"window": 20}, "description": "20d"},
    "stoch_k": {"id": "base-stoch-k", "name": "Stoch K Proxy", "code": STOCH_K_CODE, "category": "reversion",
                "parameters": {"window": 14}, "description": "14d"},
    "williams_r": {"id": "base-will-r", "name": "Williams R Proxy", "code": WILLIAMS_R_CODE, "category": "reversion",
                   "parameters": {"window": 14}, "description": "14d"},

    # Volatility
    "amplitude_20d": {"id": "base-amp-20", "name": "Amplitude 20D", "code": AMPLITUDE_20D_CODE,
                      "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "volatility_20d": {"id": "base-vol-20", "name": "Volatility 20D", "code": VOLATILITY_20D_CODE,
                       "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "downside_vol": {"id": "base-dvol-20", "name": "Downside Vol 20D", "code": DOWNSIDE_VOL_CODE,
                     "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "intraday_vol": {"id": "base-ivol-20", "name": "Intraday Vol 20D", "code": INTRADAY_VOL_CODE,
                     "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "parkinson_vol_20d": {"id": "base-park-20", "name": "Parkinson Vol 20D", "code": PARKINSON_VOL_CODE,
                          "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "garman_klass_vol_20d": {"id": "base-gk-20", "name": "Garman Klass Vol 20D", "code": GARMAN_KLASS_VOL_CODE,
                             "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "vol_of_vol_20d": {"id": "base-vov-20", "name": "Vol of Vol 20D", "code": VOL_OF_VOL_CODE, "category": "volatility",
                       "parameters": {"window": 20}, "description": "20d"},
    "upside_vol_20d": {"id": "base-uvol-20", "name": "Upside Vol 20D", "code": UPSIDE_VOL_CODE,
                       "category": "volatility", "parameters": {"window": 20}, "description": "20d"},
    "high_low_spread_10d": {"id": "base-hls-10", "name": "High Low Spread 10D", "code": HL_SPREAD_CODE,
                            "category": "volatility", "parameters": {"window": 10}, "description": "10d"},
    "open_close_spread_10d": {"id": "base-ocs-10", "name": "Open Close Spread 10D", "code": OC_SPREAD_CODE,
                              "category": "volatility", "parameters": {"window": 10}, "description": "10d"},

    # Liquidity
    "volume_ratio_20d": {"id": "base-volrat-20", "name": "Volume Ratio 20D", "code": VOL_RATIO_20D_CODE,
                         "category": "liquidity", "parameters": {"window": 20}, "description": "20d"},
    "turnover_20d": {"id": "base-turn-20", "name": "Turnover 20D", "code": TURNOVER_20D_CODE, "category": "liquidity",
                     "parameters": {"window": 20}, "description": "20d"},
    "volume_std": {"id": "base-volstd-20", "name": "Volume Std 20D", "code": VOL_STD_20D_CODE, "category": "liquidity",
                   "parameters": {"window": 20}, "description": "20d"},
    "amt_mom_10d": {"id": "base-amtmom-10", "name": "Amount Mom 10D", "code": AMT_MOM_10D_CODE, "category": "liquidity",
                    "parameters": {"window": 10}, "description": "10d"},
    "turnover_volatility_20d": {"id": "base-tvol-20", "name": "Turnover Volatility 20D", "code": TURNOVER_VOL_CODE,
                                "category": "liquidity", "parameters": {"window": 20}, "description": "20d"},
    "volume_trend_20d": {"id": "base-vtrend-20", "name": "Volume Trend 20D", "code": VOL_TREND_CODE,
                         "category": "liquidity", "parameters": {"window": 20}, "description": "20d"},
    "amount_trend_20d": {"id": "base-atrend-20", "name": "Amount Trend 20D", "code": AMT_TREND_CODE,
                         "category": "liquidity", "parameters": {"window": 20}, "description": "20d"},
    "log_volume": {"id": "base-logv", "name": "Log Volume", "code": LOG_VOL_CODE, "category": "liquidity",
                   "parameters": {"window": 1}, "description": "1d"},
    "liquidity_shock_1d": {"id": "base-lshock-1", "name": "Liquidity Shock 1D", "code": LIQ_SHOCK_CODE,
                           "category": "liquidity", "parameters": {"window": 1}, "description": "1d"},
    "volume_oscillator": {"id": "base-vosc", "name": "Volume Oscillator", "code": VOL_OSC_CODE, "category": "liquidity",
                          "parameters": {"short_w": 5, "long_w": 20}, "description": "cross"},

    # Price-Volume
    "vwap_dev_10d": {"id": "base-vwap-10", "name": "VWAP Dev 10D", "code": VWAP_DEV_10D_CODE,
                     "category": "price_volume", "parameters": {"window": 10}, "description": "10d"},
    "pv_corr_20d": {"id": "base-pvcorr-20", "name": "PV Corr 20D", "code": PV_CORR_20D_CODE, "category": "price_volume",
                    "parameters": {"window": 20}, "description": "20d"},
    "obv_10d": {"id": "base-obv-10", "name": "OBV Proxy 10D", "code": OBV_10D_CODE, "category": "price_volume",
                "parameters": {"window": 10}, "description": "10d"},
    "amihud_20d": {"id": "base-amihud-20", "name": "Amihud 20D", "code": AMIHUD_20D_CODE, "category": "price_volume",
                   "parameters": {"window": 20}, "description": "20d"},
    "force_index_13d": {"id": "base-fi-13", "name": "Force Index 13D", "code": FORCE_INDEX_CODE,
                        "category": "price_volume", "parameters": {"window": 13}, "description": "13d"},
    "mfi_proxy_14d": {"id": "base-mfi-14", "name": "MFI Proxy 14D", "code": MFI_PROXY_CODE, "category": "price_volume",
                      "parameters": {"window": 14}, "description": "14d"},
    "illiquidity_variance_20d": {"id": "base-illiqv-20", "name": "Illiquidity Variance 20D", "code": ILLIQ_VAR_CODE,
                                 "category": "price_volume", "parameters": {"window": 20}, "description": "20d"},
    "vwap_momentum_20d": {"id": "base-vwapmom-20", "name": "VWAP Momentum 20D", "code": VWAP_MOM_CODE,
                          "category": "price_volume", "parameters": {"window": 20}, "description": "20d"},
    "price_volume_trend": {"id": "base-pvt-20", "name": "Price Volume Trend", "code": PV_TREND_CODE,
                           "category": "price_volume", "parameters": {"window": 20}, "description": "20d"},
    "vwap_reversion_5d": {"id": "base-vwaprev-5", "name": "VWAP Reversion 5D", "code": VWAP_REV_5D_CODE,
                          "category": "price_volume", "parameters": {"window": 5}, "description": "5d"},
}


def get_base_factor(name: str) -> dict: return deepcopy(_BASE_FACTORS[name])


def get_base_factor_library(name: str = "momentum_20d") -> list[dict]: return [get_base_factor(name)]


def get_all_base_factor_libraries() -> dict[str, list[dict]]: return {k: [deepcopy(v)] for k, v in
                                                                      _BASE_FACTORS.items()}


def get_combined_base_factor_library() -> list[dict]: return [deepcopy(v) for v in _BASE_FACTORS.values()]


def list_base_factor_names() -> list[str]: return list(_BASE_FACTORS.keys())