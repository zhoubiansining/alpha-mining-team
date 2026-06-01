"""Reusable daily baseline factor libraries for integration debugging.

Ultimate 100-factor library.
Contains 100 DISTINCT mathematical formulations covering:
1. Trend & Momentum (20)
2. Mean Reversion & Oscillator (20)
3. Volatility & Higher Moments (20)
4. Liquidity & Volume Profile (20)
5. Price-Volume Microstructure (20)
"""
from __future__ import annotations
import logging
import re
import textwrap
import platform


# ==========================================
# 全局可视化样式配置 (Optimized Visualization Settings)
# ==========================================
def setup_quant_visualization_style():
    """初始化量化图表全局风格，智能适配多操作系统，解决中文字体乱码问题"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 根据操作系统智能选择字体
        system = platform.system()
        if system == 'Windows':
            fonts = ['Microsoft YaHei', 'SimHei']
        elif system == 'Darwin':  # macOS
            fonts = ['PingFang SC', 'Arial Unicode MS']
        else:
            fonts = ['WenQuanYi Micro Hei', 'sans-serif']

        custom_rc = {
            'font.sans-serif': fonts + ['sans-serif'],
            'axes.unicode_minus': False,  # 解决负号显示为方块的问题
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.figsize': (12, 6),
            'lines.linewidth': 1.5,  # 增加线宽使得收益率曲线更饱满
        }

        # 使用 seaborn 的 darkgrid 增加量化图表的专业感
        sns.set_theme(style="darkgrid", palette="muted", rc=custom_rc)

        logging.info(f"Quant visualization style configured. OS: {system}, Fonts: {fonts[0]}")
    except ImportError:
        logging.warning("Matplotlib or Seaborn not found. Visualization styling skipped.")


# ==========================================
# 因子定义库 (完整 100 因子)
# ==========================================
_FACTOR_DEFS = [
    # ================= Category 1: Trend & Momentum =================
    ("mom_5d", "Momentum 5D", "close / close.shift(5) - 1.0", "trend", 5),
    ("mom_10d", "Momentum 10D", "close / close.shift(10) - 1.0", "trend", 10),
    ("mom_20d", "Momentum 20D", "close / close.shift(20) - 1.0", "trend", 20),
    ("mom_60d", "Momentum 60D", "close / close.shift(60) - 1.0", "trend", 60),
    ("accel_10d", "(close/close.shift(10).replace(0, 1e-12)) - (close.shift(10)/close.shift(20).replace(0, 1e-12))",
     "trend", 10),
    ("macd_proxy", "close.ewm(span=12).mean() - close.ewm(span=26).mean()", "trend", 12),
    ("trix_15d", "close.ewm(span=15).mean().ewm(span=15).mean().ewm(span=15).mean().pct_change()", "trend", 15),
    ("ema_cross_10d", "close.ewm(span=10).mean() / close.ewm(span=30).mean().replace(0, 1e-12) - 1", "trend", 10),
    ("sma_cross_5d", "close.rolling(5).mean() / close.rolling(20).mean().replace(0, 1e-12) - 1", "trend", 5),
    ("hl_mom_10d", "(high - low) / (high.shift(10) - low.shift(10)).replace(0, 1e-12)", "trend", 10),
    ("overnight_mom", "open / close.shift(1).replace(0, 1e-12) - 1", "trend", 1),
    ("intraday_mom", "close / open.replace(0, 1e-12) - 1", "trend", 1),
    ("tp_mom_1d", "(high + low + close)/3 / close.shift(1).replace(0, 1e-12) - 1", "trend", 1),
    ("high_break_10d", "close / close.shift(10).rolling(10).max().replace(0, 1e-12) - 1", "trend", 10),
    ("low_break_20d", "-(close / close.rolling(20).min().replace(0, 1e-12) - 1)", "trend", 20),
    ("lagged_mom_5d", "(close.shift(5) - close.shift(25)) / close.shift(25).replace(0, 1e-12)", "trend", 5),
    ("sma_cross_3d", "close.rolling(3).mean() / close.rolling(10).mean().replace(0, 1e-12) - 1", "trend", 3),
    ("close_loc_1d", "(close - open) / (high - low).replace(0, 1e-12)", "trend", 1),
    ("trend_str_10d", "(close - close.shift(10)) / (high.rolling(10).max() - low.rolling(10).min()).replace(0, 1e-12)",
     "trend", 10),
    ("sign_trend_5d", "np.sign(close.diff(5))", "trend", 5),

    # ================= Category 2: Mean Reversion =================
    ("zscore_20d", "-(close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, 1e-12)", "reversion", 20),
    ("bias_10d", "-(close / close.rolling(10).mean().replace(0, 1e-12) - 1)", "reversion", 10),
    ("bias_20d", "-(close / close.rolling(20).mean().replace(0, 1e-12) - 1)", "reversion", 20),
    ("bias_60d", "-(close / close.rolling(60).mean().replace(0, 1e-12) - 1)", "reversion", 60),
    ("rsi_14d",
     "-(100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean() / (-close.diff().clip(upper=0).rolling(14).mean()).replace(0, 1e-12))))",
     "reversion", 14),
    ("stoch_k_20d",
     "-(close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, 1e-12)",
     "reversion", 20),
    ("will_r_14d",
     "(close.rolling(14).max() - close) / (close.rolling(14).max() - close.rolling(14).min()).replace(0, 1e-12)",
     "reversion", 14),
    ("mid_bias_20d",
     "-(close - (close.rolling(20).max() + close.rolling(20).min())/2) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, 1e-12)",
     "reversion", 20),
    ("rev_1d", "-close.pct_change(1)", "reversion", 1),
    ("rev_3d", "-close.pct_change(3)", "reversion", 3),
    ("rev_5d", "-close.pct_change(5)", "reversion", 5),
    ("ema_zscore_20d", "-(close - close.ewm(span=20).mean()) / close.rolling(20).std().replace(0, 1e-12)", "reversion",
     20),
    ("high_bias_20d", "-(high / close.rolling(20).mean().replace(0, 1e-12) - 1)", "reversion", 20),
    ("low_bias_20d", "-(low / close.rolling(20).mean().replace(0, 1e-12) - 1)", "reversion", 20),
    ("high_52w_dist", "close / close.rolling(252).max().replace(0, 1e-12) - 1", "reversion", 252),
    ("low_52w_dist", "-(close / close.rolling(252).min().replace(0, 1e-12) - 1)", "reversion", 252),
    ("smooth_rev_5d", "-close.diff(1).rolling(5).mean()", "reversion", 5),
    ("open_bias_1d", "-(close - open) / open.replace(0, 1e-12)", "reversion", 1),
    ("mid_price_bias_1d", "-(close - (high+low)/2) / (high-low).replace(0, 1e-12)", "reversion", 1),
    ("boll_b_20d",
     "-(close - (close.rolling(20).mean() - 2*close.rolling(20).std())) / (4*close.rolling(20).std()).replace(0, 1e-12)",
     "reversion", 20),

    # ================= Category 3: Volatility & Higher Moments =================
    ("low_vol_20d", "-close.pct_change().rolling(20).std()", "volatility", 20),
    ("low_vol_60d", "-close.pct_change().rolling(60).std()", "volatility", 60),
    ("downside_vol_20d", "-close.pct_change().clip(upper=0).rolling(20).std()", "volatility", 20),
    ("upside_vol_20d", "-close.pct_change().clip(lower=0).rolling(20).std()", "volatility", 20),
    ("amp_20d", "-((high - low) / close.replace(0, 1e-12)).rolling(20).mean()", "volatility", 20),
    ("amp_10d", "-((high - low) / close.replace(0, 1e-12)).rolling(10).mean()", "volatility", 10),
    ("parkinson_20d", "-(np.log(high / low.replace(0, 1e-12))**2).rolling(20).mean()", "volatility", 20),
    ("garman_klass_20d",
     "-(0.5 * np.log(high/low.replace(0, 1e-12))**2 - (2*np.log(2)-1)*np.log(close/open.replace(0, 1e-12))**2).rolling(20).mean()",
     "volatility", 20),
    ("vol_of_vol_20d", "-close.pct_change().rolling(20).std().rolling(20).std()", "volatility", 20),
    ("oc_spread_1d", "-np.abs(close - open) / (high - low).replace(0, 1e-12)", "volatility", 1),
    ("atr_high_14d", "-(high - close.shift(1)).abs().rolling(14).mean()", "volatility", 14),
    ("atr_low_14d", "-(low - close.shift(1)).abs().rolling(14).mean()", "volatility", 14),
    ("hl_vol_14d", "-(high - low).rolling(14).mean() / close.replace(0, 1e-12)", "volatility", 14),
    ("skewness_20d", "close.pct_change().rolling(20).skew()", "volatility", 20),
    ("kurtosis_20d", "close.pct_change().rolling(20).kurt()", "volatility", 20),
    ("range_vol_20d",
     "-(close.rolling(20).max() - close.rolling(20).min()) / close.rolling(20).mean().replace(0, 1e-12)", "volatility",
     20),
    ("ret_range_20d", "-(close.pct_change().rolling(20).max() - close.pct_change().rolling(20).min())", "volatility",
     20),
    ("gap_vol_1d", "-(open - close.shift(1)).abs() / close.shift(1).replace(0, 1e-12)", "volatility", 1),
    ("abs_ret_1d", "-np.abs(close.pct_change())", "volatility", 1),
    ("cv_10d", "-close.rolling(10).std() / close.rolling(10).mean().replace(0, 1e-12)", "volatility", 10),

    # ================= Category 4: Liquidity & Volume =================
    ("vol_ratio_20d", "volume / volume.rolling(20).mean().replace(0, 1e-12) - 1", "liquidity", 20),
    ("vol_ratio_5d", "volume / volume.rolling(5).mean().replace(0, 1e-12) - 1", "liquidity", 5),
    ("turnover_20d", "-volume.rolling(20).mean()", "liquidity", 20),
    ("turnover_60d", "-volume.rolling(60).mean()", "liquidity", 60),
    ("vol_cv_20d", "-volume.rolling(20).std() / volume.rolling(20).mean().replace(0, 1e-12)", "liquidity", 20),
    ("amt_mom_10d", "amount / amount.shift(10).replace(0, 1e-12) - 1", "liquidity", 10),
    ("vol_mom_5d", "volume / volume.shift(5).replace(0, 1e-12) - 1", "liquidity", 5),
    ("log_vol_1d", "-np.log1p(volume)", "liquidity", 1),
    ("log_amt_1d", "-np.log1p(amount)", "liquidity", 1),
    ("vol_osc_5_20", "volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, 1e-12) - 1", "liquidity", 5),
    ("vol_of_vol_ret_20d", "-volume.pct_change().rolling(20).std()", "liquidity", 20),
    ("vol_accel_1d", "(volume - volume.shift(1)) / volume.shift(1).replace(0, 1e-12)", "liquidity", 1),
    ("vol_high_20d", "-volume.rolling(20).max() / volume.rolling(20).mean().replace(0, 1e-12)", "liquidity", 20),
    ("amt_osc_10_60", "amount.rolling(10).mean() / amount.rolling(60).mean().replace(0, 1e-12) - 1", "liquidity", 10),
    ("zero_vol_days_20d", "-(volume == 0).rolling(20).sum()", "liquidity", 20),
    ("vol_zscore_20d", "-np.abs(volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, 1e-12)",
     "liquidity", 20),
    ("vol_trend_10d", "volume.diff(1).rolling(10).mean()", "liquidity", 10),
    ("vol_low_10d", "-volume.rolling(10).min() / volume.rolling(10).mean().replace(0, 1e-12)", "liquidity", 10),
    ("vwap_1d", "amount / close.replace(0, 1e-12)", "liquidity", 1),
    ("illiq_proxy_1d", "-(high - low) / volume.replace(0, 1e-12)", "liquidity", 1),

    # ================= Category 5: Price-Volume Microstructure =================
    ("pv_corr_20d", "-close.rolling(20).corr(volume).fillna(0)", "price_volume", 20),
    ("pv_corr_10d", "-close.rolling(10).corr(volume).fillna(0)", "price_volume", 10),
    ("ret_vol_corr_20d", "-(close.pct_change().rolling(20).corr(volume.pct_change()).fillna(0))", "price_volume", 20),
    ("obv_10d", "(np.sign(close.diff()) * volume).rolling(10).mean()", "price_volume", 10),
    ("obv_20d", "(np.sign(close.diff()) * volume).rolling(20).mean()", "price_volume", 20),
    ("force_idx_13d", "((close - close.shift(1)) * volume).rolling(13).mean()", "price_volume", 13),
    ("amihud_1d", "-np.abs(close.pct_change()) / amount.replace(0, 1e-12)", "price_volume", 1),
    ("amihud_20d", "-(np.abs(close.pct_change()) / amount.replace(0, 1e-12)).rolling(20).mean()", "price_volume", 20),
    ("vwap_dev_10d", "-(close / (amount / volume.replace(0, 1e-12)).replace(0, 1e-12) - 1).rolling(10).mean()",
     "price_volume", 10),
    ("vwap_dev_5d", "-(close / (amount / volume.replace(0, 1e-12)).replace(0, 1e-12) - 1).rolling(5).mean()",
     "price_volume", 5),
    ("mfi_14d", "-(((high+low+close)/3 * volume).rolling(14).mean())", "price_volume", 14),
    ("vwap_mom_10d",
     "(amount / volume.replace(0, 1e-12)) / (amount.shift(10) / volume.shift(10).replace(0, 1e-12)).replace(0, 1e-12) - 1",
     "price_volume", 10),
    ("pv_trend_20d", "(np.sign(close.pct_change()) * volume).rolling(20).sum()", "price_volume", 20),
    ("close_loc_vol_1d", "volume * (close - open) / (high - low).replace(0, 1e-12)", "price_volume", 1),
    ("close_loc_amt_1d", "amount * (close - open) / (high - low).replace(0, 1e-12)", "price_volume", 1),
    ("pv_cov_20d", "-close.rolling(20).cov(volume).fillna(0)", "price_volume", 20),
    ("pv_interaction_20d",
     "((close - close.rolling(20).mean()) * (volume - volume.rolling(20).mean())).rolling(20).mean()", "price_volume",
     20),
    ("illiq_var_20d", "-(np.abs(close.pct_change()) / volume.replace(0, 1e-12)).rolling(20).std()", "price_volume", 20),
    ("ret_log_vol_1d", "close.pct_change() * np.log1p(volume)", "price_volume", 1),
    ("open_close_vol_1d", "-(close - open) / close.replace(0, 1e-12) * volume", "price_volume", 1),
]

_BASE_FACTORS = {}

# ==========================================
# 核心构建器 (修复局部变量冲突并预编译类引用)
# ==========================================
for fid, fname, expr, cat, win in _FACTOR_DEFS:
    class_name = "".join([word.capitalize() for word in fid.split('_')])

    dynamic_expr = expr
    is_safe_dynamic = True

    # 查找除了 0, 1 以及 1e-12 的 12 以外的其他独立数字
    numbers_in_expr = set(re.findall(r'\b\d+\b', expr))
    numbers_in_expr.discard('0')
    numbers_in_expr.discard('1')
    numbers_in_expr.discard('12')

    if win > 0 and str(win) in numbers_in_expr:
        if len(numbers_in_expr) > 1:
            is_safe_dynamic = False
        else:
            win_str = str(win)
            pattern_func = r'\.(rolling|shift|diff|pct_change)\(\s*' + win_str + r'\s*\)'
            dynamic_expr = re.sub(pattern_func, r'.\1(self.window)', dynamic_expr)
            pattern_span = r'span\s*=\s*' + win_str + r'\b'
            dynamic_expr = re.sub(pattern_span, r'span=self.window', dynamic_expr)

    # 修复：将局部变量严格定为 open，以确保安全执行因子表达式中的原生 open 关键字
    code_str = textwrap.dedent(f'''\
    class {class_name}:
        """
        Factor: {fname}
        Category: {cat}
        Dynamic Support: {is_safe_dynamic}
        """
        def __init__(self, window: int = {win}, **kwargs):
            self.window = window

        def compute(self, data: dict):
            import numpy as np
            import pandas as pd

            open = data.get('open')  # FIX: 必须用 open 匹配表达式
            high = data.get('high')
            low = data.get('low')
            close = data.get('close')
            volume = data.get('volume')
            amount = data.get('amount')

            if close is None:
                raise ValueError("Required data field 'close' is missing.")

            try:
                factor_val = {dynamic_expr}
                return factor_val
            except Exception as e:
                raise RuntimeError(f"Error computing {class_name}: {{str(e)}}")

        def get_name(self) -> str:
            return "{class_name}"
    ''')

    # 动态编译类，提取引用，避免后续重复 eval/exec，大幅提高调用性能
    local_scope = {}
    try:
        exec(code_str, globals(), local_scope)
        factor_class = local_scope[class_name]
    except Exception as e:
        logging.error(f"Failed to compile class {class_name}: {e}")
        factor_class = None

    _BASE_FACTORS[fid] = {
        "id": f"base-{fid.replace('_', '-')}",
        "name": fname,
        "category": cat,
        "parameters": {"window": win, "default_win": win},
        "metadata": {
            "version": "1.0.1",
            "is_dynamic": is_safe_dynamic,
            "requires": ["numpy", "pandas"]
        },
        "description": f"Pure Quant Alpha: {fname}",
        "code": code_str,
        "class_ref": factor_class  # 新增：直接存储类引用，方便实例化
    }


# ==========================================
# 优化后的标准接口函数 (Robust Access APIs)
# ==========================================
def _safe_copy_factor(factor: dict) -> dict:
    """内部辅助函数：安全拷贝因子字典，避免深拷贝 Class 对象报错"""
    return {
        **factor,
        "parameters": factor["parameters"].copy(),
        "metadata": factor["metadata"].copy()
    }


def get_base_factor(name: str) -> dict | None:
    """安全获取单一因子配置与类引用"""
    factor = _BASE_FACTORS.get(name)
    if not factor:
        logging.warning(f"Factor '{name}' not found in the library.")
        return None
    return _safe_copy_factor(factor)


def calculate_factor(name: str, data: dict, **kwargs):
    """
    【新增API】直接计算因子的便捷接口，免去手动实例化的繁琐
    示例: val = calculate_factor('mom_5d', ohlcv_data, window=10)
    """
    factor_info = get_base_factor(name)
    if not factor_info or not factor_info['class_ref']:
        raise ValueError(f"Factor '{name}' is unavailable or failed to compile.")

    # 允许 kwargs 覆盖默认窗口
    window = kwargs.get('window', factor_info['parameters']['default_win'])
    instance = factor_info['class_ref'](window=window)
    return instance.compute(data)


def get_multiple_base_factors(names: list[str]) -> list[dict]:
    """批量获取指定列表中的多个因子"""
    return [f for name in names if (f := get_base_factor(name)) is not None]


def get_all_base_factors_dict() -> dict[str, dict]:
    """获取全量字典格式（安全拷贝）"""
    return {k: _safe_copy_factor(v) for k, v in _BASE_FACTORS.items()}


def get_all_base_factors_list() -> list[dict]:
    """获取全量列表格式（安全拷贝）"""
    return [_safe_copy_factor(v) for v in _BASE_FACTORS.values()]


def list_base_factor_names() -> list[str]:
    """列出所有因子ID"""
    return list(_BASE_FACTORS.keys())


def get_factors_by_category(category: str) -> list[dict]:
    """按类别筛选因子"""
    valid_categories = {f['category'] for f in _BASE_FACTORS.values()}
    if category not in valid_categories:
        logging.error(f"Category '{category}' is invalid. Options are: {valid_categories}")
        return []

    return [
        _safe_copy_factor(factor) for factor in _BASE_FACTORS.values()
        if factor["category"] == category
    ]


# 模块加载时直接初始化全局画图参数
setup_quant_visualization_style()