"""WorldQuant Alpha101 — subset re-implementation as AlphaFactorTemplate classes.

Source: Kakushadze, "101 Formulaic Alphas" (arXiv:1601.00991, 2015).

Each entry is a Python class string that:
- Implements `__init__(**kwargs)`, `compute(data)`, `get_name()`.
- Carries every helper it needs as `@staticmethod` so the class is safe to
  `exec()` under back_test/engine.py's split-globals/locals semantics.
- Returns a `pd.DataFrame` aligned to the close-price panel (date × symbol),
  matching the expectation of `back_test/engine.py::execute_and_evaluate`.

We standardize sign so that **larger factor value ⇒ predicted higher next-day
return** wherever the original alpha is documented as such; otherwise we
preserve the paper's sign.
"""
from __future__ import annotations

from baselines.common import build_factor_record


# Helper definitions injected verbatim into every factor class. Kept short
# because exec()'s globals/locals split prevents module-level helpers from
# being visible to class methods at call time.
_HELPERS = """
    @staticmethod
    def _rank(x):
        return x.rank(axis=1, pct=True)

    @staticmethod
    def _ts_sum(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).sum()

    @staticmethod
    def _ts_mean(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).mean()

    @staticmethod
    def _ts_std(x, d):
        return x.rolling(d, min_periods=max(2, d // 2)).std()

    @staticmethod
    def _ts_min(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).min()

    @staticmethod
    def _ts_max(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).max()

    @staticmethod
    def _ts_rank(x, d):
        return x.rolling(d, min_periods=max(2, d // 2)).rank(pct=True)

    @staticmethod
    def _ts_argmax(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmax, raw=True)

    @staticmethod
    def _ts_argmin(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmin, raw=True)

    @staticmethod
    def _delta(x, d):
        return x.diff(d)

    @staticmethod
    def _delay(x, d):
        return x.shift(d)

    @staticmethod
    def _correlation(x, y, d):
        return x.rolling(d, min_periods=max(2, d // 2)).corr(y)

    @staticmethod
    def _covariance(x, y, d):
        return x.rolling(d, min_periods=max(2, d // 2)).cov(y)

    @staticmethod
    def _scale(x, a=1.0):
        denom = x.abs().sum(axis=1).replace(0, np.nan)
        return x.div(denom, axis=0) * a

    @staticmethod
    def _signed_power(x, p):
        return np.sign(x) * np.power(np.abs(x), p)

    @staticmethod
    def _safe_div(x, y):
        return x / y.replace(0, np.nan)

    @staticmethod
    def _vwap(data):
        amount = data["amount"]
        volume = data["volume"].replace(0, np.nan)
        return amount / volume

    @staticmethod
    def _returns(close):
        return close.pct_change()
"""


def _wrap(class_name: str, body: str, doc: str) -> str:
    return (
        f"class {class_name}:\n"
        f'    """{doc}"""\n'
        f"    def __init__(self, **kwargs):\n"
        f"        self.kwargs = kwargs\n"
        f"\n"
        f"    def get_name(self):\n"
        f'        return "{class_name}"\n'
        f"{_HELPERS}\n"
        f"    def compute(self, data):\n"
        f"{body}\n"
    )


# ---------------------------------------------------------------------------
# Factor body strings. Each is the body of `compute(self, data)` starting at
# 8-space indent. Conventions: `close`, `open_`, `high`, `low`, `volume`,
# `amount`, `vwap`, `returns` are made available at the top of compute().
# ---------------------------------------------------------------------------

_PREAMBLE = """\
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        amount = data["amount"]
        vwap = self._vwap(data)
        returns = self._returns(close)
        adv20 = self._ts_mean(volume, 20)
"""


def _body(formula: str) -> str:
    return _PREAMBLE + f"        result = {formula}\n        return result\n"


_FACTORS: list[tuple[str, str, str, str]] = [
    # (factor_id, class_name, doc, body_or_formula). If body starts with whitespace
    # it's treated as a custom body; otherwise it's a one-line return formula.

    ("alpha101-002", "Alpha002",
     "Alpha#2: -corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)",
     "-self._correlation(self._rank(np.log(volume.replace(0, np.nan)).diff(2)), self._rank((close - open_) / open_.replace(0, np.nan)), 6)"),

    ("alpha101-003", "Alpha003",
     "Alpha#3: -corr(rank(open), rank(volume), 10)",
     "-self._correlation(self._rank(open_), self._rank(volume), 10)"),

    ("alpha101-004", "Alpha004",
     "Alpha#4: -ts_rank(rank(low), 9)",
     "-self._ts_rank(self._rank(low), 9)"),

    ("alpha101-005", "Alpha005",
     "Alpha#5: rank(open - mean(vwap,10)) * -abs(rank(close - vwap))",
     "self._rank(open_ - self._ts_mean(vwap, 10)) * (-1 * self._rank(close - vwap).abs())"),

    ("alpha101-006", "Alpha006",
     "Alpha#6: -corr(open, volume, 10)",
     "-self._correlation(open_, volume, 10)"),

    ("alpha101-012", "Alpha012",
     "Alpha#12: sign(delta(volume,1)) * -delta(close,1)",
     "np.sign(self._delta(volume, 1)) * (-self._delta(close, 1))"),

    ("alpha101-013", "Alpha013",
     "Alpha#13: -rank(cov(rank(close), rank(volume), 5))",
     "-self._rank(self._covariance(self._rank(close), self._rank(volume), 5))"),

    ("alpha101-014", "Alpha014",
     "Alpha#14: -rank(delta(returns,3)) * corr(open, volume, 10)",
     "(-self._rank(self._delta(returns, 3))) * self._correlation(open_, volume, 10)"),

    ("alpha101-015", "Alpha015",
     "Alpha#15: -sum(rank(corr(rank(high), rank(volume), 3)), 3)",
     "-self._ts_sum(self._rank(self._correlation(self._rank(high), self._rank(volume), 3)), 3)"),

    ("alpha101-016", "Alpha016",
     "Alpha#16: -rank(cov(rank(high), rank(volume), 5))",
     "-self._rank(self._covariance(self._rank(high), self._rank(volume), 5))"),

    ("alpha101-018", "Alpha018",
     "Alpha#18: -rank(stddev(|close-open|,5) + (close-open) + corr(close, open, 10))",
     "-self._rank(self._ts_std((close - open_).abs(), 5) + (close - open_) + self._correlation(close, open_, 10))"),

    ("alpha101-019", "Alpha019",
     "Alpha#19: -sign((close - delay(close,7)) + delta(close,7)) * (1 + rank(1 + sum(returns,250)))",
     "(-np.sign((close - self._delay(close, 7)) + self._delta(close, 7))) * (1 + self._rank(1 + self._ts_sum(returns, 250)))"),

    ("alpha101-022", "Alpha022",
     "Alpha#22: -delta(corr(high, volume, 5), 5) * rank(stddev(close, 20))",
     "(-self._delta(self._correlation(high, volume, 5), 5)) * self._rank(self._ts_std(close, 20))"),

    ("alpha101-028", "Alpha028",
     "Alpha#28: scale((corr(adv20, low, 5) + (high+low)/2) - close)",
     "self._scale(self._correlation(adv20, low, 5) + (high + low) / 2 - close)"),

    ("alpha101-033", "Alpha033",
     "Alpha#33: rank(-1 * (1 - open/close))",
     "self._rank(-1.0 * (1.0 - open_ / close.replace(0, np.nan)))"),

    ("alpha101-034", "Alpha034",
     "Alpha#34: rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))",
     "self._rank((1 - self._rank(self._safe_div(self._ts_std(returns, 2), self._ts_std(returns, 5)))) + (1 - self._rank(self._delta(close, 1))))"),

    ("alpha101-038", "Alpha038",
     "Alpha#38: -rank(ts_rank(close,10)) * rank(close/open)",
     "(-self._rank(self._ts_rank(close, 10))) * self._rank(close / open_.replace(0, np.nan))"),

    ("alpha101-040", "Alpha040",
     "Alpha#40: -rank(stddev(high,10)) * corr(high, volume, 10)",
     "(-self._rank(self._ts_std(high, 10))) * self._correlation(high, volume, 10)"),

    ("alpha101-041", "Alpha041",
     "Alpha#41: sqrt(high*low) - vwap",
     "np.sqrt((high * low).clip(lower=0)) - vwap"),

    ("alpha101-042", "Alpha042",
     "Alpha#42: rank(vwap - close) / rank(vwap + close)",
     "self._safe_div(self._rank(vwap - close), self._rank(vwap + close))"),

    ("alpha101-043", "Alpha043",
     "Alpha#43: ts_rank(volume/adv20,20) * ts_rank(-delta(close,7),8)",
     "self._ts_rank(self._safe_div(volume, adv20), 20) * self._ts_rank(-self._delta(close, 7), 8)"),

    ("alpha101-054", "Alpha054",
     "Alpha#54: -((low - close) * open^5) / ((low - high) * close^5)",
     "(-((low - close) * np.power(open_, 5))) / ((low - high).replace(0, np.nan) * np.power(close.replace(0, np.nan), 5))"),

    ("alpha101-101", "Alpha101",
     "Alpha#101: (close - open) / ((high - low) + 0.001)",
     "(close - open_) / ((high - low) + 0.001)"),

    # Two extras with custom multi-line bodies:

    ("alpha101-007", "Alpha007", "Alpha#7: if adv20<volume then -ts_rank(|delta(close,7)|,60)*sign(delta(close,7)) else -1",
     """\
        d7 = self._delta(close, 7)
        rank_part = (-self._ts_rank(d7.abs(), 60)) * np.sign(d7)
        cond = (adv20 < volume)
        result = rank_part.where(cond, other=-1.0)
        return result
"""),

    ("alpha101-009", "Alpha009", "Alpha#9: piecewise on rolling extremes of delta(close,1)",
     """\
        dc = self._delta(close, 1)
        roll_min = self._ts_min(dc, 5)
        roll_max = self._ts_max(dc, 5)
        result = dc.where(roll_min > 0, other=-dc)
        result = dc.where(roll_max < 0, other=result)
        return result
"""),
]


def get_library() -> list[dict]:
    """Return the Alpha101 baseline library, ready for evaluation."""
    records: list[dict] = []
    for factor_id, class_name, doc, formula_or_body in _FACTORS:
        if formula_or_body.startswith("        "):
            body = _PREAMBLE + formula_or_body
        else:
            body = _body(formula_or_body)
        code = _wrap(class_name, body, doc)
        records.append(
            build_factor_record(
                factor_id=factor_id,
                name=class_name,
                code=code,
                description=doc,
                parameters={},
                category="alpha101",
            )
        )
    return records


if __name__ == "__main__":
    import sys
    lib = get_library()
    print(f"Alpha101 library: {len(lib)} factors")
    for rec in lib:
        print(f"  {rec['id']:18s}  {rec['name']:10s}  {rec['description'][:80]}")
    if "--print" in sys.argv:
        print(lib[0]["code"])
