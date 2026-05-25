"""Bridge back_test panel data ↔ upstream AlphaGen interfaces.

This module lets users who actually run the full upstream
[AlphaGen](https://github.com/RL-MLDM/alphagen) repo (PyTorch + Qlib + PPO)
feed it our back_test/data_api panel and then export the discovered
expressions back into our `AlphaFactorTemplate` library schema.

Two functions:

- `panel_to_alphagen_arrays(panel)`: convert the {field → date×symbol
  DataFrame} dict returned by `back_test/data_loader.load_market_data`
  into the (T, S, F) float32 tensor + feature index that AlphaGen's
  `StockData` expects.
- `expressions_to_library(exprs)`: turn AlphaGen's text expressions
  (e.g. `"Mul($close, Ref($volume, 5))"`) into AlphaFactorTemplate class
  records that drop straight into the rest of the pipeline.

We translate AlphaGen's prefix-form DSL into pandas-on-panel code that
back_test can `exec()`. The translator covers AlphaGen's published
operator set (Ref, Mean, Std, Var, Skew, Med, Mad, Max, Min, Sum, EMA,
Abs, Sign, Log, Add, Sub, Mul, Div, Greater, Less, Rank); missing
operators raise `NotImplementedError` so failures are loud rather than
silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baselines.autoalpha.miner import _CLASS_TEMPLATE
from baselines.common import build_factor_record


# ---------------------------------------------------------------------------
# panel -> (T, S, F) tensor
# ---------------------------------------------------------------------------

DEFAULT_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def panel_to_alphagen_arrays(
    panel: dict[str, pd.DataFrame],
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> tuple[np.ndarray, list[str], list[pd.Timestamp], dict[str, int]]:
    """Return (data[T,S,F], symbols, dates, field_index).

    AlphaGen's `StockData.data` is float32 of shape (n_days, n_stocks,
    n_features). We mirror that exactly. Missing values are filled with
    NaN (AlphaGen masks them downstream).
    """
    first = panel[fields[0]].sort_index()
    dates = list(first.index)
    symbols = list(first.columns)
    n_days, n_stocks = len(dates), len(symbols)
    arr = np.full((n_days, n_stocks, len(fields)), np.nan, dtype=np.float32)
    for f_idx, field in enumerate(fields):
        df = panel[field].reindex(index=dates, columns=symbols)
        arr[:, :, f_idx] = df.to_numpy(dtype=np.float32)
    field_index = {f: i for i, f in enumerate(fields)}
    return arr, symbols, dates, field_index


# ---------------------------------------------------------------------------
# AlphaGen expression text → AlphaFactorTemplate class
# ---------------------------------------------------------------------------

# AlphaGen op → pandas-on-panel translation. Operators take pre-rendered
# argument strings (already valid pandas Python) and return a new string.
_TS_TEMPLATES = {
    "Ref":   lambda x, d:    f"({x}).shift({d})",
    "Mean":  lambda x, d:    f"({x}).rolling({d}, min_periods=max(1, {d}//2)).mean()",
    "Std":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(2, {d}//2)).std()",
    "Var":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(2, {d}//2)).var()",
    "Skew":  lambda x, d:    f"({x}).rolling({d}, min_periods=max(3, {d}//2)).skew()",
    "Med":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(1, {d}//2)).median()",
    "Mad":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(2, {d}//2)).apply(lambda v: np.mean(np.abs(v - np.mean(v))), raw=True)",
    "Max":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(1, {d}//2)).max()",
    "Min":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(1, {d}//2)).min()",
    "Sum":   lambda x, d:    f"({x}).rolling({d}, min_periods=max(1, {d}//2)).sum()",
    "EMA":   lambda x, d:    f"({x}).ewm(span={d}, min_periods=max(1, {d}//2)).mean()",
    "Delta": lambda x, d:    f"({x}).diff({d})",
}

_UN_TEMPLATES = {
    "Abs":   lambda x: f"({x}).abs()",
    "Sign":  lambda x: f"np.sign({x})",
    "Log":   lambda x: f"np.log(({x}).clip(lower=1e-9))",
    "Rank":  lambda x: f"({x}).rank(axis=1, pct=True)",
}

_BIN_TEMPLATES = {
    "Add":     lambda l, r: f"(({l}) + ({r}))",
    "Sub":     lambda l, r: f"(({l}) - ({r}))",
    "Mul":     lambda l, r: f"(({l}) * ({r}))",
    "Div":     lambda l, r: f"self._safe_div(({l}), ({r}))",
    "Greater": lambda l, r: f"(({l}) > ({r})).astype(float)",
    "Less":    lambda l, r: f"(({l}) < ({r})).astype(float)",
}

_PRIMITIVE_TO_PANDAS = {
    "close": "close", "open": "open_", "high": "high", "low": "low",
    "volume": "volume", "amount": "amount", "vwap": "vwap",
    "returns": "returns",
}


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\$[A-Za-z_]+|-?\d+|[(),]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _parse(tokens: list[str], idx: int = 0):
    """Tiny recursive-descent parser for `Op(arg1, arg2, ...)` / `$field` /
    integers / bare identifiers. Returns (AST node, next_idx)."""
    tok = tokens[idx]
    if tok.startswith("$"):
        return ("prim", tok[1:]), idx + 1
    if re.fullmatch(r"-?\d+", tok):
        return ("const", int(tok)), idx + 1
    # operator call: Name ( arg, arg, ... )
    name = tok
    if idx + 1 >= len(tokens) or tokens[idx + 1] != "(":
        # bare identifier — treat as a primitive without $
        return ("prim", name), idx + 1
    idx += 2  # skip Name (
    args = []
    while tokens[idx] != ")":
        node, idx = _parse(tokens, idx)
        args.append(node)
        if tokens[idx] == ",":
            idx += 1
    return ("call", name, args), idx + 1


def _render(node) -> str:
    """Convert AST → pandas-on-panel expression string."""
    if node[0] == "prim":
        name = node[1]
        if name not in _PRIMITIVE_TO_PANDAS:
            raise NotImplementedError(f"Unknown primitive: ${name}")
        return _PRIMITIVE_TO_PANDAS[name]
    if node[0] == "const":
        return str(node[1])
    if node[0] == "call":
        op, args = node[1], node[2]
        rendered = [_render(a) for a in args]
        if op in _BIN_TEMPLATES and len(rendered) == 2:
            return _BIN_TEMPLATES[op](*rendered)
        if op in _UN_TEMPLATES and len(rendered) == 1:
            return _UN_TEMPLATES[op](*rendered)
        if op in _TS_TEMPLATES and len(rendered) == 2:
            return _TS_TEMPLATES[op](rendered[0], rendered[1])
        raise NotImplementedError(f"Unsupported AlphaGen op or arity: {op}({len(rendered)} args)")
    raise ValueError(node)


def translate_alphagen_expression(text: str) -> str:
    """Translate an AlphaGen text expression to a pandas-on-panel code body.

    Example
    -------
    >>> translate_alphagen_expression("Mul($close, Ref($volume, 5))")
    '((close) * ((volume).shift(5)))'
    """
    tokens = _tokenize(text)
    ast, _ = _parse(tokens, 0)
    return _render(ast)


@dataclass
class AlphaGenExpression:
    """One trained-output AlphaGen formula."""
    expression: str        # original DSL text, e.g. "Mul($close, Ref($volume, 5))"
    label: str             # short tag
    description: str = ""  # paper / source attribution


def expressions_to_library(exprs: list[AlphaGenExpression]) -> list[dict]:
    """Materialize AlphaGen DSL formulas into AlphaFactorTemplate classes."""
    records: list[dict] = []
    for i, e in enumerate(exprs, start=1):
        rendered = translate_alphagen_expression(e.expression)
        class_name = f"AlphaGenUpstream{i:03d}"
        doc = f"AlphaGen upstream #{i} — {e.label} — {e.expression}"
        code = _CLASS_TEMPLATE.format(class_name=class_name, doc=doc, expr_str=rendered)
        records.append(
            build_factor_record(
                factor_id=f"alphagen-upstream-{i:03d}",
                name=class_name,
                code=code,
                description=e.description or doc,
                parameters={"alphagen_expression": e.expression},
                category="alphagen_upstream",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Canonical examples — operator-grammar smoke set, NOT trained outputs.
# Replace these with the actual top expressions from your AlphaGen PPO run.
# ---------------------------------------------------------------------------

CANONICAL_EXPRESSIONS: list[AlphaGenExpression] = [
    AlphaGenExpression("Mul($close, Ref($volume, 5))", "MomVol",
                       "Price × delayed-volume — captures lag-volume momentum."),
    AlphaGenExpression("Sub(Mean($close, 5), Mean($close, 20))", "MA5_MA20",
                       "Short-vs-long moving-average crossover."),
    AlphaGenExpression("Div($close, Mean($close, 20))", "PriceToMA20",
                       "Price relative to 20-day mean."),
    AlphaGenExpression("Std($returns, 20)", "Vol20",
                       "20-day return volatility."),
    AlphaGenExpression("Rank(Sub($close, Ref($close, 5)))", "Rank5dRet",
                       "Cross-sectional rank of 5-day return."),
    AlphaGenExpression("Mul(Rank($volume), Rank(Sub($close, $open)))", "RankVol_RankIntraday",
                       "Rank-of-volume × rank-of-intraday-return."),
    AlphaGenExpression("Sub(EMA($close, 12), EMA($close, 26))", "MACD",
                       "Classic MACD-style EMA difference."),
    AlphaGenExpression("Div(Sub($high, $low), Mean($volume, 5))", "AmpPerVol",
                       "Daily range scaled by avg volume."),
    AlphaGenExpression("Rank(Div(Sub($vwap, $close), $close))", "VwapDevRank",
                       "Cross-sectional rank of vwap deviation."),
    AlphaGenExpression("Mul(Sign(Sub($close, Ref($close, 1))), $volume)", "OBV1d",
                       "On-balance-volume style 1-day directional volume."),
]
