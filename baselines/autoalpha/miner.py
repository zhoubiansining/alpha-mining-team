"""Random / beam search over a small alpha grammar.

Grammar
-------
    Primitive  P  ::= close | open | high | low | volume | amount | vwap | returns
    UnaryOp    U  ::= rank | neg | abs | log | sign | sqrt
    TSOp       T  ::= ts_mean | ts_std | ts_rank | ts_argmax | ts_argmin
                       | ts_max  | ts_min  | delta   | delay
    BinaryOp   B  ::= add | sub | mul | safe_div
    Window     W  ::= 5 | 10 | 20

    Expr ::= P
           | U(Expr)
           | T(Expr, W)
           | B(Expr, Expr)

Search
------
1. Sample N random Exprs (bounded depth).
2. Score each by in-sample mean IC on a provided market panel.
3. Beam refinement: for the top-K, try local edits (swap primitive,
   shift window, wrap with unary).
4. Return top-K by IC, exported as AlphaFactorTemplate class strings.

The scorer is deliberately offline / in-process: the miner does **not**
need back_test running. `run_baseline.py` will later send these mined
formulas through `/evaluate` to get the canonical metrics that match the
agent pipeline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from baselines.common import build_factor_record


PRIMITIVES = ["close", "open_", "high", "low", "volume", "amount", "vwap", "returns"]
UNARY = ["rank", "neg", "abs", "log", "sign", "sqrt"]
TSOPS = ["ts_mean", "ts_std", "ts_rank", "ts_argmax", "ts_argmin",
         "ts_max", "ts_min", "delta", "delay"]
BINARY = ["add", "sub", "mul", "safe_div"]
WINDOWS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Expression AST. Each node renders to (a) an expression string used inside
# the generated AlphaFactorTemplate class, and (b) directly-callable Python
# for in-process scoring.
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    """Tiny AST: kind in {prim, unary, ts, bin}."""
    kind: str
    op: str = ""
    args: list["Expr"] = field(default_factory=list)
    window: int = 0
    name: str = ""

    def render(self) -> str:
        """Render as an expression that can appear inside `compute()`."""
        if self.kind == "prim":
            return self.name
        if self.kind == "unary":
            inner = self.args[0].render()
            return {
                "rank": f"self._rank({inner})",
                "neg": f"(-({inner}))",
                "abs": f"({inner}).abs()",
                "log": f"np.log(({inner}).clip(lower=1e-9))",
                "sign": f"np.sign({inner})",
                "sqrt": f"np.sqrt(({inner}).clip(lower=0))",
            }[self.op]
        if self.kind == "ts":
            inner = self.args[0].render()
            return {
                "ts_mean":  f"self._ts_mean({inner}, {self.window})",
                "ts_std":   f"self._ts_std({inner}, {self.window})",
                "ts_rank":  f"self._ts_rank({inner}, {self.window})",
                "ts_argmax": f"self._ts_argmax({inner}, {self.window})",
                "ts_argmin": f"self._ts_argmin({inner}, {self.window})",
                "ts_max":   f"self._ts_max({inner}, {self.window})",
                "ts_min":   f"self._ts_min({inner}, {self.window})",
                "delta":    f"({inner}).diff({self.window})",
                "delay":    f"({inner}).shift({self.window})",
            }[self.op]
        if self.kind == "bin":
            l = self.args[0].render()
            r = self.args[1].render()
            return {
                "add": f"(({l}) + ({r}))",
                "sub": f"(({l}) - ({r}))",
                "mul": f"(({l}) * ({r}))",
                "safe_div": f"self._safe_div(({l}), ({r}))",
            }[self.op]
        raise ValueError(self.kind)

    def label(self) -> str:
        """Short label for naming."""
        if self.kind == "prim":
            return self.name.rstrip("_")
        if self.kind == "unary":
            return f"{self.op}({self.args[0].label()})"
        if self.kind == "ts":
            return f"{self.op}{self.window}({self.args[0].label()})"
        if self.kind == "bin":
            return f"{self.op}({self.args[0].label()},{self.args[1].label()})"
        return "?"

    def depth(self) -> int:
        if self.kind == "prim":
            return 1
        return 1 + max((a.depth() for a in self.args), default=0)


def _sample(rng: random.Random, depth_budget: int) -> Expr:
    if depth_budget <= 1:
        return Expr(kind="prim", name=rng.choice(PRIMITIVES))
    # Pick a kind, with a bias toward TS / binary at moderate depth.
    kind = rng.choices(
        population=["prim", "unary", "ts", "bin"],
        weights=[1, 2, 3, 2],
        k=1,
    )[0]
    if kind == "prim":
        return Expr(kind="prim", name=rng.choice(PRIMITIVES))
    if kind == "unary":
        return Expr(kind="unary", op=rng.choice(UNARY),
                    args=[_sample(rng, depth_budget - 1)])
    if kind == "ts":
        return Expr(kind="ts", op=rng.choice(TSOPS), window=rng.choice(WINDOWS),
                    args=[_sample(rng, depth_budget - 1)])
    # binary
    return Expr(kind="bin", op=rng.choice(BINARY),
                args=[_sample(rng, depth_budget - 1), _sample(rng, depth_budget - 1)])


# ---------------------------------------------------------------------------
# In-process scoring. We mirror the helpers used by the generated class so
# the scorer and the back_test execution path agree numerically.
# ---------------------------------------------------------------------------

class _Helpers:
    @staticmethod
    def _rank(x):       return x.rank(axis=1, pct=True)
    @staticmethod
    def _ts_mean(x, d): return x.rolling(d, min_periods=max(1, d // 2)).mean()
    @staticmethod
    def _ts_std(x, d):  return x.rolling(d, min_periods=max(2, d // 2)).std()
    @staticmethod
    def _ts_rank(x, d): return x.rolling(d, min_periods=max(2, d // 2)).rank(pct=True)
    @staticmethod
    def _ts_max(x, d):  return x.rolling(d, min_periods=max(1, d // 2)).max()
    @staticmethod
    def _ts_min(x, d):  return x.rolling(d, min_periods=max(1, d // 2)).min()
    @staticmethod
    def _ts_argmax(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmax, raw=True)
    @staticmethod
    def _ts_argmin(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmin, raw=True)
    @staticmethod
    def _safe_div(x, y): return x / y.replace(0, np.nan)


def _evaluate_local(expr: Expr, data: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """Compute mean IC and IR of expression on the given panel."""
    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    amount = data["amount"]
    vwap = amount / volume.replace(0, np.nan)
    returns = close.pct_change()
    self = _Helpers()  # noqa: F841 — referenced inside eval()'d expression

    expr_str = expr.render()
    try:
        factor = eval(expr_str, {"np": np, "pd": pd}, {
            "close": close, "open_": open_, "high": high, "low": low,
            "volume": volume, "amount": amount, "vwap": vwap,
            "returns": returns, "self": self,
        })
    except Exception:
        return 0.0, 0.0
    if not isinstance(factor, pd.DataFrame):
        return 0.0, 0.0
    factor = factor.reindex_like(close).replace([np.inf, -np.inf], np.nan)
    if factor.notna().mean().mean() < 0.3:
        return 0.0, 0.0
    fwd = close.shift(-1) / close - 1.0
    # Spearman = Pearson on cross-sectional ranks. We rank in-house so we
    # don't pull in scipy as a hard dep of the miner.
    f_rank = factor.rank(axis=1, pct=True)
    r_rank = fwd.rank(axis=1, pct=True)
    ic = f_rank.corrwith(r_rank, axis=1, method="pearson").dropna()
    if ic.empty:
        return 0.0, 0.0
    return float(ic.mean()), float(ic.mean() / (ic.std() + 1e-9))


# ---------------------------------------------------------------------------
# Local edits used by beam refinement.
# ---------------------------------------------------------------------------

def _mutate(rng: random.Random, expr: Expr) -> Expr:
    """Apply a small structural edit."""
    edit = rng.choice(["wrap_unary", "wrap_ts", "swap_prim", "shift_window", "neg_top"])
    if edit == "wrap_unary":
        return Expr(kind="unary", op=rng.choice(UNARY), args=[expr])
    if edit == "wrap_ts":
        return Expr(kind="ts", op=rng.choice(TSOPS), window=rng.choice(WINDOWS), args=[expr])
    if edit == "swap_prim":
        # find a primitive and replace it
        def _replace(e: Expr) -> Expr:
            if e.kind == "prim":
                return Expr(kind="prim", name=rng.choice(PRIMITIVES))
            return Expr(kind=e.kind, op=e.op, window=e.window, name=e.name,
                        args=[_replace(a) for a in e.args])
        return _replace(expr)
    if edit == "shift_window":
        def _shift(e: Expr) -> Expr:
            if e.kind == "ts":
                new_window = rng.choice([w for w in WINDOWS if w != e.window]) or e.window
                return Expr(kind="ts", op=e.op, window=new_window,
                            args=[_shift(a) for a in e.args])
            return Expr(kind=e.kind, op=e.op, window=e.window, name=e.name,
                        args=[_shift(a) for a in e.args])
        return _shift(expr)
    # neg_top
    return Expr(kind="unary", op="neg", args=[expr])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Self-contained class template — does not depend on np being in caller locals;
# it's already in back_test/engine.py's exec_globals.
_CLASS_TEMPLATE = '''class {class_name}:
    """{doc}"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_name(self):
        return "{class_name}"

    @staticmethod
    def _rank(x):       return x.rank(axis=1, pct=True)
    @staticmethod
    def _ts_mean(x, d): return x.rolling(d, min_periods=max(1, d // 2)).mean()
    @staticmethod
    def _ts_std(x, d):  return x.rolling(d, min_periods=max(2, d // 2)).std()
    @staticmethod
    def _ts_rank(x, d): return x.rolling(d, min_periods=max(2, d // 2)).rank(pct=True)
    @staticmethod
    def _ts_max(x, d):  return x.rolling(d, min_periods=max(1, d // 2)).max()
    @staticmethod
    def _ts_min(x, d):  return x.rolling(d, min_periods=max(1, d // 2)).min()
    @staticmethod
    def _ts_argmax(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmax, raw=True)
    @staticmethod
    def _ts_argmin(x, d):
        return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmin, raw=True)
    @staticmethod
    def _safe_div(x, y): return x / y.replace(0, np.nan)

    def compute(self, data):
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        amount = data["amount"]
        vwap = amount / volume.replace(0, np.nan)
        returns = close.pct_change()
        return {expr_str}
'''


def mine(
    data: dict[str, pd.DataFrame],
    *,
    n_random: int = 400,
    top_k: int = 15,
    beam_iters: int = 2,
    max_depth: int = 4,
    seed: int = 7,
) -> list[tuple[Expr, float, float]]:
    """Return a list of (expr, ic_mean, ir) sorted by |ic_mean| desc."""
    rng = random.Random(seed)
    scored: dict[str, tuple[Expr, float, float]] = {}

    def consider(expr: Expr) -> None:
        key = expr.render()
        if key in scored:
            return
        ic, ir = _evaluate_local(expr, data)
        if not np.isfinite(ic) or ic == 0:
            return
        scored[key] = (expr, ic, ir)

    # 1. Random sampling
    for _ in range(n_random):
        expr = _sample(rng, max_depth)
        if expr.depth() < 2:
            continue
        consider(expr)

    # 2. Beam refinement
    for _ in range(beam_iters):
        top = sorted(scored.values(), key=lambda t: abs(t[1]), reverse=True)[:top_k]
        for expr, _, _ in top:
            for _ in range(6):
                consider(_mutate(rng, expr))

    ranked = sorted(scored.values(), key=lambda t: abs(t[1]), reverse=True)
    return ranked[:top_k]


def expr_to_record(expr: Expr, ic: float, ir: float, idx: int) -> dict:
    """Materialize a mined Expr into the library record schema."""
    class_name = f"AutoAlpha{idx:03d}"
    doc = f"AutoAlpha mined #{idx} (in-sample ic={ic:+.4f}, ir={ir:+.4f}) — {expr.label()}"
    code = _CLASS_TEMPLATE.format(class_name=class_name, doc=doc, expr_str=expr.render())
    return build_factor_record(
        factor_id=f"autoalpha-{idx:03d}",
        name=class_name,
        code=code,
        description=doc,
        parameters={"in_sample_ic": ic, "in_sample_ir": ir},
        category="autoalpha",
    )


def get_library(data: dict[str, pd.DataFrame], **kwargs) -> list[dict]:
    """Run the miner and return its top-K factors as library records."""
    top = mine(data, **kwargs)
    return [expr_to_record(expr, ic, ir, i + 1) for i, (expr, ic, ir) in enumerate(top)]
