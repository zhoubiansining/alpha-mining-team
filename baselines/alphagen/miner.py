"""Numpy-only REINFORCE policy over AlphaGen's operator DSL.

This is a deliberately lightweight reproduction of AlphaGen
(Yu et al., IJCAI 2023, github.com/RL-MLDM/alphagen). The upstream paper
uses a transformer policy trained with PPO; here we use a single softmax
over per-position action distributions trained with REINFORCE + baseline.
The point is to give the rest of the agent pipeline an honest RL-based
baseline producing factors in the same library schema, runnable in
seconds on CPU.

Action space (mirroring AlphaGen's published DSL where it overlaps with
our panel ops):

    primitives:       close, open, high, low, volume, amount, vwap, returns
    unary  ops:       Abs, Sign, Log, Rank (cross-sectional)
    ts unary ops:     Ref(d), Mean(d), Med(d), Std(d), Max(d), Min(d), Sum(d), Delta(d)
    binary ops:       Add, Sub, Mul, Div, Greater, Less

State / generation
------------------
Expressions are sampled in prefix-traversal order following an action
type schedule that guarantees a well-formed AST. Each position has its
own logit vector; rewards (mean IC) are propagated back as a single
return signal — the simplest form of REINFORCE.

Output: top-K sampled expressions ranked by validation IC, exported as
AlphaFactorTemplate classes via the same `_CLASS_TEMPLATE` used by the
gplearn/AutoAlpha miners (one canonical helper toolkit).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baselines.autoalpha.miner import (
    Expr,
    _CLASS_TEMPLATE,
    _evaluate_local,
)
from baselines.common import build_factor_record


# ---------------------------------------------------------------------------
# AlphaGen-aligned action vocabularies
# ---------------------------------------------------------------------------

PRIMS = ["close", "open_", "high", "low", "volume", "amount", "vwap", "returns"]
UNARIES = ["abs", "sign", "log", "rank"]
TS_OPS = [
    ("delay", 5),  ("delay", 10),  ("delay", 20),
    ("ts_mean", 5), ("ts_mean", 10), ("ts_mean", 20),
    ("ts_std", 5),  ("ts_std", 10),  ("ts_std", 20),
    ("ts_max", 5),  ("ts_max", 10),  ("ts_max", 20),
    ("ts_min", 5),  ("ts_min", 10),  ("ts_min", 20),
    ("ts_rank", 5), ("ts_rank", 10), ("ts_rank", 20),
    ("delta",  5),  ("delta",  10),  ("delta",  20),
]
BIN_OPS = ["add", "sub", "mul", "safe_div"]

# Action-type schedule. Each generated expression follows this fixed shape
# template; the policy chooses *which* op/primitive at each slot. Three
# templates are mixed during sampling to give the policy structural variety
# without exploding the search space.
TEMPLATES = [
    # T0: TS(unary(prim))                         — "transformed primitive"
    [("TS",), ("UNARY",), ("PRIM",)],
    # T1: BIN(TS(prim), TS(prim))                 — pairwise comparison
    [("BIN",), ("TS",), ("PRIM",), ("TS",), ("PRIM",)],
    # T2: unary(BIN(prim, prim))                  — interaction term
    [("UNARY",), ("BIN",), ("PRIM",), ("PRIM",)],
]


# ---------------------------------------------------------------------------
# Softmax policy (one logit vector per slot per template)
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    template_logits: np.ndarray  # (n_templates,)
    slot_logits: list[list[np.ndarray]]  # [tpl][slot] -> logits over action set

    @staticmethod
    def init(rng: np.random.Generator) -> "Policy":
        templates = []
        for tpl in TEMPLATES:
            per_slot = []
            for (kind,) in tpl:
                size = {"PRIM": len(PRIMS), "UNARY": len(UNARIES),
                        "TS": len(TS_OPS), "BIN": len(BIN_OPS)}[kind]
                per_slot.append(rng.normal(scale=0.01, size=size))
            templates.append(per_slot)
        return Policy(
            template_logits=rng.normal(scale=0.01, size=len(TEMPLATES)),
            slot_logits=templates,
        )

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()


def _sample_action(rng: np.random.Generator, logits: np.ndarray) -> tuple[int, float]:
    probs = Policy._softmax(logits)
    a = int(rng.choice(len(probs), p=probs))
    return a, math.log(probs[a] + 1e-12)


def _sample_expression(rng: np.random.Generator, policy: Policy):
    """Sample one expression. Returns (Expr, total_log_prob, trace)."""
    t_idx, log_t = _sample_action(rng, policy.template_logits)
    tpl = TEMPLATES[t_idx]
    slot_logits = policy.slot_logits[t_idx]

    actions: list[tuple[str, int]] = []
    total_logp = log_t
    for (kind,), logits in zip(tpl, slot_logits):
        a, lp = _sample_action(rng, logits)
        actions.append((kind, a))
        total_logp += lp

    expr = _materialize(t_idx, actions)
    trace = {"t_idx": t_idx, "actions": actions}
    return expr, total_logp, trace


def _materialize(t_idx: int, actions: list[tuple[str, int]]) -> Expr:
    """Assemble an Expr from the template + chosen actions."""
    def prim(i): return Expr(kind="prim", name=PRIMS[i])
    def unary(i, child):
        return Expr(kind="unary", op=UNARIES[i], args=[child])
    def ts(i, child):
        op, w = TS_OPS[i]
        return Expr(kind="ts", op=op, window=w, args=[child])
    def binop(i, l, r):
        return Expr(kind="bin", op=BIN_OPS[i], args=[l, r])

    if t_idx == 0:
        _, ts_a = actions[0]; _, u_a = actions[1]; _, p_a = actions[2]
        return ts(ts_a, unary(u_a, prim(p_a)))
    if t_idx == 1:
        _, b_a = actions[0]
        _, ts_l = actions[1]; _, p_l = actions[2]
        _, ts_r = actions[3]; _, p_r = actions[4]
        return binop(b_a, ts(ts_l, prim(p_l)), ts(ts_r, prim(p_r)))
    if t_idx == 2:
        _, u_a = actions[0]; _, b_a = actions[1]
        _, p_l = actions[2]; _, p_r = actions[3]
        return unary(u_a, binop(b_a, prim(p_l), prim(p_r)))
    raise ValueError(t_idx)


def _action_size(kind: str) -> int:
    return {"PRIM": len(PRIMS), "UNARY": len(UNARIES),
            "TS": len(TS_OPS), "BIN": len(BIN_OPS)}[kind]


def _policy_grad_update(policy: Policy, trajectories, lr: float) -> None:
    """REINFORCE with baseline = batch mean reward.

    Performs one gradient step for each trajectory's chosen template + slots.
    Gradient of log softmax wrt logits is (one_hot - softmax_probs).
    """
    rewards = np.array([r for _, _, _, r in trajectories], dtype=np.float64)
    baseline = rewards.mean() if len(rewards) else 0.0

    for expr, _logp, trace, r in trajectories:
        adv = r - baseline
        # Template logit gradient
        probs_t = Policy._softmax(policy.template_logits)
        grad_t = -probs_t.copy()
        grad_t[trace["t_idx"]] += 1.0
        policy.template_logits += lr * adv * grad_t

        # Slot logit gradients
        slot_logits = policy.slot_logits[trace["t_idx"]]
        for slot_i, ((kind, a_i), logits) in enumerate(zip(trace["actions"], slot_logits)):
            probs = Policy._softmax(logits)
            grad = -probs.copy()
            grad[a_i] += 1.0
            slot_logits[slot_i] += lr * adv * grad


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class AlphaGenConfig:
    n_iters: int = 30
    batch_size: int = 16
    lr: float = 0.05
    top_k: int = 12
    seed: int = 19
    keep_above_ic: float = 0.0  # only keep candidates with |IC| > this


def train_policy(data: dict[str, pd.DataFrame],
                 config: AlphaGenConfig | None = None
                ) -> tuple[Policy, list[tuple[Expr, float, float]]]:
    cfg = config or AlphaGenConfig()
    rng = np.random.default_rng(cfg.seed)
    policy = Policy.init(rng)

    scored: dict[str, tuple[Expr, float, float]] = {}
    history: list[float] = []

    for it in range(cfg.n_iters):
        trajectories = []
        for _ in range(cfg.batch_size):
            expr, logp, trace = _sample_expression(rng, policy)
            ic, ir = _evaluate_local(expr, data)
            if not np.isfinite(ic):
                ic = 0.0
            key = expr.render()
            if key not in scored and abs(ic) > cfg.keep_above_ic:
                scored[key] = (expr, ic, ir)
            trajectories.append((expr, logp, trace, abs(ic)))
        _policy_grad_update(policy, trajectories, cfg.lr)
        history.append(float(np.mean([r for _, _, _, r in trajectories])))

    ranked = sorted(scored.values(), key=lambda t: abs(t[1]), reverse=True)
    return policy, ranked[: cfg.top_k]


def expr_to_record(expr: Expr, ic: float, ir: float, idx: int) -> dict:
    class_name = f"AlphaGen{idx:03d}"
    doc = f"AlphaGen-RL mined #{idx} (in-sample ic={ic:+.4f}, ir={ir:+.4f}) — {expr.label()}"
    code = _CLASS_TEMPLATE.format(class_name=class_name, doc=doc, expr_str=expr.render())
    return build_factor_record(
        factor_id=f"alphagen-{idx:03d}",
        name=class_name,
        code=code,
        description=doc,
        parameters={"in_sample_ic": ic, "in_sample_ir": ir},
        category="alphagen_rl",
    )


def get_library(data: dict[str, pd.DataFrame],
                config: AlphaGenConfig | None = None) -> list[dict]:
    _, top = train_policy(data, config)
    return [expr_to_record(e, ic, ir, i + 1) for i, (e, ic, ir) in enumerate(top)]
