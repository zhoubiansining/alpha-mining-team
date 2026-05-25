"""Panel-aware GP miner (gplearn-style).

Algorithm
---------
- Initialization: ramped half-and-half (grow + full) over depths [2, max_depth].
- Fitness:        mean cross-sectional IC vs. T+1 returns (penalized for NaN
                  coverage), computed in-process on a single panel snapshot.
- Selection:      tournament selection of size `tournament_size`.
- Variation:      subtree crossover, subtree mutation, point mutation,
                  hoist mutation — all panel-aware on the shared `Expr` AST.
- Elitism:        top-`elite_size` carried over each generation.

Reuses the `Expr` AST and the in-process scorer from
`baselines.autoalpha.miner` so the two GP/heuristic baselines share one
operator grammar and one numerical contract. The differentiator is the
search algorithm: GP performs population-based evolution with crossover,
while AutoAlpha does random + beam local edits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baselines.autoalpha.miner import (
    BINARY,
    PRIMITIVES,
    TSOPS,
    UNARY,
    WINDOWS,
    Expr,
    _evaluate_local,
    _sample,
    _CLASS_TEMPLATE,
)
from baselines.common import build_factor_record


# ---------------------------------------------------------------------------
# Ramped half-and-half initialization
# ---------------------------------------------------------------------------

def _full(rng: random.Random, depth: int) -> Expr:
    """Build a tree that's full to `depth` (every leaf at the same depth)."""
    if depth <= 1:
        return Expr(kind="prim", name=rng.choice(PRIMITIVES))
    kind = rng.choices(["unary", "ts", "bin"], weights=[2, 3, 2])[0]
    if kind == "unary":
        return Expr(kind="unary", op=rng.choice(UNARY), args=[_full(rng, depth - 1)])
    if kind == "ts":
        return Expr(kind="ts", op=rng.choice(TSOPS), window=rng.choice(WINDOWS),
                    args=[_full(rng, depth - 1)])
    return Expr(kind="bin", op=rng.choice(BINARY),
                args=[_full(rng, depth - 1), _full(rng, depth - 1)])


def _ramped_half_and_half(rng: random.Random, pop_size: int, max_depth: int) -> list[Expr]:
    pop: list[Expr] = []
    for i in range(pop_size):
        d = (i % (max_depth - 1)) + 2  # cycle depths [2..max_depth]
        builder = _full if i % 2 == 0 else _sample
        pop.append(builder(rng, d))
    return pop


# ---------------------------------------------------------------------------
# AST helpers: enumerate nodes for crossover / mutation
# ---------------------------------------------------------------------------

def _clone(expr: Expr) -> Expr:
    return Expr(kind=expr.kind, op=expr.op, window=expr.window, name=expr.name,
                args=[_clone(a) for a in expr.args])


def _enumerate(expr: Expr) -> list[tuple[Expr, list[int]]]:
    """Return (node, path-from-root) for every node in the tree."""
    out: list[tuple[Expr, list[int]]] = [(expr, [])]
    for i, a in enumerate(expr.args):
        for node, sub in _enumerate(a):
            out.append((node, [i] + sub))
    return out


def _replace_at(root: Expr, path: list[int], new_subtree: Expr) -> Expr:
    if not path:
        return new_subtree
    cloned = _clone(root)
    cursor = cloned
    for step in path[:-1]:
        cursor = cursor.args[step]
    cursor.args[path[-1]] = new_subtree
    return cloned


# ---------------------------------------------------------------------------
# Variation operators
# ---------------------------------------------------------------------------

def _crossover(rng: random.Random, p1: Expr, p2: Expr, max_depth: int) -> Expr:
    nodes1 = _enumerate(p1)
    nodes2 = _enumerate(p2)
    _, path1 = rng.choice(nodes1)
    sub2, _ = rng.choice(nodes2)
    child = _replace_at(p1, path1, _clone(sub2))
    if child.depth() > max_depth + 1:
        return _clone(p1)
    return child


def _subtree_mutation(rng: random.Random, parent: Expr, max_depth: int) -> Expr:
    nodes = _enumerate(parent)
    _, path = rng.choice(nodes)
    new_sub = _sample(rng, max(2, max_depth // 2))
    child = _replace_at(parent, path, new_sub)
    if child.depth() > max_depth + 1:
        return _clone(parent)
    return child


def _point_mutation(rng: random.Random, parent: Expr) -> Expr:
    nodes = _enumerate(parent)
    target, _ = rng.choice(nodes)
    mutated = _clone(target)
    if mutated.kind == "prim":
        mutated.name = rng.choice(PRIMITIVES)
    elif mutated.kind == "unary":
        mutated.op = rng.choice(UNARY)
    elif mutated.kind == "ts":
        mutated.op = rng.choice(TSOPS)
        mutated.window = rng.choice(WINDOWS)
    elif mutated.kind == "bin":
        mutated.op = rng.choice(BINARY)
    # replace target with mutated copy
    _, path = next((n, p) for n, p in nodes if n is target)
    return _replace_at(parent, path, mutated)


def _hoist_mutation(rng: random.Random, parent: Expr) -> Expr:
    nodes = _enumerate(parent)
    non_root = [(n, p) for n, p in nodes if p]
    if not non_root:
        return _clone(parent)
    chosen, _ = rng.choice(non_root)
    return _clone(chosen)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@dataclass
class GPConfig:
    pop_size: int = 200
    generations: int = 12
    tournament_size: int = 7
    max_depth: int = 4
    p_crossover: float = 0.6
    p_subtree_mut: float = 0.2
    p_point_mut: float = 0.1
    p_hoist_mut: float = 0.05
    elite_size: int = 4
    top_k: int = 15
    seed: int = 13


def _fitness(expr: Expr, data: dict[str, pd.DataFrame]) -> tuple[float, float]:
    ic, ir = _evaluate_local(expr, data)
    return ic, ir


def _tournament(rng: random.Random,
                pop: list[tuple[Expr, float, float]],
                size: int) -> Expr:
    contenders = rng.sample(pop, k=min(size, len(pop)))
    contenders.sort(key=lambda t: abs(t[1]), reverse=True)
    return contenders[0][0]


def evolve(data: dict[str, pd.DataFrame],
           config: GPConfig | None = None) -> list[tuple[Expr, float, float]]:
    cfg = config or GPConfig()
    rng = random.Random(cfg.seed)

    scored: dict[str, tuple[Expr, float, float]] = {}

    def score(expr: Expr) -> tuple[Expr, float, float] | None:
        key = expr.render()
        if key in scored:
            return scored[key]
        ic, ir = _fitness(expr, data)
        if not np.isfinite(ic) or ic == 0.0:
            return None
        rec = (expr, ic, ir)
        scored[key] = rec
        return rec

    pop_raw = _ramped_half_and_half(rng, cfg.pop_size, cfg.max_depth)
    population: list[tuple[Expr, float, float]] = []
    for e in pop_raw:
        r = score(e)
        if r is not None:
            population.append(r)

    for gen in range(cfg.generations):
        population.sort(key=lambda t: abs(t[1]), reverse=True)
        new_pop: list[tuple[Expr, float, float]] = population[: cfg.elite_size]
        while len(new_pop) < cfg.pop_size:
            r = rng.random()
            if r < cfg.p_crossover and len(population) >= 2:
                a = _tournament(rng, population, cfg.tournament_size)
                b = _tournament(rng, population, cfg.tournament_size)
                child = _crossover(rng, a, b, cfg.max_depth)
            elif r < cfg.p_crossover + cfg.p_subtree_mut:
                a = _tournament(rng, population, cfg.tournament_size)
                child = _subtree_mutation(rng, a, cfg.max_depth)
            elif r < cfg.p_crossover + cfg.p_subtree_mut + cfg.p_point_mut:
                a = _tournament(rng, population, cfg.tournament_size)
                child = _point_mutation(rng, a)
            elif r < cfg.p_crossover + cfg.p_subtree_mut + cfg.p_point_mut + cfg.p_hoist_mut:
                a = _tournament(rng, population, cfg.tournament_size)
                child = _hoist_mutation(rng, a)
            else:
                child = _sample(rng, cfg.max_depth)
            rec = score(child)
            if rec is not None:
                new_pop.append(rec)
        population = new_pop

    population = list(scored.values())
    population.sort(key=lambda t: abs(t[1]), reverse=True)
    return population[: cfg.top_k]


def expr_to_record(expr: Expr, ic: float, ir: float, idx: int) -> dict:
    class_name = f"GpAlpha{idx:03d}"
    doc = f"GP-mined #{idx} (in-sample ic={ic:+.4f}, ir={ir:+.4f}) — {expr.label()}"
    code = _CLASS_TEMPLATE.format(class_name=class_name, doc=doc, expr_str=expr.render())
    return build_factor_record(
        factor_id=f"gp-{idx:03d}",
        name=class_name,
        code=code,
        description=doc,
        parameters={"in_sample_ic": ic, "in_sample_ir": ir},
        category="gp_symbolic",
    )


def get_library(data: dict[str, pd.DataFrame],
                config: GPConfig | None = None) -> list[dict]:
    top = evolve(data, config)
    return [expr_to_record(e, ic, ir, i + 1) for i, (e, ic, ir) in enumerate(top)]
