"""CLI runner: build a baseline factor library and evaluate via /evaluate.

Usage
-----
    # Alpha101 — fixed library, no mining needed
    python -m baselines.run_baseline --baseline alpha101 \\
        --universe HS300 --start 2023-01-01 --end 2023-03-31 \\
        --out baselines/results/alpha101.json

    # Search-based miners — need a panel for in-process scoring during search
    python -m baselines.run_baseline --baseline autoalpha --use-shadow-data
    python -m baselines.run_baseline --baseline gp        --use-shadow-data
    python -m baselines.run_baseline --baseline alphagen  --use-shadow-data

    # AlphaGen upstream canonical expressions (no mining)
    python -m baselines.run_baseline --baseline alphagen-upstream

The output JSON has the **same schema** as the agent pipeline's factor
library — every record carries `id, name, code, description, parameters,
category, evaluation` so direct comparison is trivial.

Notes
-----
- `--skip-evaluate` lets you produce the code-only library without
  hitting back_test (useful for offline inspection / committing).
- `--use-shadow-data` builds a deterministic shadow panel for the
  in-process scorer (no data_api needed). For realistic search you
  should instead point `--data-api-base` at a live data_api.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common import call_evaluator_http, dump_library


logger = logging.getLogger("run_baseline")


def _load_shadow_panel(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Build the same deterministic shadow panel back_test/data_loader uses."""
    from back_test.data_loader import _build_shadow_bars
    raw = _build_shadow_bars(symbols, start, end)
    master = pd.concat(raw, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    return {col: master.pivot(index="date", columns="symbol", values=col).sort_index().ffill().bfill()
            for col in ["open", "high", "low", "close", "volume", "amount"]}


async def _load_real_panel(data_api_base: str, universe: str, start: str, end: str
                            ) -> dict[str, pd.DataFrame]:
    """Hit data_api directly (bypassing back_test) to build a panel for offline mining."""
    os.environ["DATA_API_BASE"] = data_api_base
    from back_test.data_loader import load_market_data
    return await load_market_data({"universe": universe, "start_date": start, "end_date": end})


# Default shadow symbol set — same one back_test/data_loader.py falls back to.
SHADOW_SYMBOLS = ["000001", "600000", "000002", "600016", "600030"]


def _build_alpha101() -> list[dict]:
    from baselines.alpha101 import get_library
    return get_library()


def _build_autoalpha(panel: dict[str, pd.DataFrame], top_k: int) -> list[dict]:
    from baselines.autoalpha import get_library
    return get_library(panel, n_random=400, top_k=top_k, beam_iters=2, seed=7)


def _build_gp(panel: dict[str, pd.DataFrame], top_k: int) -> list[dict]:
    from baselines.gplearn_gp import GPConfig, get_library
    cfg = GPConfig(pop_size=200, generations=12, top_k=top_k, seed=13)
    return get_library(panel, cfg)


def _build_alphagen(panel: dict[str, pd.DataFrame], top_k: int) -> list[dict]:
    from baselines.alphagen import get_library
    from baselines.alphagen.miner import AlphaGenConfig
    cfg = AlphaGenConfig(n_iters=30, batch_size=16, top_k=top_k, seed=19)
    return get_library(panel, cfg)


def _build_alphagen_upstream() -> list[dict]:
    from baselines.alphagen.adapter import CANONICAL_EXPRESSIONS, expressions_to_library
    return expressions_to_library(CANONICAL_EXPRESSIONS)


BASELINE_BUILDERS: dict[str, Callable] = {
    "alpha101":          ("fixed",   lambda panel, top_k: _build_alpha101()),
    "autoalpha":         ("mined",   _build_autoalpha),
    "gp":                ("mined",   _build_gp),
    "alphagen":          ("mined",   _build_alphagen),
    "alphagen-upstream": ("fixed",   lambda panel, top_k: _build_alphagen_upstream()),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, choices=sorted(BASELINE_BUILDERS))
    p.add_argument("--universe", default="HS300")
    p.add_argument("--start",    default="2023-01-01", help="Eval start date (inclusive)")
    p.add_argument("--end",      default="2023-03-31", help="Eval end date (inclusive)")
    p.add_argument("--top-k",    type=int, default=15,
                   help="Top-K kept by search-based baselines")

    p.add_argument("--data-api-base", default=os.getenv("DATA_API_BASE", "http://127.0.0.1:18001"),
                   help="Used by search-based baselines for in-process scoring")
    p.add_argument("--use-shadow-data", action="store_true",
                   help="Build the in-process scoring panel from shadow data instead of data_api")

    p.add_argument("--evaluator", default=os.getenv("EVALUATOR_ENDPOINT", "http://127.0.0.1:18000/evaluate"))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--skip-evaluate", action="store_true",
                   help="Produce the library file only — don't call /evaluate.")

    p.add_argument("--out", default=None,
                   help="Output JSON path (default: baselines/results/<baseline>.json)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    kind, builder = BASELINE_BUILDERS[args.baseline]
    out_path = Path(args.out) if args.out else ROOT / "baselines" / "results" / f"{args.baseline}.json"

    # 1. Acquire in-process panel for search-based miners.
    panel: dict[str, pd.DataFrame] = {}
    if kind == "mined":
        if args.use_shadow_data:
            logger.info("Building shadow panel for in-process scoring | symbols=%d | %s..%s",
                        len(SHADOW_SYMBOLS), args.start, args.end)
            panel = _load_shadow_panel(SHADOW_SYMBOLS, args.start, args.end)
        else:
            logger.info("Fetching real panel from data_api=%s | universe=%s | %s..%s",
                        args.data_api_base, args.universe, args.start, args.end)
            panel = asyncio.run(_load_real_panel(args.data_api_base, args.universe, args.start, args.end))
        logger.info("Panel ready: close shape=%s", panel["close"].shape)

    # 2. Build the library
    t0 = time.time()
    library = builder(panel, args.top_k)
    logger.info("Built %s library | factors=%d | took=%.1fs", args.baseline, len(library), time.time() - t0)

    # 3. Evaluate via /evaluate
    if not args.skip_evaluate:
        eval_config = {
            "universe": args.universe,
            "start_date": args.start,
            "end_date": args.end,
        }
        logger.info("Evaluating via %s | universe=%s | %s..%s",
                    args.evaluator, args.universe, args.start, args.end)
        for rec in library:
            t1 = time.time()
            try:
                result = call_evaluator_http(
                    record=rec, eval_config=eval_config,
                    endpoint=args.evaluator, timeout=args.timeout,
                )
            except Exception as e:
                result = {"status": "error", "error_code": "TRANSPORT_ERROR", "error_message": str(e)}
            if result["status"] == "success":
                rec["evaluation"] = result["metrics"]
                ic = result["metrics"].get("ic_mean", float("nan"))
                sharpe = result["metrics"].get("sharpe", float("nan"))
                logger.info("  %-22s  ic=%+.4f  sharpe=%+.4f  (%.1fs)",
                            rec["id"], ic, sharpe, time.time() - t1)
            else:
                rec["evaluation"] = {}
                rec["evaluation_error"] = {
                    "error_code": result.get("error_code"),
                    "error_message": result.get("error_message"),
                }
                logger.warning("  %-22s  EVAL_FAIL: %s — %s",
                               rec["id"], result.get("error_code"), result.get("error_message"))

    dump_library(library, out_path)
    logger.info("Wrote %d records → %s", len(library), out_path)

    # 4. Compact summary to stdout
    summary = {
        "baseline": args.baseline,
        "n_factors": len(library),
        "out": str(out_path),
        "universe": args.universe,
        "window": [args.start, args.end],
        "top_5_by_ic": sorted(
            [(r["id"], r.get("evaluation", {}).get("ic_mean", 0.0)) for r in library],
            key=lambda t: abs(t[1]), reverse=True,
        )[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
