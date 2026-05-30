"""End-to-end smoke test for real LLM + data_api + back_test.

This script expects the following services to be running:
- data_api on :8001
- back_test on :8000

It runs a minimal one-iteration mining job against a small baseline factor library.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.config import AlphaMiningConfig
from alpha_mining.workflow import run_mining
from tests.base_factors import get_base_factor_library


def setup_logging() -> None:
    level_name = os.getenv("ALPHA_MINING_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_config(max_iterations: int, universe: str, start_date: str, end_date: str) -> AlphaMiningConfig:
    config = AlphaMiningConfig.from_env()
    config.iteration.max_iterations = max_iterations
    config.iteration.min_proposals_per_iteration = 1
    config.iteration.max_proposals_per_iteration = 1
    config.target_assets = [universe]
    config.eval_period = {"start_date": start_date, "end_date": end_date}
    config.evaluator.use_mock = False
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-LLM alpha mining smoke test.")
    parser.add_argument("--baseline", default="momentum_20d", help="Baseline factor key from tests.base_factors")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--universe", default="HS300")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-03-31")
    parser.add_argument("--use-parallel", action="store_true", default=True)
    parser.add_argument("--serial", dest="use_parallel", action="store_false", help="Disable parallel execution")
    return parser.parse_args()


async def main() -> int:
    setup_logging()
    args = parse_args()
    baseline_factors = get_base_factor_library(args.baseline)
    config = build_config(args.max_iterations, args.universe, args.start_date, args.end_date)

    result = await run_mining(config, baseline_factors, use_parallel=args.use_parallel)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
