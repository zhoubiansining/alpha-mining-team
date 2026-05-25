"""Check whether data_api returns real daily market data for the smoke-test setup.

This script talks directly to data_api. It does not use back_test.data_loader,
so it cannot silently fall back to shadow market data.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.parse
import urllib.request


def fetch_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_url(base_url: str, path: str, params: dict) -> str:
    return f"{base_url.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether data_api returns real daily bars.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18001", help="data_api base URL")
    parser.add_argument("--universe", default="HS300")
    parser.add_argument("--market", default="cn_stock")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-03-31")
    parser.add_argument("--max-symbols", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    health = fetch_json(f"{base_url}/health", args.timeout)
    print(f"health: {health}")

    universe_url = build_url(base_url, "/universe", {"name": args.universe, "market": args.market})
    universe = fetch_json(universe_url, args.timeout)
    symbols = universe.get("symbols", [])[: args.max_symbols]
    print(f"universe: name={universe.get('name')} count={universe.get('count')} sampled={symbols}")

    if not symbols:
        print("ERROR: universe returned no symbols; back_test would likely use fallback symbols.", file=sys.stderr)
        return 2

    failures = []
    summaries = []
    for symbol in symbols:
        bars_url = build_url(
            base_url,
            "/bars/daily",
            {
                "symbol": symbol,
                "start": args.start_date,
                "end": args.end_date,
                "market": args.market,
                "adjust": "qfq",
            },
        )
        try:
            payload = fetch_json(bars_url, args.timeout)
        except Exception as exc:
            failures.append((symbol, f"request failed: {exc}"))
            continue

        bars = payload.get("bars", [])
        closes = [float(bar.get("close", 0.0)) for bar in bars if bar.get("close") is not None]
        volumes = [float(bar.get("volume", 0.0)) for bar in bars if bar.get("volume") is not None]
        close_std = statistics.pstdev(closes) if len(closes) >= 2 else 0.0
        volume_std = statistics.pstdev(volumes) if len(volumes) >= 2 else 0.0
        summaries.append({
            "symbol": symbol,
            "count": len(bars),
            "close_std": close_std,
            "volume_std": volume_std,
            "first_date": bars[0].get("date") if bars else None,
            "last_date": bars[-1].get("date") if bars else None,
        })

        if not bars:
            failures.append((symbol, "no daily bars"))
        elif close_std == 0.0 and volume_std == 0.0:
            failures.append((symbol, "bars are constant; suspicious data"))

    print("daily bar summaries:")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))

    if failures:
        print("ERROR: data_api real-data check failed:", file=sys.stderr)
        for symbol, reason in failures:
            print(f"- {symbol}: {reason}", file=sys.stderr)
        return 3

    print("OK: data_api returned non-empty, non-constant daily bars for sampled symbols.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
