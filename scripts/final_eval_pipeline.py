"""
Multi-Agent Alpha Mining: Final Evaluation & Visualization Pipeline

Features:
- Results organized by baseline subdirectory (results/{baseline}/)
- Per-factor JSON metrics, comparison charts, and summary reports
- Uses metrics from the mining stage (ensures data consistency)

Usage:
  Single baseline: python scripts/final_eval_pipeline.py --baseline momentum_20d
  All baselines:   python scripts/final_eval_pipeline.py --baseline all
"""

import asyncio
import argparse
import json
import os
import time as time_module
import httpx
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.config import AlphaMiningConfig
from alpha_mining.workflow import run_mining
from tests.base_factors import get_base_factor_library, get_all_base_factor_libraries

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("ALPHA_MINING_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("final_eval_pipeline")

# ── Config ───────────────────────────────────────────────────────────────────
BACK_TEST_PORT = os.getenv("BACK_TEST_PORT", "18000")
EVALUATOR_ENDPOINT = f"http://127.0.0.1:{BACK_TEST_PORT}/evaluate"
MAX_CONCURRENT_EVALS = int(os.getenv("MAX_CONCURRENT_EVALS", "5"))

# ── Matplotlib 中文字体配置 ──────────────────────────────────────────────────
# 自动检测系统中文字体
CHINESE_FONTS = [
    "Heiti SC", "PingFang SC", "Hiragino Sans GB",
    "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
]
for font_name in CHINESE_FONTS:
    try:
        matplotlib.font_manager.fontManager.addfont(
            matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=font_name))
        )
    except Exception:
        pass

_available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_chinese_available = next((f for f in CHINESE_FONTS if f in _available_fonts), None)

if _chinese_available:
    plt.rcParams["font.sans-serif"] = [_chinese_available] + plt.rcParams["font.sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    logger.debug("Chinese font found: %s", _chinese_available)
else:
    logger.debug("No Chinese font; falling back to DejaVu Sans")

plt.rcParams["axes.unicode_minus"] = False

# ── 全局图表风格 ──────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    palette="muted",
    font=_chinese_available or "DejaVu Sans",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)

# ── 评估工具函数 ─────────────────────────────────────────────────────────────

async def evaluate_factor(
    factor_dict: dict,
    eval_config: dict,
    client: httpx.AsyncClient | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> dict:
    if not isinstance(factor_dict, dict):
        return {"status": "error", "metrics": None, "error_message": f"Invalid input type: {type(factor_dict)}"}

    factor_id = factor_dict.get("id", "unknown")
    factor_name = factor_dict.get("name", "unnamed")

    async def _do_request() -> dict:
        payload = {
            "alpha_id": factor_id,
            "alpha_description": factor_dict.get("description", ""),
            "alpha_code": factor_dict.get("code", ""),
            "parameters": factor_dict.get("parameters", {}),
            "eval_config": eval_config,
        }
        try:
            if client is not None:
                response = await client.post(EVALUATOR_ENDPOINT, json=payload, timeout=300.0)
            else:
                async with httpx.AsyncClient(timeout=300.0) as tmp:
                    response = await tmp.post(EVALUATOR_ENDPOINT, json=payload)
            response.raise_for_status()
            result = response.json()
            m = result.get("metrics") or {}
            logger.debug(
                "Evaluator | id=%s | status=%s | ic=%.4f | sharpe=%.4f",
                factor_id, result.get("status", "?"), m.get("ic_mean", 0), m.get("sharpe", 0),
            )
            return result
        except Exception as e:
            logger.error("Evaluator HTTP failed | id=%s | error=%s", factor_id, e)
            return {"status": "error", "metrics": None, "error_message": str(e)}

    if semaphore is not None:
        async with semaphore:
            return await _do_request()
    return await _do_request()


async def evaluate_factors_parallel(
    factors: list,
    eval_config: dict,
    max_concurrent: int = MAX_CONCURRENT_EVALS,
) -> list[dict]:
    if not factors:
        return []

    actual = min(max_concurrent, len(factors))
    logger.info("Evaluating %d factors in parallel (max_concurrent=%d)", len(factors), actual)

    semaphore = asyncio.Semaphore(actual)
    limits = httpx.Limits(max_connections=actual * 2, max_keepalive_connections=actual)
    completed = {"count": 0}
    total = len(factors)
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=300.0, limits=limits) as client:
        async def _eval(idx: int, factor: dict) -> dict:
            fid = factor.get("id", "?")
            fname = factor.get("name", "unnamed")
            try:
                result = await evaluate_factor(factor, eval_config, client=client, semaphore=semaphore)
            except Exception as e:
                logger.error("Eval exception | id=%s | error=%s", fid, e)
                result = {"status": "error", "metrics": None, "error_message": str(e)}

            async with lock:
                completed["count"] += 1
                done = completed["count"]

            m = result.get("metrics") or {}
            if m:
                logger.info(
                    "  [%d/%d] OK | %s | Sharpe=%.4f IC=%.4f IR=%.4f",
                    done, total, fname,
                    m.get("sharpe", 0), m.get("ic_mean", 0), m.get("ir", 0),
                )
            else:
                logger.warning("  [%d/%d] FAIL | %s | %s", done, total, fname, result.get("error_message", "unknown"))
            return result

        tasks = [_eval(i, f) for i, f in enumerate(factors)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

    return [r if not isinstance(r, Exception) else {"status": "error", "metrics": None, "error_message": str(r)} for r in raw]


# ── 可视化函数 ───────────────────────────────────────────────────────────────

def _draw_radar(ax, labels: list[str], baseline_vals: list[float], opt_vals: list[float],
                title: str, color_base: str, color_opt: str):
    x = range(len(labels))
    width = 0.35

    bars1 = ax.barh([i - width / 2 for i in x], baseline_vals, height=width, color=color_base, alpha=0.85, label="Baseline")
    bars2 = ax.barh([i + width / 2 for i in x], opt_vals, height=width, color=color_opt, alpha=0.85, label="Optimized")

    ax.set_yticks(list(x))
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars1 + bars2:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}" if abs(w) < 10 else f"{w:.2f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color="gray")


def plot_comparison_chart(
    baseline_metrics: dict,
    optimized_results: list[dict],
    baseline_name: str,
    output_dir: Path,
):
    metric_keys = ["ic_mean", "ir", "sharpe", "max_drawdown", "long_short_return", "win_rate"]
    metric_labels = ["IC Mean", "IR", "Sharpe", "Max Drawdown", "Annual Return", "Win Rate"]

    records = []
    records.append({
        "Factor": f"Baseline\n({baseline_name})",
        **{k: baseline_metrics.get(k, 0) for k in metric_keys},
    })
    valid_count = 0
    for res in optimized_results:
        m = res.get("metrics") or {}
        if not m:
            continue
        valid_count += 1
        records.append({
            "Factor": f"Agent_Opt_{valid_count}",
            **{k: m.get(k, 0) for k in metric_keys},
        })

    if len(records) < 2:
        logger.warning("Insufficient data for chart generation (records=%d)", len(records))
        return

    df = pd.DataFrame(records)
    n_opt = len(records) - 1
    colors_base = ["#4C72B0"] * len(records)
    palette = sns.color_palette("Blues_d", n_opt + 1)
    colors_opt = [palette[i + 1] for i in range(n_opt)]

    # ── Figure: 4-panel ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Alpha Mining Evaluation Report\nBaseline: {baseline_name}  |  Optimized: {n_opt} factor(s)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: Core metrics bar comparison
    ax1 = axes[0, 0]
    bvals = [df[k].iloc[0] for k in metric_keys[:4]]
    # Baseline bar centered at x - 0.2, width 0.35 → spans [x-0.375, x-0.025]
    # Optimized bars start at x + 0.1 (clearing baseline right edge at x-0.025)
    for i, row in df.iloc[1:].iterrows():
        opt_vals = [row[k] for k in metric_keys[:4]]
        offset = i * 0.12
        ax1.bar([xi + offset for xi in range(len(bvals))], opt_vals,
                width=0.10, color=colors_opt[i - 1], alpha=0.85)
    ax1.bar([xi - 0.2 for xi in range(len(bvals))], bvals,
            width=0.35, color=colors_base[0], alpha=0.9, label="Baseline", edgecolor="white")
    ax1.set_xticks(range(len(metric_labels[:4])))
    ax1.set_xticklabels(metric_labels[:4])
    ax1.set_title("Core Metrics Comparison", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: IC & Sharpe trend
    ax2 = axes[0, 1]
    ic_vals = [df["ic_mean"].iloc[0]] + [df["ic_mean"].iloc[i] for i in range(1, len(df))]
    sharpe_vals = [df["sharpe"].iloc[0]] + [df["sharpe"].iloc[i] for i in range(1, len(df))]
    x_labels = [df["Factor"].iloc[0]] + [df["Factor"].iloc[i] for i in range(1, len(df))]
    x_pos = range(len(x_labels))
    ax2.plot(x_pos, ic_vals, "o-", color="#4C72B0", linewidth=2, markersize=7, label="IC Mean")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_pos, sharpe_vals, "s--", color="#C44E52", linewidth=2, markersize=7, label="Sharpe")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_ylabel("IC Mean", color="#4C72B0")
    ax2_twin.set_ylabel("Sharpe", color="#C44E52")
    ax2.set_title("IC Mean & Sharpe Ratio Trend", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2_twin.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: Improvement heatmap
    ax3 = axes[1, 0]
    improvement_data = []
    opt_labels = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        improvements = []
        for k in metric_keys:
            b = df[k].iloc[0]
            o = row[k]
            if b == 0:
                improvements.append(0.0)
            elif k == "max_drawdown":
                improvements.append((b - o) / abs(b) if b != 0 else 0.0)
            else:
                improvements.append((o - b) / abs(b) if b != 0 else 0.0)
        improvement_data.append(improvements)
        opt_labels.append(df["Factor"].iloc[i])

    imp_df = pd.DataFrame(improvement_data, index=opt_labels, columns=metric_labels).T
    sns.heatmap(imp_df, annot=True, fmt="+.1%", cmap="RdYlGn", center=0,
                ax=ax3, cbar_kws={"label": "Improvement vs Baseline"}, linewidths=0.5)
    ax3.set_title("Improvement Rate Heatmap (vs Baseline)", fontsize=12)

    # Panel 4: Comprehensive comparison (horizontal bars)
    ax4 = axes[1, 1]
    radar_keys = ["ic_mean", "ir", "sharpe", "long_short_return", "win_rate"]
    radar_labels = ["IC Mean", "IR", "Sharpe", "Ann. Return", "Win Rate"]
    bvals = [baseline_metrics.get(k, 0) for k in radar_keys]
    max_opt_vals = [max(df[k].iloc[i] for i in range(1, len(df))) for k in radar_keys]
    _draw_radar(ax4, radar_labels, bvals, max_opt_vals,
                "Comprehensive Metrics (Baseline vs Best Optimized)", colors_base[0], colors_opt[0])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = output_dir / "comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Chart saved: %s", chart_path)


# ── 结果存储函数 ──────────────────────────────────────────────────────────────

def save_results(
    baseline_name: str,
    baseline_metrics: dict,
    baseline_eval_result: dict,
    discovered_factors: list[dict],
    optimized_results: list[dict],
    mining_result: dict,
    eval_config: dict,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Per-factor JSON ─────────────────────────────────────────────────
    factor_records = []
    for i, (factor, result) in enumerate(zip(discovered_factors, optimized_results), 1):
        m = result.get("metrics") or {}
        record = {
            "rank": i,
            "factor_id": factor.get("id", ""),
            "name": factor.get("name", ""),
            "description": factor.get("description", ""),
            "parent_id": factor.get("parent_id", ""),
            "iteration": factor.get("iteration", 0),
            "optimization_rationale": factor.get("optimization_rationale", ""),
            "status": result.get("status", "unknown"),
            "metrics": {
                "ic_mean": m.get("ic_mean", 0),
                "ic_std": m.get("ic_std", 0),
                "ir": m.get("ir", 0),
                "sharpe": m.get("sharpe", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "turnover": m.get("turnover", 0),
                "long_short_return": m.get("long_short_return", 0),
                "win_rate": m.get("win_rate", 0),
            },
            "improvement_vs_baseline": {
                "ic_mean_delta": m.get("ic_mean", 0) - baseline_metrics.get("ic_mean", 0),
                "sharpe_delta": m.get("sharpe", 0) - baseline_metrics.get("sharpe", 0),
                "ir_delta": m.get("ir", 0) - baseline_metrics.get("ir", 0),
                "return_delta": m.get("long_short_return", 0) - baseline_metrics.get("long_short_return", 0),
                "mdd_delta": m.get("max_drawdown", 0) - baseline_metrics.get("max_drawdown", 0),
            },
            "error_message": result.get("error_message", ""),
        }
        factor_records.append(record)

        safe_name = factor.get('name', 'unknown').replace(' ', '_').replace('/', '_')
        factor_json_path = output_dir / f"factor_{i:02d}_{safe_name}.json"
        with open(factor_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    # ── 2. Baseline metrics JSON ───────────────────────────────────────────
    baseline_record = {
        "name": baseline_name,
        "metrics": {
            "ic_mean": baseline_metrics.get("ic_mean", 0),
            "ic_std": baseline_metrics.get("ic_std", 0),
            "ir": baseline_metrics.get("ir", 0),
            "sharpe": baseline_metrics.get("sharpe", 0),
            "max_drawdown": baseline_metrics.get("max_drawdown", 0),
            "turnover": baseline_metrics.get("turnover", 0),
            "long_short_return": baseline_metrics.get("long_short_return", 0),
            "win_rate": baseline_metrics.get("win_rate", 0),
        },
        "raw_response": baseline_eval_result,
    }
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_record, f, ensure_ascii=False, indent=2)

    # ── 3. Chart ──────────────────────────────────────────────────────────
    plot_comparison_chart(baseline_metrics, optimized_results, baseline_name, output_dir)

    # ── 4. Summary JSON ───────────────────────────────────────────────────
    valid_results = [r for r in optimized_results if r.get("metrics")]
    if valid_results:
        opt_metrics = [r["metrics"] for r in valid_results]
        avg_ic = sum(m.get("ic_mean", 0) for m in opt_metrics) / len(opt_metrics)
        avg_sharpe = sum(m.get("sharpe", 0) for m in opt_metrics) / len(opt_metrics)
        avg_ir = sum(m.get("ir", 0) for m in opt_metrics) / len(opt_metrics)
        best_ic_idx = max(range(len(opt_metrics)), key=lambda i: opt_metrics[i].get("ic_mean", 0))
        best_sharpe_idx = max(range(len(opt_metrics)), key=lambda i: opt_metrics[i].get("sharpe", 0))
        best_ic = opt_metrics[best_ic_idx]
        best_sharpe = opt_metrics[best_sharpe_idx]
    else:
        avg_ic = avg_sharpe = avg_ir = 0.0
        best_ic = best_sharpe = {}

    summary = {
        "baseline_name": baseline_name,
        "eval_config": eval_config,
        "total_candidates": len(discovered_factors),
        "valid_candidates": len(valid_results),
        "mining_iterations": mining_result.get("iterations", 0),
        "final_candidates": mining_result.get("final_candidates", []),
        "termination_reason": mining_result.get("termination_reason", ""),
        "baseline_metrics": baseline_record["metrics"],
        "aggregate": {
            "avg_ic_mean": round(avg_ic, 6),
            "avg_sharpe": round(avg_sharpe, 6),
            "avg_ir": round(avg_ir, 6),
            "improvement_ic_mean": round(avg_ic - baseline_metrics.get("ic_mean", 0), 6),
            "improvement_sharpe": round(avg_sharpe - baseline_metrics.get("sharpe", 0), 6),
        },
        "best_by_ic": {
            "factor": factor_records[best_ic_idx] if valid_results else {},
            "metrics": best_ic,
            "improvement": {
                "ic_mean_delta": best_ic.get("ic_mean", 0) - baseline_metrics.get("ic_mean", 0),
            },
        } if valid_results else {},
        "best_by_sharpe": {
            "factor": factor_records[best_sharpe_idx] if valid_results else {},
            "metrics": best_sharpe,
            "improvement": {
                "sharpe_delta": best_sharpe.get("sharpe", 0) - baseline_metrics.get("sharpe", 0),
            },
        } if valid_results else {},
        "all_factors": factor_records,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved: %s", summary_path)

    # ── 5. Trajectory JSON ─────────────────────────────────────────────────
    iteration_history = mining_result.get("iteration_history", [])
    if iteration_history:
        trajectory_path = output_dir / "trajectory.json"
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump({
                "baseline_name": baseline_name,
                "total_iterations": len(iteration_history),
                "iterations": iteration_history,
            }, f, ensure_ascii=False, indent=2)
        logger.info("Trajectory saved: %s (%d iterations)", trajectory_path, len(iteration_history))

    return summary


# ── 主 Pipeline ───────────────────────────────────────────────────────────────

async def run_single_pipeline(
    baseline_name: str,
    base_factors_list: list,
    config: AlphaMiningConfig,
    eval_config: dict,
):
    results_dir = ROOT / "results" / baseline_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Task: %s  |  Output: %s", baseline_name, results_dir)
    logger.info("=" * 60)

    logger.info("Baseline factor: id=%s name=%s", base_factors_list[0].get("id"), base_factors_list[0].get("name"))

    # Step 1: Agent mining (baseline pre-evaluated during mining stage)
    logger.info("[Step 1/3] Starting agent mining (max_iterations=%d)...", config.iteration.max_iterations)
    t0 = time_module.time()
    mining_result = await run_mining(config, base_factors_list, use_parallel=True)
    mining_elapsed = time_module.time() - t0
    logger.info("Agent mining done (%.1fs)", mining_elapsed)

    # Extract baseline metrics from mining result (pre-evaluated during mining stage).
    # This is critical: re-evaluating here would use different market data (real vs shadow),
    # causing inconsistent IC/Sharpe values between mining logs and final output.
    all_factors = mining_result.get("factors", [])
    baseline_factors = [f for f in all_factors if f.get("is_baseline", False)]

    if not baseline_factors:
        logger.error("No baseline factor found in mining result")
        return

    baseline_factor = baseline_factors[0]
    baseline_metrics = baseline_factor.get("evaluation") or {}
    baseline_eval_res = {"status": "success", "metrics": baseline_metrics, "alpha_id": baseline_factor.get("id")}

    if not baseline_metrics:
        logger.error("Baseline factor has no evaluation metrics in mining result")
        return

    logger.info(
        "Baseline (from mining) | Sharpe=%.4f | IC=%.4f | IR=%.4f | MDD=%.4f | Ret=%.4f",
        baseline_metrics.get("sharpe", 0),
        baseline_metrics.get("ic_mean", 0),
        baseline_metrics.get("ir", 0),
        baseline_metrics.get("max_drawdown", 0),
        baseline_metrics.get("long_short_return", 0),
    )

    # Step 2: Extract non-baseline factors filtered to final_candidates only.
    # No re-evaluation: mining stage already called Evaluator.
    all_discovered = [f for f in all_factors if not f.get("is_baseline", False)]
    final_candidate_ids = set(mining_result.get("final_candidates", []))
    if final_candidate_ids:
        discovered_factors = [f for f in all_discovered if f.get("id") in final_candidate_ids]
        logger.info(
            "Filtered to final_candidates: %d/%d factors selected",
            len(discovered_factors), len(all_discovered),
        )
    else:
        discovered_factors = all_discovered

    optimized_results = []
    for f in discovered_factors:
        metrics = f.get("evaluation") or {}
        optimized_results.append({
            "status": "success" if metrics else "error",
            "metrics": metrics,
            "error_message": "",
        })

    logger.info(
        "Discovered %d candidate factors | final_candidates=%s",
        len(discovered_factors), mining_result.get("final_candidates", []),
    )
    logger.info("Reusing stored evaluation metrics from mining stage (no re-evaluation)")

    if not discovered_factors:
        logger.warning("No new factors discovered, saving baseline only")
        save_results(
            baseline_name, baseline_metrics, baseline_eval_res,
            [], [], mining_result, eval_config, results_dir,
        )
        return

    # Step 3: Save results
    logger.info("[Step 3/3] Saving results to %s ...", results_dir)
    summary = save_results(
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        baseline_eval_result=baseline_eval_res,
        discovered_factors=discovered_factors,
        optimized_results=optimized_results,
        mining_result=mining_result,
        eval_config=eval_config,
        output_dir=results_dir,
    )

    agg = summary.get("aggregate", {})
    logger.info(
        "Done | valid=%d/%d | avg_IC=%.4f (delta=%.4f) | avg_Sharpe=%.4f (delta=%.4f)",
        summary.get("valid_candidates", 0), summary.get("total_candidates", 0),
        agg.get("avg_ic_mean", 0), agg.get("improvement_ic_mean", 0),
        agg.get("avg_sharpe", 0), agg.get("improvement_sharpe", 0),
    )


async def main():
    parser = argparse.ArgumentParser(description="Alpha Mining Final Evaluation Pipeline")
    parser.add_argument("--baseline", default="momentum_20d", help="Baseline factor name, or 'all'")
    parser.add_argument("--universe", default="HS300")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger("final_eval_pipeline").setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("Alpha Mining Final Evaluation Pipeline")
    logger.info("baseline=%s | universe=%s | period=%s ~ %s | max_iter=%d",
                args.baseline, args.universe, args.start, args.end, args.max_iter)
    logger.info("Output directory: %s/results/", ROOT)
    logger.info("=" * 60)

    config = AlphaMiningConfig.from_env()
    config.iteration.max_iterations = args.max_iter
    eval_config = {"universe": args.universe, "start_date": args.start, "end_date": args.end}

    if args.baseline.lower() == "all":
        all_libs = get_all_base_factor_libraries()
        logger.info("Running all %d built-in baselines", len(all_libs))
        for name, factor_lib in all_libs.items():
            await run_single_pipeline(name, factor_lib, config, eval_config)
    else:
        base_factors = get_base_factor_library(args.baseline)
        await run_single_pipeline(args.baseline, base_factors, config, eval_config)

    logger.info("All tasks completed.")


if __name__ == "__main__":
    asyncio.run(main())
