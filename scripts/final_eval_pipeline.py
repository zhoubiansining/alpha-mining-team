"""
多智能体量化因子挖掘：最终评测与可视化 Pipeline
运行方式:
单测模式: python scripts/final_eval_pipeline.py --baseline momentum_20d
全量模式: python scripts/final_eval_pipeline.py --baseline all
"""

import asyncio
import argparse
import os
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.config import AlphaMiningConfig
from alpha_mining.workflow import run_mining
from tests.base_factors import get_base_factor_library, get_all_base_factor_libraries

BACK_TEST_PORT = os.getenv("BACK_TEST_PORT", "18000")
EVALUATOR_ENDPOINT = f"http://127.0.0.1:{BACK_TEST_PORT}/evaluate"

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


async def evaluate_factor(factor_dict, eval_config: dict) -> dict:
    """调用后端回测服务评估单个因子，增加类型防御"""

    # 强制安全校验，避免 TypeError 或 AttributeError
    if not isinstance(factor_dict, dict):
        error_msg = f"严重错误：预期接收字典格式因子，实际接收到的是 {type(factor_dict)}。\n内容：{factor_dict}"
        print(error_msg)
        return {"status": "error", "metrics": None, "error_message": error_msg}

    payload = {
        "alpha_id": factor_dict.get("id", "unknown"),
        "alpha_description": factor_dict.get("description", ""),
        "alpha_code": factor_dict.get("code", ""),
        "parameters": factor_dict.get("parameters", {}),
        "eval_config": eval_config
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(EVALUATOR_ENDPOINT, json=payload, timeout=300.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "metrics": None, "error_message": f"网络请求失败: {e}"}


def plot_metrics_comparison(baseline_metrics: dict, optimized_eval_results: list, baseline_name: str):
    """生成并保存 2x2 性能对比柱状图"""
    records = []

    if baseline_metrics:
        records.append({
            "Factor": f"Baseline\n({baseline_name})",
            "IC Mean": baseline_metrics.get("ic_mean", 0),
            "IR": baseline_metrics.get("ir", 0),
            "Sharpe": baseline_metrics.get("sharpe", 0),
            "Annual Return": baseline_metrics.get("long_short_return", 0)
        })

    valid_optimizations = 0
    for res in optimized_eval_results:
        metrics = res.get("metrics")
        if metrics:
            records.append({
                "Factor": f"Agent_Opt_{valid_optimizations + 1}",
                "IC Mean": metrics.get("ic_mean", 0),
                "IR": metrics.get("ir", 0),
                "Sharpe": metrics.get("sharpe", 0),
                "Annual Return": metrics.get("long_short_return", 0)
            })
            valid_optimizations += 1

    df = pd.DataFrame(records)
    if df.empty or len(df) < 2:
        print(f"⚠️ {baseline_name}: 数据不足，无法生成对比图表。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Agent 因子挖掘评估报告 ({baseline_name})', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ("IC Mean", axes[0, 0], "#4C72B0"),
        ("IR", axes[0, 1], "#55A868"),
        ("Sharpe", axes[1, 0], "#DD8452"),
        ("Annual Return", axes[1, 1], "#C44E52")
    ]

    for metric, ax, color in metrics_to_plot:
        sns.barplot(data=df, x="Factor", y=metric, ax=ax, color=color)
        ax.set_title(metric, fontsize=14)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=10)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    output_path = ROOT / f"eval_report_{baseline_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 可视化报告已保存至: {output_path}")


async def run_single_pipeline(baseline_name: str, base_factors_list: list, config: AlphaMiningConfig,
                              eval_config: dict):
    """运行单个 baseline 的完整评估流"""
    print(f"\n{'=' * 60}\n🚀 开始测试任务: {baseline_name}\n{'=' * 60}")

    # 确保取出的是字典
    factor_dict = base_factors_list[0]

    baseline_eval_res = await evaluate_factor(factor_dict, eval_config)
    baseline_metrics = baseline_eval_res.get("metrics")

    if not baseline_metrics:
        print(f"❌ Baseline回测失败！详情: {baseline_eval_res.get('error_message', '无')}")
        return

    print(
        f"📈 Baseline初始表现: Sharpe={baseline_metrics.get('sharpe', 0):.4f}, IC={baseline_metrics.get('ic_mean', 0):.4f}")
    print("🤖 启动智能体挖掘...")

    mining_result = await run_mining(config, base_factors_list, use_parallel=True)
    discovered_factors = mining_result.get("discovered_factors", [])

    if not discovered_factors:
        print("⚠️ 未发现超越 Baseline 的新因子。")
        return

    print(f"🔍 发现 {len(discovered_factors)} 个新因子。正在回测...")

    optimized_eval_results = []
    for f in discovered_factors:
        res = await evaluate_factor(f, eval_config)
        optimized_eval_results.append(res)
        m = res.get("metrics", {})
        if m:
            print(f"  ✅ 优化因子 {f.get('id')}: Sharpe={m.get('sharpe', 0):.4f}, IC={m.get('ic_mean', 0):.4f}")
        else:
            print(f"  ❌ 优化因子 {f.get('id')} 回测失败。")

    plot_metrics_comparison(baseline_metrics, optimized_eval_results, baseline_name)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="momentum_20d", help="输入 'all' 测试所有内置因子")
    parser.add_argument("--universe", default="HS300")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-06-30")
    parser.add_argument("--max-iter", type=int, default=3)
    args = parser.parse_args()

    config = AlphaMiningConfig.from_env()
    config.iteration.max_iterations = args.max_iter
    eval_config = {"universe": args.universe, "start_date": args.start, "end_date": args.end}

    if args.baseline.lower() == "all":
        print(f"🔥 启动全量评测模式！")
        all_libs = get_all_base_factor_libraries()
        for name, factor_lib in all_libs.items():
            await run_single_pipeline(name, factor_lib, config, eval_config)
    else:
        base_factors = get_base_factor_library(args.baseline)
        await run_single_pipeline(args.baseline, base_factors, config, eval_config)

    print("\n🎉 所有任务执行完毕！")


if __name__ == "__main__":
    asyncio.run(main())