# Baselines — alpha-mining-team 配套基线

四个公式化因子挖掘 baseline，全部产出 **与 `tests/base_factors.py` 完全一致的因子库 schema**，从而可以和 agent pipeline 的输出直接做对照。每条因子库记录的字段：

```python
{
    "id":          str,
    "name":        str,
    "code":        str,   # 实现 AlphaFactorTemplate 的 Python class 源码
    "description": str,
    "parameters":  dict,
    "category":    str,
    "evaluation":  dict,  # 由 run_baseline.py 通过 /evaluate 回填
}
```

## 总览

| Baseline | 出处 | 因子来源 | 搜索算法 | 默认产出 |
| --- | --- | --- | --- | --- |
| `alpha101` | Kakushadze 2015 (arXiv:1601.00991) | WorldQuant 101 alpha 中挑选 25 条经典公式 | 无搜索，固定因子库 | 25 个 class |
| `autoalpha` | Lin et al. 2019《AutoAlpha》 | 在算子-原语文法上做随机采样 + 束局部编辑 | 随机采样 + beam refinement，本地 IC 打分 | top-K（默认 15）|
| `gp` | gplearn 风格符号回归 | 在共享 AST 上做种群进化 | 锦标赛选择 + 子树交叉 / 子树·点·提升变异 + ramped half-and-half 初始化 | top-K（默认 15）|
| `alphagen` | Yu et al. IJCAI 2023 (RL-MLDM/alphagen) | 由学到的策略对 DSL 表达式采样 | 纯 numpy 的 REINFORCE + baseline | top-K（默认 12）|
| `alphagen-upstream` | ↑ 同论文真实 PPO 输出 | 上游仓库训练得到的 top 表达式 → 经 `adapter.translate_alphagen_expression` 翻译 | 无搜索，固定列表 | 10 个 class（占位示例，请替换为真实 PPO 输出）|

> **关于 `gp`**：上游 `gplearn` 只能处理扁平的 `(n_samples × n_features)`，不理解我们这里横截面挖掘需要的 `(date × stock)` 面板结构。所以我们没有用上游 gplearn，而是直接在面板感知的 AST 上重新实现了同一算法族（锦标赛选择、子树交叉、子树/点/提升变异、ramped half-and-half 初始化）。**不需要 `pip install gplearn`**。

> **关于 `alphagen`**：真正按论文复现需要 PyTorch + Qlib + PPO，加上约 1–2 小时的 GPU 训练。本项目内置的是一个 numpy-only REINFORCE 实现，目的是给 agent pipeline 一个公平、快、像样的 RL-类 baseline；如果你要跑论文级真值，请走 `alphagen/README.md` 介绍的上游路径，并用 `adapter.py` 完成数据桥接。

## 快速开始

```bash
# 1. 启动后端服务
DATA_API_PORT=18001 python -m data_api.main          &
BACK_TEST_PORT=18000 python -m back_test.main        &

# 2. 跑任意 baseline，结果 JSON 落在 baselines/results/<name>.json
python -m baselines.run_baseline --baseline alpha101 \
    --universe HS300 --start 2023-01-01 --end 2023-03-31
python -m baselines.run_baseline --baseline autoalpha --top-k 15
python -m baselines.run_baseline --baseline gp        --top-k 15
python -m baselines.run_baseline --baseline alphagen  --top-k 12
python -m baselines.run_baseline --baseline alphagen-upstream
```

三个搜索型 miner（`autoalpha` / `gp` / `alphagen`）在 **搜索过程中** 也需要面板用来做本地打分。默认会去访问 `data_api` 取真实面板；如果想完全离线跑（无网络、无 data_api），加 `--use-shadow-data` 即可使用与 `back_test/data_loader._build_shadow_bars` 一致的确定性影子面板。

如果你只想生成因子库代码而不触发后端评估（例如要把库 commit 到仓库 / 仅做代码 review），加 `--skip-evaluate`。

## 目录结构

```
baselines/
├── __init__.py
├── common.py                      # build_factor_record / /evaluate 调用 / JSON 落盘
├── run_baseline.py                # 统一 CLI 入口
├── alpha101/
│   ├── __init__.py
│   └── factors.py                 # 25 个 WorldQuant Alpha101 class
├── autoalpha/
│   ├── __init__.py
│   └── miner.py                   # 随机 + 束搜索；共享 Expr AST
├── gplearn_gp/
│   ├── __init__.py
│   └── miner.py                   # 面板感知 GP，复用同一 AST
├── alphagen/
│   ├── __init__.py
│   ├── miner.py                   # numpy-only REINFORCE 策略
│   ├── adapter.py                 # 面板 ↔ AlphaGen 数组 + DSL 翻译
│   └── README.md                  # 如何接到上游 PyTorch 仓库
└── results/                       # 因子库 JSON 输出目录
```

## 与 agent pipeline 的对照方式

`run_mining(...)["factors"]` 的每条记录 schema 与上面一致，所以直接 diff 即可：

```python
import json
from baselines.common import load_library

agent_factors = run_mining_result["factors"]
alpha101 = load_library("baselines/results/alpha101.json")

def best_by_ic(lib):
    return sorted(lib, key=lambda r: r["evaluation"].get("ic_mean", 0), reverse=True)

print("Agent top-3:    ", [(r["name"], r["evaluation"]["ic_mean"]) for r in best_by_ic(agent_factors)[:3]])
print("Alpha101 top-3: ", [(r["name"], r["evaluation"]["ic_mean"]) for r in best_by_ic(alpha101)[:3]])
```

可比较的标准指标全部来自 `back_test /evaluate` 原样回填：`ic_mean / ic_std / ir / sharpe / max_drawdown / turnover / long_short_return / win_rate`。

## 几点注意事项

- **miner 报告的 in-sample IC**（写在 `parameters.in_sample_ic`）是本地用搜索面板算的，**不是** 走 `/evaluate` 的标准结果。做横向对比时永远以 `evaluation.ic_mean` 为准。
- 默认 mining 参数面向"笔记本几秒钟跑完"的快速反馈。如果要做更严肃的对比，建议调大：
  - `autoalpha`：`n_random=2000, beam_iters=4, top_k=30`
  - `gp`：`pop_size=500, generations=40, top_k=30`
  - `alphagen`：`n_iters=300, batch_size=64, lr=0.02`
- `back_test/engine.py` 用 `corrwith(..., method='spearman')`，依赖 `scipy`。如果 `/evaluate` 报 `EVAL_ERROR: No module named 'scipy'`，`pip install scipy` 即可。
