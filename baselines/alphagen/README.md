# AlphaGen baseline

对应论文：Yu et al.《Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning》(IJCAI 2023)，上游仓库：`github.com/RL-MLDM/alphagen`。

本目录提供两条接入路径：

## 路径 A — 进程内 REINFORCE miner（默认）

`baselines.alphagen.miner` 在给定的面板上跑一个纯 numpy 的 REINFORCE 策略，遍历论文里定义的算子 DSL。**不依赖** PyTorch / Qlib / 任何训练基础设施，CPU 上几秒钟出结果。

```python
from baselines.alphagen import get_library, AlphaGenConfig
lib = get_library(panel, AlphaGenConfig(n_iters=30, batch_size=16, top_k=12))
```

这是对论文 PPO + transformer 设置的**刻意简化**。存在的意义是给 agent pipeline 提供一个**公平、快、可比较**的 RL-类 baseline，而**不是**论文级忠实复现。如果你需要论文级结果，请走路径 B。

## 路径 B — 上游 `alphagen` 仓库 + 我们的面板

`baselines.alphagen.adapter` 提供两个关键工具：

- `panel_to_alphagen_arrays(panel)` —— 把我们这边 `{field → DataFrame}` 形式的面板转成 AlphaGen `StockData` 期望的 `(T, S, F)` float32 张量，外加 `symbols / dates / field_index`。
- `expressions_to_library(exprs)` —— 把 AlphaGen 输出的文本 DSL 表达式（例如 `"Mul($close, Ref($volume, 5))"`）翻译成可以直接被本项目其它模块消费的 AlphaFactorTemplate class 因子库。

接入步骤：

```bash
pip install torch qlib stable-baselines3
git clone https://github.com/RL-MLDM/alphagen /tmp/alphagen
```

随后用 `panel_to_alphagen_arrays` 把面板喂给 AlphaGen 的训练循环；训练结束后把 top-K 表达式文本走一遍翻译：

```python
from baselines.alphagen.adapter import AlphaGenExpression, expressions_to_library
lib = expressions_to_library([
    AlphaGenExpression(text, label="best_run_42") for text in top_exprs
])
```

返回的 `lib` 与 `tests/base_factors.get_combined_base_factor_library()` **完全同构**，可以直接：

- 传给 `run_mining(..., baseline_factor_library=lib)`；
- 或通过 `python -m baselines.run_baseline --baseline alphagen-upstream` 走 `/evaluate` 走标准评估流程。

## DSL 覆盖范围

`adapter.translate_alphagen_expression` 覆盖论文公开的算子集合：

- 时序一元 + window：`Ref, Mean, Std, Var, Skew, Med, Mad, Max, Min, Sum, EMA, Delta`
- 一元：`Abs, Sign, Log, Rank`
- 二元：`Add, Sub, Mul, Div, Greater, Less`
- 原语：`$close, $open, $high, $low, $volume, $amount, $vwap, $returns`

未覆盖的算子会显式抛 `NotImplementedError` —— 我们倾向于让翻译失败"响"出来，而不是悄悄丢弃。
