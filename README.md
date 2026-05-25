# Alpha Mining Team

基于多LLM智能体团队的自动化金融量化因子挖掘系统。

## 项目简介

本系统采用对抗性反馈循环架构，由三个专业智能体（Leader、Proposer、Critic）协同工作，从给定的基线因子库出发，迭代优化生成更优的因子组合。

### 核心特性

- **对抗性反馈循环**：Proposer生成候选因子 → Evaluator评估 → Critic提供事实性批评
- **真实LLM调用**：基于OpenAI Compatible API的真实Agent调用，支持并行处理
- **单因子优化**：Leader每轮选择一个因子进行深度优化
- **因子库管理**：支持添加优质候选、选择性删除低价值因子
- **预期匹配度评估**：Critic评估Proposer优化思路与实际回测结果的匹配程度
- **合规性检查**：自动过滤不合规因子，支持重试修复
- **并行处理**：多因子评估和Critic并行执行，提升效率

## 系统架构

```
┌─────────┐    ┌───────────┐    ┌────────────┐    ┌────────┐
│ Leader  │───▶│  Proposer │───▶│  Evaluator │───▶│ Critic │
└─────────┘    └───────────┘    └────────────┘    └────────┘
     ▲                                       │           │
     │                                       │           │
     └───────────────────────────────────────┴───────────┘
                         反馈循环
```

### 智能体职责

| 智能体             | 职责                                             |
| ------------------ | ------------------------------------------------ |
| **Leader**   | 决策是否继续迭代、设定优化方向、选择单因子优化、管理因子库 |
| **Proposer** | 基于Leader指令生成新因子候选，提供优化思路解释 |
| **Critic**   | 基于评估结果提供事实性批评和改进建议，评估预期匹配度 |

## 目录结构

```
.
├── alpha_mining/          # 核心代码
│   ├── agents/           # 智能体实现
│   ├── config.py         # 配置管理
│   ├── prompts/          # 智能体提示词
│   ├── schemas/          # 数据模型
│   ├── tools/            # 工具函数
│   └── workflow.py       # LangGraph工作流（含并行处理）
├── tests/                 # 测试用例
├── pyproject.toml         # 项目配置
└── README.md
```

## 快速开始

### 安装

```bash
pip install -e .
```

### 配置

设置环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 或其他兼容API
export LEADER_MODEL="gpt-4o"
export PROPOSER_MODEL="gpt-4o-mini"
export CRITIC_MODEL="gpt-4o"
export EVALUATOR_ENDPOINT="http://localhost:18000/evaluate"
export EVALUATOR_TIMEOUT=300
```

### 运行示例

```python
import numpy as np
import asyncio
from alpha_mining.config import AlphaMiningConfig
from alpha_mining.workflow import run_mining

# 基线因子必须使用Python class格式（实现AlphaFactorTemplate接口）
baseline_factors = [
    {
        "id": "baseline-1",
        "name": "Momentum 20d",
        "code": '''class MomentumAlpha:
    def __init__(self, window: int = 20, **kwargs):
        self.window = window

    def compute(self, data: dict) -> np.ndarray:
        close = data["close"]
        return (close - close.shift(self.window)) / close.shift(self.window)

    def get_name(self) -> str:
        return f"Momentum_{self.window}d"
''',
        "description": "20日动量因子",
        "parameters": {"window": 20},
        "evaluation": {"ic_mean": 0.03, "sharpe": 0.8}
    }
]

# 创建配置
config = AlphaMiningConfig()
config.iteration.max_iterations = 20

# 运行挖掘
result = asyncio.run(run_mining(config, baseline_factors, use_parallel=True))
print(f"发现 {result['discovered_count']} 个新因子")
```

### 测试

```bash
pytest tests/ -v
```

## Data API (模块 1 + 2：行情数据接入)

`data_api/` 提供基于 akshare 的真实 A 股行情 HTTP 服务，对应 alpha_pipeline 的
Data IO + Featurizer 底层。落盘 parquet 缓存，新浪为主，东方财富兜底。

### 启动

```bash
pip install -e ".[data-api]"
data-api  # 默认监听 0.0.0.0:8001，可用 DATA_API_PORT 覆盖
```

### 支持市场 (`market` 参数)

| market       | 日线 | 分钟 | universe          | 符号示例                       |
| ------------ | ---- | ---- | ----------------- | ------------------------------ |
| `cn_stock`   | ✅   | ✅   | A-share / HS300 / ZZ500 / ZZ1000 / SSE50 | `600000`, `SHSE.600000`, `sh600000` |
| `us_stock`   | ✅   | ❌   | 全部 (~6k)        | `AAPL`, `US.AAPL`             |
| `hk_stock`   | ✅   | ❌   | 全部              | `00700`, `HK.00700`           |
| `cn_etf`     | ✅   | ❌   | 全部 (~1500)      | `510300`, `SHSE.510300`       |
| `cn_index`   | ✅   | ❌   | 别名列表          | `HS300`, `CYB`, `SHCOMP`      |
| `cn_future`  | ✅   | ❌   | 主力合约 ~20 个   | `IF0`, `RB0`, `CU0`           |
| `fx`         | ✅   | ❌   | USD/EUR/JPY/...   | `USD`, `EUR`, `HKD`           |

> 分钟数据仅 `cn_stock` 当前可用，且最多保留最近约 12 个交易日（akshare 新浪源限制）。

### 端点

| Method | Path                  | 说明                                  |
| ------ | --------------------- | ------------------------------------- |
| GET    | `/health`             | 健康检查                              |
| GET    | `/markets`            | 列出所有可用 market 名称              |
| GET    | `/universe`           | 成分股列表，按 `market` + `name` 查询 |
| GET    | `/trade_calendar`     | A 股区间内交易日 (YYYY-MM-DD)         |
| GET    | `/bars/minute`        | 单标的分钟 K 线（含 vwap）            |
| POST   | `/bars/minute/panel`  | 多标的分钟面板，最多 50 只            |
| GET    | `/bars/daily`         | 单标的日线，跨市场通用                |

### 示例

```bash
# A 股
curl "http://localhost:8001/bars/daily?market=cn_stock&symbol=SHSE.600519&start=2026-01-01&end=2026-05-12&adjust=qfq"
curl "http://localhost:8001/bars/minute?market=cn_stock&symbol=SHSE.600000&period=1"

# 美股
curl "http://localhost:8001/bars/daily?market=us_stock&symbol=AAPL&start=2026-01-01&end=2026-05-12"

# 港股
curl "http://localhost:8001/bars/daily?market=hk_stock&symbol=00700&start=2026-01-01&end=2026-05-12"

# ETF / 指数 / 期货 / 汇率
curl "http://localhost:8001/bars/daily?market=cn_etf&symbol=510300&start=2026-01-01&end=2026-05-12"
curl "http://localhost:8001/bars/daily?market=cn_index&symbol=HS300&start=2026-01-01&end=2026-05-12"
curl "http://localhost:8001/bars/daily?market=cn_future&symbol=IF0&start=2026-01-01&end=2026-05-12"
curl "http://localhost:8001/bars/daily?market=fx&symbol=USD&start=2026-01-01&end=2026-05-13"

# Universe
curl "http://localhost:8001/universe?market=us_stock&name=all"
curl "http://localhost:8001/universe?market=cn_stock&name=HS300&canonical=true"
```

字段与 backtest.md 对齐：分钟 bar 返回 `open/high/low/close/volume/amount/vwap`，
日线 bar 返回 `open/high/low/close/volume/amount/pct_chg/turnover_rate`。新增 market 只需要
在 `data_api/markets/` 下放一个文件并在 `markets/__init__.py` 注册。
### 真实LLM端到端Smoke Test

配置好真实 OpenAI Compatible API 后，可以一键启动 `data_api`、`back_test` 并运行一轮智能体挖掘：

```bash
export OPENAI_API_KEY="your-api-key"

./scripts/run_real_smoke.sh momentum_20d
```

脚本内已提供默认模型配置；如果需要覆盖，可以在命令前额外设置 `OPENAI_API_BASE`、`LEADER_MODEL`、`PROPOSER_MODEL`、`CRITIC_MODEL`。

为避免和本机已有服务冲突，脚本默认使用 `BACK_TEST_PORT=18000` 和 `DATA_API_PORT=18001`。如需改回旧端口，可显式设置：

```bash
BACK_TEST_PORT=8000 DATA_API_PORT=8001 ./scripts/run_real_smoke.sh momentum_20d
```

常用调试参数：

```bash
MAX_ITERATIONS=1 UNIVERSE=HS300 START_DATE=2023-01-01 END_DATE=2023-03-31 ./scripts/run_real_smoke.sh mean_reversion_20d
USE_SERIAL=1 ./scripts/run_real_smoke.sh momentum_20d
```

日志调试参数：

```bash
# 默认：INFO级别，并打印prompt/response摘要
ALPHA_MINING_LOG_LEVEL=INFO ALPHA_MINING_DEBUG_PROMPTS=1 ./scripts/run_real_smoke.sh momentum_20d

# 关闭prompt/response，只看节点状态和关键指标
ALPHA_MINING_DEBUG_PROMPTS=0 ./scripts/run_real_smoke.sh momentum_20d
```

如果你已经手动启动服务，也可以只运行 agent pipeline：

```bash
python scripts/real_llm_smoke_test.py --baseline momentum_20d --max-iterations 1 --serial
```

### Data API真实数据检查

如果怀疑回测一直落到 shadow 数据，可以先单独检查 `data_api` 是否返回真实日频行情：

```bash
# 先启动 data_api，例如：DATA_API_PORT=18001 python -m data_api.main
python scripts/check_data_api_real_data.py --base-url http://127.0.0.1:18001 --universe HS300 --start-date 2023-01-01 --end-date 2023-03-31
```

该脚本直接请求 `data_api /universe` 和 `data_api /bars/daily`，不经过 `back_test.data_loader`，因此不会被 shadow fallback 掩盖。

## 因子代码格式

所有因子必须实现 `AlphaFactorTemplate` 接口：

```python
import numpy as np

class MyAlpha:
    def __init__(self, window: int = 20, **kwargs):
        self.window = window

    def compute(self, data: dict) -> np.ndarray:
        """计算因子值

        Args:
            data: 包含 open, high, low, close, volume 等市场数据

        Returns:
            因子值数组，形状 (n_days, n_stocks)
        """
        close = data["close"]
        return (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()

    def get_name(self) -> str:
        return f"MomentumAlpha_{self.window}d"
```

## 对接回测框架

`alpha_mining.tools.eval_tools.call_evaluator` 会通过 HTTP 调用 `back_test` 服务的 `/evaluate` 接口。详见 `../docs/integration_spec.md`，包含HTTP接口规范、错误码定义和合规检查要求。

## 基线因子库

`tests/base_factors.py` 提供了可复用的日频基础因子库，用于 smoke test 和完整联调。

```python
from tests.base_factors import (
    get_base_factor_library,
    get_combined_base_factor_library,
)

baseline_factors = get_base_factor_library("momentum_20d")
all_baselines = get_combined_base_factor_library()
```

每个 helper 返回的因子都使用统一 Python class 格式，且可以直接传给 `run_mining(..., baseline_factor_library=baseline_factors)`。

## 主要接口

### Evaluator调用

```python
result = call_evaluator.invoke({
    "alpha_code": "class MyAlpha: ...",
    "alpha_description": "Z-score均值回归因子",
    "parameters": {"window": 20},
    "eval_config": {
        "alpha_id": "alpha-1",
        "universe": "HS300",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
    }
})
```

### 工作流构建

```python
from alpha_mining.workflow import build_mining_workflow, MiningState

# 支持并行模式
graph = build_mining_workflow(config, use_parallel=True)

initial_state: MiningState = {
    "session_id": "...",
    "baseline_factor_library": baseline_factors,
    # ...
}
```

## 配置选项

| 配置项                          | 默认值 | 说明                 |
| ------------------------------- | ------ | -------------------- |
| `max_iterations`              | 20     | 最大迭代轮次         |
| `min_proposals_per_iteration` | 3      | 每轮最小提案数       |
| `max_proposals_per_iteration` | 5      | 每轮最大提案数       |
| `max_proposer_retries`        | 3      | 合规错误最大重试次数 |
| `leader_model`                | gpt-4o | Leader模型           |
| `proposer_model`              | gpt-4o-mini | Proposer模型     |
| `critic_model`                | gpt-4o | Critic模型           |

## 数据模型

### AlphaExpression

```python
class AlphaExpression:
    id: str                    # 唯一标识符
    name: str                  # 因子名称
    code: str                  # Python class代码（实现AlphaFactorTemplate）
    description: str            # 数学/经济描述
    parameters: dict            # 参数配置
    optimization_rationale: str # 优化思路解释
    parent_id: str | None       # 父因子ID
```

### CriticFeedback

```python
class CriticFeedback:
    expected_match_score: float  # 预期匹配度（0.0-1.0）
    expected_match_reason: str    # 匹配度评估理由
    ratings: dict                # 各维度评分
    actionable_suggestions: list # 可执行建议
```

### LeaderDecision

```python
class LeaderDecision:
    selected_factor_id: str           # 本轮选择的基线因子ID
    factors_to_remove: list[str]     # 建议删除的因子
    removal_reasoning: str            # 删除理由
```

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码结构

- `alpha_mining/agents/` - 智能体实现（Leader、Proposer、Critic）
- `alpha_mining/prompts/` - 各智能体的系统提示词和用户提示词
- `alpha_mining/schemas/` - Pydantic数据模型
- `alpha_mining/tools/` - 存储工具和评估接口
- `alpha_mining/workflow.py` - LangGraph状态机工作流（支持并行）
