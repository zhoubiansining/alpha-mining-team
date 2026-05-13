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
```

### 运行示例

```python
import asyncio
from alpha_mining.config import AlphaMiningConfig
from alpha_mining.workflow import run_mining

# 定义基线因子库
baseline_factors = [
    {
        "id": "baseline-1",
        "name": "Momentum 20d",
        "code": "(close - close.shift(20)) / close.shift(20)",
        "description": "20日动量因子",
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

详见 `../docs/integration_spec.md`，包含HTTP接口规范、错误码定义和合规检查要求。

## 主要接口

### Evaluator调用

```python
result = call_evaluator.invoke({
    "alpha_code": "(close - close.rolling(20).mean()) / close.rolling(20).std()",
    "alpha_description": "Z-score均值回归因子",
    "parameters": {"window": 20},
    "eval_config": {"universe": "A-share", ...}
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
