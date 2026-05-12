# Alpha Mining Team

基于多LLM智能体团队的自动化金融量化因子挖掘系统。

## 项目简介

本系统采用对抗性反馈循环架构，由三个专业智能体（Leader、Proposer、Critic）协同工作，从给定的基线因子库出发，迭代优化生成更优的因子组合。

### 核心特性

- **对抗性反馈循环**：Proposer生成候选因子 → Evaluator评估 → Critic提供事实性批评
- **Leader智能决策**：基于因子库表现和Critic反馈动态调整优化方向
- **合规性检查**：自动过滤不合规因子，支持重试修复
- **灵活的评估接口**：通过HTTP API对接外部回测框架

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
| **Leader**   | 决策是否继续迭代、设定优化方向、选择最终候选因子 |
| **Proposer** | 基于Leader指令生成新因子候选                     |
| **Critic**   | 基于评估结果提供事实性批评和改进建议             |

## 目录结构

```
.
├── alpha_mining/          # 核心代码
│   ├── agents/           # 智能体实现
│   ├── config.py         # 配置管理
│   ├── prompts/          # 智能体提示词
│   ├── schemas/          # 数据模型
│   ├── tools/            # 工具函数
│   └── workflow.py       # LangGraph工作流
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

设置环境变量（可选）：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 或其他兼容API
```

### 运行示例

```python
from alpha_mining.config import AlphaMiningConfig
from alpha_mining.workflow import run_mining

# 定义基线因子库
baseline_factors = [
    {
        "id": "baseline-1",
        "name": "Momentum 20d",
        "code": "(close - ts_delay(close, 20)) / ts_delay(close, 20)",
        "description": "20日动量因子",
        "evaluation": {"ic_mean": 0.03, "sharpe": 0.8}
    }
]

# 创建配置
config = AlphaMiningConfig()
config.iteration.max_iterations = 10

# 运行挖掘
result = await run_mining(config, baseline_factors)
```

### 测试

```bash
pytest tests/ -v
```

## 对接回测框架

详见 `../docs/integration_spec.md`，包含HTTP接口规范、错误码定义和合规检查要求。

## 主要接口

### Evaluator调用

```python
result = call_evaluator.invoke({
    "alpha_code": "(close - ts_mean(close, 20)) / ts_std(close, 20)",
    "alpha_description": "Z-score均值回归因子",
    "parameters": {"window": 20},
    "eval_config": {"universe": "A-share", ...}
})
```

### 工作流构建

```python
from alpha_mining.workflow import build_mining_workflow, MiningState

graph = build_mining_workflow(config)

initial_state: MiningState = {
    "session_id": "...",
    "baseline_factor_library": baseline_factors,
    # ...
}
```

## 配置选项

| 配置项                          | 默认值 | 说明                 |
| ------------------------------- | ------ | -------------------- |
| `max_iterations`              | 10     | 最大迭代轮次         |
| `min_proposals_per_iteration` | 3      | 每轮最小提案数       |
| `max_proposals_per_iteration` | 5      | 每轮最大提案数       |
| `max_proposer_retries`        | 3      | 合规错误最大重试次数 |

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码结构

- `alpha_mining/agents/` - 智能体实现，包含Leader、Proposer、Critic
- `alpha_mining/prompts/` - 各智能体的系统提示词和用户提示词
- `alpha_mining/schemas/` - Pydantic数据模型
- `alpha_mining/tools/` - 存储工具和评估接口
- `alpha_mining/workflow.py` - LangGraph状态机工作流
