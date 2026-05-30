"""Configuration management."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """LLM模型配置"""
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI Compatible API基础URL"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="API密钥"
    )
    leader_model: str = Field(
        default="gpt-4o",
        description="Leader使用的模型"
    )
    proposer_model: str = Field(
        default="gpt-4o-mini",
        description="Proposer使用的模型"
    )
    critic_model: str = Field(
        default="gpt-4o",
        description="Critic使用的模型"
    )


class IterationConfig(BaseModel):
    """迭代配置"""
    max_iterations: int = Field(
        default=20,
        description="最大迭代轮次（每轮基于单个因子进行优化）"
    )
    min_proposals_per_iteration: int = Field(
        default=3,
        description="每轮最小提案数"
    )
    max_proposals_per_iteration: int = Field(
        default=5,
        description="每轮最大提案数"
    )


class EvaluatorConfig(BaseModel):
    """Evaluator配置"""
    endpoint: str = Field(
        default="http://localhost:8000/evaluate",
        description="Evaluator服务地址"
    )
    timeout: int = Field(
        default=300,
        description="评估超时时间（秒）"
    )
    use_mock: bool = Field(
        default=True,
        description="是否使用mock评估"
    )


class StorageConfig(BaseModel):
    """存储配置"""
    type: str = Field(
        default="memory",
        description="存储类型：memory, file, database"
    )
    path: str = Field(
        default="./data/sessions",
        description="存储路径"
    )


class AlphaMiningConfig(BaseModel):
    """Alpha挖掘完整配置"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    iteration: IterationConfig = Field(default_factory=IterationConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # 市场配置
    target_assets: list[str] = Field(
        default=["A-share"],
        description="目标资产类别"
    )
    eval_period: dict = Field(
        default_factory=lambda: {
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        },
        description="评估时间段"
    )

    @classmethod
    def from_env(cls) -> "AlphaMiningConfig":
        """从环境变量创建配置"""
        return cls(
            model=ModelConfig(
                api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                leader_model=os.getenv("LEADER_MODEL", "gpt-4o"),
                proposer_model=os.getenv("PROPOSER_MODEL", "gpt-4o-mini"),
                critic_model=os.getenv("CRITIC_MODEL", "gpt-4o"),
            ),
            iteration=IterationConfig(
                max_iterations=int(os.getenv("MAX_ITERATIONS", "20")),
                min_proposals_per_iteration=int(os.getenv("MIN_PROPOSALS", "3")),
                max_proposals_per_iteration=int(os.getenv("MAX_PROPOSALS", "5")),
            ),
        )
