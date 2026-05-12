"""Alpha expression data model."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class AlphaExpression(BaseModel):
    """Alpha因子表达式"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="唯一标识符 (UUID)")
    name: str = Field(description="因子名称")
    code: str = Field(description="Python/numpy表达式")
    description: str = Field(description="数学描述")
    parameters: dict = Field(default_factory=dict, description="参数配置")

    # 元信息
    created_by: str = Field(default="proposer", description="创建者")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    iteration: int = Field(description="迭代轮次")
    parent_ids: list[str] = Field(default_factory=list, description="父因子ID列表")
    intuition: str = Field(default="", description="经济直觉描述")
    improvement_targets: list[str] = Field(default_factory=list, description="改进目标列表")

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }

    def to_summary(self) -> dict:
        """转换为摘要字典，用于显示和上下文"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "iteration": self.iteration,
            "intuition": self.intuition,
            "improvement_targets": self.improvement_targets,
        }

    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()
