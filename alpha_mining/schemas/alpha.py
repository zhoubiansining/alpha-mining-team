"""Alpha expression data model."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class AlphaFactorTemplate:
    """
    因子代码模板类。

    所有因子代码必须实现此模板定义的接口，
    以便回测框架能够统一执行。

    示例：
    ```python
    class MyAlpha(AlphaFactorTemplate):
        def __init__(self, window: int = 20):
            self.window = window

        def compute(self, data: dict) -> np.ndarray:
            close = data["close"]
            return (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()

        def get_name(self) -> str:
            return f"Momentum_{self.window}d"
    ```
    """

    def __init__(self, **kwargs):
        """初始化因子参数"""
        raise NotImplementedError("Subclass must implement __init__")

    def compute(self, data: dict) -> "np.ndarray":
        """
        计算因子值。

        Args:
            data: 包含市场数据的字典，键包括 open, high, low, close, volume 等

        Returns:
            因子值数组，形状为 (n_days, n_stocks)
        """
        raise NotImplementedError("Subclass must implement compute")

    def get_name(self) -> str:
        """获取因子名称"""
        raise NotImplementedError("Subclass must implement get_name")


class AlphaExpression(BaseModel):
    """Alpha因子表达式"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="唯一标识符 (UUID)")
    name: str = Field(description="因子名称")
    code: str = Field(description="Python class代码，实现AlphaFactorTemplate接口")
    description: str = Field(description="数学/经济描述")
    parameters: dict = Field(default_factory=dict, description="参数配置")

    # 元信息
    created_by: str = Field(default="proposer", description="创建者")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    iteration: int = Field(description="迭代轮次")
    parent_id: str | None = Field(default=None, description="父因子ID（本次优化基于哪个因子）")

    # 优化思路（Proposer提供）
    intuition: str = Field(default="", description="经济直觉描述")
    optimization_rationale: str = Field(
        default="",
        description="优化思路解释：为什么要这样设计因子，预期达到什么效果"
    )
    improvement_targets: list[str] = Field(
        default_factory=list,
        description="改进目标列表"
    )

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
            "optimization_rationale": self.optimization_rationale,
            "improvement_targets": self.improvement_targets,
        }

    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()

    def validate_code_format(self) -> bool:
        """
        验证代码格式是否符合规范。

        检查点：
        1. 代码是有效的Python class定义
        2. 类实现了compute方法
        3. 类实现了get_name方法
        """
        import ast

        try:
            tree = ast.parse(self.code)

            # 查找class定义
            class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if not class_names:
                return False

            # 检查class内部是否有compute和get_name方法
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    method_names = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    if "compute" in method_names and "get_name" in method_names:
                        return True

            return False
        except SyntaxError:
            return False
