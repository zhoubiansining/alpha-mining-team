"""Alpha evaluation data model."""

from datetime import datetime

from pydantic import BaseModel, Field


class AlphaEvaluation(BaseModel):
    """Alpha评估结果 (来自Evaluator接口)"""

    alpha_id: str = Field(description="关联的alpha ID")
    eval_time: datetime = Field(default_factory=datetime.now, description="评估时间")

    # 核心指标
    ic_mean: float = Field(description="IC均值")
    ic_std: float = Field(description="IC标准差")
    ir: float = Field(description="IC IR")
    sharpe: float = Field(description="夏普比率")
    max_drawdown: float = Field(description="最大回撤")
    turnover: float = Field(description="换手率")

    # 额外指标
    long_short_return: float = Field(default=0.0, description="多空收益")
    win_rate: float = Field(default=0.0, description="胜率")

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }

    @classmethod
    def create_mock(
        cls,
        alpha_id: str,
        ic_mean: float = 0.05,
        ic_std: float = 0.1,
        sharpe: float = 1.0,
        max_drawdown: float = 0.1,
        turnover: float = 0.5,
    ) -> "AlphaEvaluation":
        """创建mock评估结果，用于测试"""
        return cls(
            alpha_id=alpha_id,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ic_mean / ic_std if ic_std > 0 else 0.0,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            turnover=turnover,
            long_short_return=sharpe * 0.1,
            win_rate=0.55,
        )

    def to_summary(self) -> dict:
        """转换为摘要字典"""
        return {
            "alpha_id": self.alpha_id,
            "ic_mean": self.ic_mean,
            "ir": self.ir,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
        }
