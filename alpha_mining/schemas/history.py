"""Critic feedback data model."""

from datetime import datetime

from pydantic import BaseModel, Field


class CriticFeedback(BaseModel):
    """Critic的批评反馈"""

    id: str = Field(description="唯一标识符")
    alpha_id: str = Field(description="被评价的alpha ID")
    iteration: int = Field(description="迭代轮次")
    eval_time: datetime = Field(default_factory=datetime.now, description="评价时间")

    # 评分 (1-5)
    ratings: dict[str, int] = Field(
        default_factory=dict,
        description="各维度评分"
    )

    # 具体反馈
    factual_observations: list[str] = Field(
        default_factory=list,
        description="基于事实的观察"
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="关切问题"
    )
    actionable_suggestions: list[str] = Field(
        default_factory=list,
        description="可执行的建议"
    )

    # 决策
    can_proceed: bool = Field(description="是否可以进入下一阶段")

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }

    def to_summary(self) -> dict:
        """转换为摘要字典"""
        return {
            "id": self.id,
            "alpha_id": self.alpha_id,
            "iteration": self.iteration,
            "ratings": self.ratings,
            "concerns": self.concerns,
            "actionable_suggestions": self.actionable_suggestions,
            "can_proceed": self.can_proceed,
        }


class LeaderDecision(BaseModel):
    """Leader的决策输出"""

    should_continue: bool = Field(description="是否继续迭代")
    reason: str = Field(description="决策理由")

    # 优化方向
    optimization_direction: str | None = Field(
        default=None,
        description="本轮优化方向"
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="重点关注领域"
    )

    # 上下文管理
    selected_for_context: list[str] = Field(
        default_factory=list,
        description="选入探索上下文的因子ID"
    )
    reasoning_for_selection: str = Field(
        default="",
        description="选择理由"
    )

    # 反馈路由
    suggestions_to_proposer: list[str] = Field(
        default_factory=list,
        description="传递给Proposer的建议"
    )

    # 终止时
    final_candidates: list[str] | None = Field(
        default=None,
        description="最终候选因子"
    )
    termination_reason: str | None = Field(
        default=None,
        description="终止原因"
    )

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }


class IterationHistory(BaseModel):
    """单轮迭代历史"""

    iteration: int = Field(description="迭代轮次")

    # Leader决策
    leader_decision: str = Field(description="Leader的决策")
    optimization_direction: str = Field(description="优化方向描述")
    selected_for_context: list[str] = Field(
        default_factory=list,
        description="选入上下文的因子ID"
    )
    feedback_to_proposer: list[str] = Field(
        default_factory=list,
        description="传递给Proposer的建议"
    )

    # 本轮产出
    proposed_alphas: list[str] = Field(
        default_factory=list,
        description="本轮Proposer生成的因子ID"
    )
    evaluations: dict[str, dict] = Field(
        default_factory=dict,
        description="因子评估结果 (alpha_id -> eval_summary)"
    )
    critic_feedbacks: dict[str, dict] = Field(
        default_factory=dict,
        description="Critic反馈 (alpha_id -> feedback_summary)"
    )

    # 状态
    is_complete: bool = Field(description="本轮是否完成")

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }


class MiningSession(BaseModel):
    """整个挖掘会话"""

    session_id: str = Field(description="会话ID")
    start_time: datetime = Field(description="开始时间")

    # 配置
    config: dict = Field(default_factory=dict, description="会话配置")

    # 状态
    current_iteration: int = Field(default=0, description="当前轮次")
    is_active: bool = Field(default=True, description="是否进行中")

    # 历史
    iterations: list[IterationHistory] = Field(
        default_factory=list,
        description="各轮迭代历史"
    )

    # 最终结果
    final_candidates: list[str] = Field(
        default_factory=list,
        description="最终候选因子ID"
    )
    end_time: datetime | None = Field(default=None)

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }
