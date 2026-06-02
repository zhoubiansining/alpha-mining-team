"""Storage tools for managing alpha factors and history.

This module provides both:
1. Pure functions for internal use (can be called directly)
2. LangChain tools for agent interaction (use .invoke() method)
"""

from typing import Optional
import uuid
import json
from pathlib import Path

from langchain_core.tools import tool

from alpha_mining.schemas.alpha import AlphaExpression
from alpha_mining.schemas.evaluation import AlphaEvaluation
from alpha_mining.schemas.history import CriticFeedback, IterationHistory, MiningSession


# Global storage (in-memory for now, can be extended to file/db)
_storage: dict = {
    "alphas": {},          # alpha_id -> AlphaExpression
    "evaluations": {},     # alpha_id -> AlphaEvaluation
    "feedbacks": {},       # feedback_id -> CriticFeedback
    "sessions": {},        # session_id -> MiningSession
    "iterations": {},      # session_id -> list[IterationHistory]
    "baseline": [],        # list of baseline factor dicts (not stored as AlphaExpression)
    "discovered_alphas": [],  # list of discovered alpha_ids (for tracking)
}


def reset_storage():
    """重置存储，用于测试"""
    global _storage
    _storage = {
        "alphas": {},
        "evaluations": {},
        "feedbacks": {},
        "sessions": {},
        "iterations": {},
        "baseline": [],
        "discovered_alphas": [],
    }


# ============== Pure Functions (for internal use) ==============

def create_alpha(
    name: str,
    code: str,
    description: str,
    parameters: dict,
    iteration: int,
    parent_id: str | None = None,
    intuition: str = "",
    optimization_rationale: str = "",
    improvement_targets: list | None = None,
) -> AlphaExpression:
    """创建并存储一个alpha因子"""
    alpha = AlphaExpression(
        id=str(uuid.uuid4()),
        name=name,
        code=code,
        description=description,
        parameters=parameters,
        iteration=iteration,
        parent_id=parent_id,
        intuition=intuition,
        optimization_rationale=optimization_rationale,
        improvement_targets=improvement_targets or [],
    )
    _storage["alphas"][alpha.id] = alpha
    return alpha


def store_evaluation(alpha_id: str, metrics: dict) -> AlphaEvaluation:
    """存储评估结果"""
    evaluation = AlphaEvaluation(
        alpha_id=alpha_id,
        ic_mean=metrics.get("ic_mean", 0.0),
        ic_std=metrics.get("ic_std", 0.0),
        ir=metrics.get("ir", 0.0),
        sharpe=metrics.get("sharpe", 0.0),
        max_drawdown=metrics.get("max_drawdown", 0.0),
        turnover=metrics.get("turnover", 0.0),
        long_short_return=metrics.get("long_short_return", 0.0),
        win_rate=metrics.get("win_rate", 0.0),
    )
    _storage["evaluations"][alpha_id] = evaluation
    return evaluation


def store_feedback(
    alpha_id: str,
    iteration: int,
    ratings: dict,
    factual_observations: list,
    concerns: list,
    actionable_suggestions: list,
    can_proceed: bool,
    expected_match_score: float = 0.5,
    expected_match_reason: str = "",
) -> CriticFeedback:
    """存储Critic反馈"""
    feedback = CriticFeedback(
        id=str(uuid.uuid4()),
        alpha_id=alpha_id,
        iteration=iteration,
        ratings=ratings,
        factual_observations=factual_observations,
        concerns=concerns,
        actionable_suggestions=actionable_suggestions,
        can_proceed=can_proceed,
        expected_match_score=expected_match_score,
        expected_match_reason=expected_match_reason,
    )
    _storage["feedbacks"][feedback.id] = feedback
    return feedback


def delete_factor_record(alpha_id: str) -> bool:
    """
    删除指定因子及其关联数据。

    Args:
        alpha_id: 要删除的因子ID

    Returns:
        是否成功删除
    """
    if alpha_id not in _storage["alphas"]:
        return False

    # 删除因子本身
    del _storage["alphas"][alpha_id]

    # 删除关联的评估结果
    if alpha_id in _storage["evaluations"]:
        del _storage["evaluations"][alpha_id]

    # 删除关联的反馈（只删除alpha_id匹配的部分）
    feedbacks_to_delete = [
        fid for fid, fb in _storage["feedbacks"].items()
        if fb.alpha_id == alpha_id
    ]
    for fid in feedbacks_to_delete:
        del _storage["feedbacks"][fid]

    # 从discovered_alphas中移除
    if alpha_id in _storage["discovered_alphas"]:
        _storage["discovered_alphas"].remove(alpha_id)

    return True


def delete_factors(alpha_ids: list[str]) -> dict[str, bool]:
    """
    批量删除因子。

    Args:
        alpha_ids: 要删除的因子ID列表

    Returns:
        每个ID的删除结果
    """
    results = {}
    for alpha_id in alpha_ids:
        results[alpha_id] = delete_factor_record(alpha_id)
    return results


def list_factors() -> list[AlphaExpression]:
    """列出所有因子"""
    return list(_storage["alphas"].values())


def get_factor_by_id(alpha_id: str) -> AlphaExpression | None:
    """根据ID获取因子"""
    return _storage["alphas"].get(alpha_id)


def get_evaluation_by_alpha_id(alpha_id: str) -> AlphaEvaluation | None:
    """根据alpha ID获取评估结果"""
    return _storage["evaluations"].get(alpha_id)


def get_feedbacks_by_alpha_id(alpha_id: str) -> list[CriticFeedback]:
    """获取某个因子的所有反馈"""
    return [
        f for f in _storage["feedbacks"].values()
        if f.alpha_id == alpha_id
    ]


def create_session(config: dict) -> MiningSession:
    """创建挖掘会话"""
    from datetime import datetime
    session = MiningSession(
        session_id=str(uuid.uuid4()),
        start_time=datetime.now(),
        config=config,
        current_iteration=0,
        is_active=True,
    )
    _storage["sessions"][session.session_id] = session
    _storage["iterations"][session.session_id] = []
    return session


def store_iteration(session_id: str, history: IterationHistory) -> None:
    """存储迭代历史"""
    if session_id not in _storage["iterations"]:
        _storage["iterations"][session_id] = []
    _storage["iterations"][session_id].append(history)

    # Update session
    if session_id in _storage["sessions"]:
        _storage["sessions"][session_id].current_iteration = history.iteration


def finalize_mining_session(session_id: str, final_candidates: list[str]) -> None:
    """结束挖掘会话"""
    from datetime import datetime
    if session_id in _storage["sessions"]:
        session = _storage["sessions"][session_id]
        session.is_active = False
        session.final_candidates = final_candidates
        session.end_time = datetime.now()


def get_session_iterations(session_id: str) -> list[IterationHistory]:
    """获取会话的迭代历史"""
    return _storage["iterations"].get(session_id, [])


def set_baseline_factor_library(factors: list[dict]) -> None:
    """设置基线因子库（从外部输入的已有因子）"""
    _storage["baseline"] = factors
    _storage["discovered_alphas"] = []  # 每次新会话重置


def get_baseline_factor_library() -> list[dict]:
    """获取基线因子库"""
    return _storage["baseline"]


def get_discovered_alpha_ids() -> list[str]:
    """获取已发现因子的ID列表"""
    return _storage["discovered_alphas"]


def add_discovered_alpha(alpha_id: str) -> None:
    """添加已发现因子ID"""
    if alpha_id not in _storage["discovered_alphas"]:
        _storage["discovered_alphas"].append(alpha_id)


# ============== LangChain Tools (for agent use) ==============

@tool
def save_alpha(
    name: str,
    code: str,
    description: str,
    parameters: dict,
    iteration: int,
    parent_id: str | None = None,
    intuition: str = "",
    optimization_rationale: str = "",
    improvement_targets: list | None = None,
) -> str:
    """
    保存新生成的alpha因子。

    Returns:
        alpha_id: 生成的因子唯一标识符
    """
    alpha = create_alpha(
        name=name,
        code=code,
        description=description,
        parameters=parameters,
        iteration=iteration,
        parent_id=parent_id,
        intuition=intuition,
        optimization_rationale=optimization_rationale,
        improvement_targets=improvement_targets,
    )
    return alpha.id


@tool
def save_evaluation(
    alpha_id: str,
    metrics: dict,
) -> str:
    """
    保存alpha的评估结果。

    Returns:
        status: 保存状态
    """
    store_evaluation(alpha_id, metrics)
    return "success"


@tool
def save_critic_feedback(
    alpha_id: str,
    iteration: int,
    ratings: dict,
    factual_observations: list,
    concerns: list,
    actionable_suggestions: list,
    can_proceed: bool,
    expected_match_score: float = 0.5,
    expected_match_reason: str = "",
) -> str:
    """
    保存Critic的反馈。

    Returns:
        feedback_id: 反馈记录的唯一标识符
    """
    feedback = store_feedback(
        alpha_id=alpha_id,
        iteration=iteration,
        ratings=ratings,
        factual_observations=factual_observations,
        concerns=concerns,
        actionable_suggestions=actionable_suggestions,
        can_proceed=can_proceed,
        expected_match_score=expected_match_score,
        expected_match_reason=expected_match_reason,
    )
    return feedback.id


@tool
def get_baseline_factor_library() -> list[dict]:
    """
    获取基线因子库（外部提供的已有因子）。

    这些因子是优化任务的基准，需要尽可能地提升其表现。

    Returns:
        基线因子列表，每项包含id、name、description、code、evaluation等
    """
    return _storage["baseline"]


@tool
def get_factor_library() -> list[dict]:
    """
    获取当前因子库中所有因子的摘要信息。

    Returns:
        因子列表，每项包含id、name、iteration等摘要信息
    """
    result = []
    for alpha in _storage["alphas"].values():
        eval_summary = None
        if alpha.id in _storage["evaluations"]:
            eval_summary = _storage["evaluations"][alpha.id].to_summary()

        feedback_list = [
            f.to_summary() for f in _storage["feedbacks"].values()
            if f.alpha_id == alpha.id
        ]

        result.append({
            "id": alpha.id,
            "name": alpha.name,
            "description": alpha.description,
            "code": alpha.code,
            "iteration": alpha.iteration,
            "intuition": alpha.intuition,
            "evaluation": eval_summary,
            "feedbacks": feedback_list,
        })
    return result


@tool
def get_iteration_history(iteration: int | None = None) -> list[dict]:
    """
    获取迭代历史。

    Args:
        iteration: 指定轮次，None则返回全部

    Returns:
        迭代历史列表
    """
    result = []
    for session_iterations in _storage["iterations"].values():
        for hist in session_iterations:
            if iteration is None or hist.iteration == iteration:
                result.append(hist.model_dump())
    return result


@tool
def get_alpha_by_id(alpha_id: str) -> dict | None:
    """
    根据ID获取alpha详情。

    Returns:
        alpha详情或None
    """
    alpha = _storage["alphas"].get(alpha_id)
    if alpha is None:
        return None

    result = alpha.to_dict()
    if alpha_id in _storage["evaluations"]:
        result["evaluation"] = _storage["evaluations"][alpha_id].to_summary()

    feedbacks = [
        f.to_summary() for f in _storage["feedbacks"].values()
        if f.alpha_id == alpha_id
    ]
    result["feedbacks"] = feedbacks

    return result


@tool
def get_critic_suggestions_for_alpha(alpha_id: str) -> list[dict]:
    """
    获取某个alpha的所有Critic建议。

    Returns:
        建议列表
    """
    result = []
    for feedback in _storage["feedbacks"].values():
        if feedback.alpha_id == alpha_id:
            result.append({
                "id": feedback.id,
                "alpha_id": feedback.alpha_id,
                "iteration": feedback.iteration,
                "actionable_suggestions": feedback.actionable_suggestions,
                "concerns": feedback.concerns,
                "can_proceed": feedback.can_proceed,
            })
    return result


@tool
def create_mining_session(config: dict) -> str:
    """
    创建新的挖掘会话。

    Returns:
        session_id: 会话ID
    """
    session = create_session(config)
    return session.session_id


@tool
def save_iteration_history(
    session_id: str,
    iteration: int,
    leader_decision: str,
    optimization_direction: str,
    selected_for_context: list,
    feedback_to_proposer: list,
    proposed_alphas: list,
    evaluations: dict,
    critic_feedbacks: dict,
    is_complete: bool,
) -> str:
    """
    保存迭代历史。

    Returns:
        status: 保存状态
    """
    history = IterationHistory(
        iteration=iteration,
        leader_decision=leader_decision,
        optimization_direction=optimization_direction,
        selected_for_context=selected_for_context,
        feedback_to_proposer=feedback_to_proposer,
        proposed_alphas=proposed_alphas,
        evaluations=evaluations,
        critic_feedbacks=critic_feedbacks,
        is_complete=is_complete,
    )

    store_iteration(session_id, history)
    return "success"


@tool
def finalize_session(session_id: str, final_candidates: list[str]) -> str:
    """
    结束挖掘会话。

    Returns:
        status: 保存状态
    """
    finalize_mining_session(session_id, final_candidates)
    return "success"


@tool
def delete_factor(alpha_id: str) -> str:
    """
    删除指定因子及其所有关联数据（评估结果、反馈记录）。

    此操作需要谨慎，只有当因子确定无价值时才应调用。

    Args:
        alpha_id: 要删除的因子ID

    Returns:
        删除结果描述
    """
    success = delete_factor_record(alpha_id)
    if success:
        return f"Factor {alpha_id} and associated data deleted successfully"
    else:
        return f"Factor {alpha_id} not found, no deletion performed"


@tool
def get_factor_metrics(alpha_id: str) -> dict | None:
    """
    获取因子的评估指标摘要。

    Returns:
        包含评估指标的字典，或None（如果不存在）
    """
    eval_result = get_evaluation_by_alpha_id(alpha_id)
    if eval_result is None:
        return None

    return {
        "alpha_id": alpha_id,
        "ic_mean": eval_result.ic_mean,
        "ic_std": eval_result.ic_std,
        "ir": eval_result.ir,
        "sharpe": eval_result.sharpe,
        "max_drawdown": eval_result.max_drawdown,
        "turnover": eval_result.turnover,
        "long_short_return": eval_result.long_short_return,
        "win_rate": eval_result.win_rate,
    }
