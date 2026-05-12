"""Main workflow for alpha factor mining using LangGraph."""

import json
import logging
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END

from alpha_mining.config import AlphaMiningConfig
from alpha_mining.tools.eval_tools import call_evaluator
from alpha_mining.tools.storage_tools import (
    reset_storage,
    create_alpha,
    store_evaluation,
    store_feedback,
    list_factors,
    get_factor_by_id,
    get_feedbacks_by_alpha_id,
    get_evaluation_by_alpha_id,
    create_session,
    store_iteration,
    finalize_mining_session,
    set_baseline_factor_library,
    add_discovered_alpha,
)
from alpha_mining.schemas.history import IterationHistory
from alpha_mining.schemas.alpha import AlphaExpression
from alpha_mining.schemas.evaluation import AlphaEvaluation
from alpha_mining.agents.leader import build_leader_agent, parse_leader_decision
from alpha_mining.agents.proposer import build_proposer_agent, parse_alpha_proposals
from alpha_mining.agents.critic import parse_critic_feedback
from alpha_mining.prompts.leader_prompts import LEADER_ITERATION_PROMPT
from alpha_mining.prompts.proposer_prompts import PROPOSER_USER_PROMPT
from alpha_mining.prompts.critic_prompts import CRITIC_USER_PROMPT


logger = logging.getLogger(__name__)


class MiningState(TypedDict):
    """挖掘状态"""
    session_id: str
    iteration: int
    config: dict

    # 基线因子库（不可变的初始输入）
    baseline_factor_library: list[dict]

    # 当前轮次产出
    current_proposals: list[dict]
    pending_evaluations: list[str]

    # Leader决策
    leader_decision: Optional[dict]
    should_continue: bool
    optimization_direction: Optional[str]
    context_factors: list[dict]

    # 已发现的因子（探索过程中产生的）
    discovered_factors: list[str]

    # 控制
    is_complete: bool
    final_candidates: list[str]


def _build_baseline_summary(baseline: list[dict]) -> str:
    """构建基线因子库摘要"""
    if not baseline:
        return "No baseline factors provided. Empty library."

    summary_parts = ["## Baseline Factor Library\n"]
    for f in baseline:
        metrics = f.get("evaluation", {})
        summary_parts.append(
            f"- **{f.get('name', 'Unknown')}**: {f.get('description', '')}\n"
            f"  Code: `{f.get('code', '')}`\n"
            f"  Metrics: IC={metrics.get('ic_mean', 0):.4f}, IR={metrics.get('ir', 0):.2f}, Sharpe={metrics.get('sharpe', 0):.2f}\n"
        )
    return "\n".join(summary_parts)


def _build_factor_library_summary() -> str:
    """构建因子库摘要（已发现的因子）"""
    factors = list_factors()
    if not factors:
        return "No discovered factors yet."

    summary_parts = []
    for f in factors[-5:]:  # 最近5个
        eval_result = get_evaluation_by_alpha_id(f.id)
        eval_info = ""
        if eval_result:
            eval_info = f"IC={eval_result.ic_mean:.4f}, IR={eval_result.ir:.2f}, Sharpe={eval_result.sharpe:.2f}"

        summary_parts.append(
            f"- {f.name} (iter {f.iteration}): {eval_info}"
        )

    return "\n".join(summary_parts)


def _build_iteration_summary(iteration: int) -> str:
    """构建迭代历史摘要"""
    from alpha_mining.tools.storage_tools import get_session_iterations

    history = get_session_iterations(iteration)
    if not history:
        return "No history for this iteration."

    summary_parts = []
    for h in history:
        summary_parts.append(
            f"Iteration {h.iteration}: {h.leader_decision}\n"
            f"  Proposals: {len(h.proposed_alphas)}\n"
            f"  Optimization: {h.optimization_direction}"
        )

    return "\n".join(summary_parts)


def _build_proposals_summary(proposals: list[dict]) -> str:
    """构建提案摘要"""
    if not proposals:
        return "No proposals in this iteration."

    summary_parts = []
    for p in proposals:
        summary_parts.append(
            f"- {p.get('name', 'Unknown')}: {p.get('description', '')}"
        )

    return "\n".join(summary_parts)


def _build_feedback_summary() -> str:
    """构建反馈摘要"""
    factors = list_factors()
    if not factors:
        return "No feedback yet."

    summary_parts = []
    for f in factors[-3:]:  # 最近3个
        feedbacks = get_feedbacks_by_alpha_id(f.id)
        if feedbacks:
            fb = feedbacks[-1]
            summary_parts.append(
                f"- {f.name}: {fb.concerns[:2] if fb.concerns else 'No concerns'}"
            )

    return "\n".join(summary_parts) if summary_parts else "No feedback yet."


def leader_node(state: MiningState) -> MiningState:
    """
    Leader决策节点。
    评估当前状态，决定是否继续迭代，设置优化方向。
    决策基于基线因子库和已发现的因子。
    """
    logger.info(f"Leader node - Iteration {state['iteration']}")

    iteration = state["iteration"]
    config = state["config"]

    # 获取上下文信息
    discovered_summary = _build_factor_library_summary()
    baseline = state.get("baseline_factor_library", [])
    discovered_ids = state.get("discovered_factors", [])

    # 检查是否达到最大迭代次数
    if iteration >= config.get("max_iterations", 10):
        state["should_continue"] = False
        # 最终候选：已发现因子优先，但至少返回基线中最好的
        all_candidates = discovered_ids + [f.get("id", "") for f in baseline]
        state["leader_decision"] = {
            "should_continue": False,
            "reason": "Maximum iterations reached",
            "termination_reason": "Max iterations reached",
            "final_candidates": all_candidates[:10],
        }
        return state

    # 决策：基于基线库和已发现因子的对比
    state["should_continue"] = True
    state["optimization_direction"] = "Improve upon baseline factor library"
    state["context_factors"] = discovered_ids[-3:] if discovered_ids else []
    state["leader_decision"] = {
        "should_continue": True,
        "reason": f"Continue: {len(discovered_ids)} discovered, {len(baseline)} baseline",
        "optimization_direction": "Improve upon baseline factor library",
        "suggestions_to_proposer": [],
    }

    return state


def proposer_node(state: MiningState) -> MiningState:
    """
    Proposer生成节点。
    根据Leader的指示生成alpha候选。
    """
    logger.info(f"Proposer node - Iteration {state['iteration']}")

    factors = list_factors()
    config = state["config"]

    # 提案数量
    n_proposals = config.get("min_proposals_per_iteration", 3)

    # 简化处理：生成示例alpha（实际会通过agent调用LLM生成）
    sample_alphas = [
        {
            "name": f"Alpha_{state['iteration']}_{i+1}",
            "code": f"(close - ts_mean(close, {20+i*5})) / ts_std(close, {20+i*5})",
            "description": f"Mean reversion factor with window {20+i*5}",
            "parameters": {"window": 20+i*5},
            "intuition": "Captures mean reversion behavior",
            "improvement_targets": [],
        }
        for i in range(n_proposals)
    ]

    state["current_proposals"] = sample_alphas

    # 保存alpha
    for alpha in state["current_proposals"]:
        alpha_obj = create_alpha(
            name=alpha["name"],
            code=alpha["code"],
            description=alpha["description"],
            parameters=alpha.get("parameters", {}),
            iteration=state["iteration"],
            intuition=alpha.get("intuition", ""),
            improvement_targets=alpha.get("improvement_targets", []),
        )
        alpha["id"] = alpha_obj.id
        state["pending_evaluations"].append(alpha_obj.id)

    return state


def evaluator_node(state: MiningState) -> MiningState:
    """
    Evaluator节点。
    调用评估接口获取alpha的量化指标。
    包含合规性检查循环：不合规因子会返回error，触发proposer重新生成。
    """
    logger.info(f"Evaluator node - Iteration {state['iteration']}")

    max_retries = state["config"].get("max_proposer_retries", 3)

    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id")
        if not alpha_id:
            continue

        # 第一次尝试：直接调用Evaluator（同时做合规检查）
        result = call_evaluator.invoke({
            "alpha_description": alpha.get("description", ""),
            "alpha_code": alpha.get("code", ""),
            "parameters": alpha.get("parameters", {}),
            "eval_config": state["config"].get("eval_config", {}),
        })

        status = result.get("status")
        retry_count = alpha.get("compliance_retry_count", 0)

        # 合规性检查循环
        while status == "error" and result.get("error_code") == "COMPLIANCE_ERROR":
            retry_count += 1
            alpha["compliance_retry_count"] = retry_count
            if retry_count >= max_retries:
                # 达到最大重试次数，放弃此因子
                alpha["compliance_status"] = "failed"
                alpha["compliance_error"] = result.get("error_message")
                alpha["evaluator_status"] = "skipped"
                alpha["evaluation"] = None
                logger.warning(f"Alpha {alpha_id} failed compliance after {max_retries} retries")
                break

            # 将错误信息反馈给proposer（通过标记让下一个proposer调用知道需要修复）
            alpha["needs_fix"] = True
            alpha["compliance_error"] = result.get("error_message")
            alpha["error_code"] = result.get("error_code")
            alpha["evaluator_status"] = "retry"
            logger.info(f"Alpha {alpha_id} needs fix (retry {retry_count}/{max_retries}): {result.get('error_message')}")
            # 注意：实际重生成由proposer_node处理，这里只记录状态

            # 跳过当前因子的评估，等待proposer重生成
            break

        if status == "success":
            store_evaluation(alpha_id, result.get("metrics", {}))
            alpha["evaluation"] = result.get("metrics", {})
            alpha["evaluator_status"] = "success"
            alpha["compliance_status"] = "passed"
            # 标记为已发现
            if alpha_id not in state.get("discovered_factors", []):
                state.setdefault("discovered_factors", []).append(alpha_id)
                add_discovered_alpha(alpha_id)
        elif status == "error" and result.get("error_code") != "COMPLIANCE_ERROR":
            alpha["evaluator_status"] = "error"
            alpha["error_message"] = result.get("error_message")

    return state


def critic_node(state: MiningState) -> MiningState:
    """
    Critic节点。
    基于评估结果提供事实性批评。
    """
    logger.info(f"Critic node - Iteration {state['iteration']}")

    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id")
        if not alpha_id:
            continue

        # 获取评估结果
        eval_result = alpha.get("evaluation", {})
        sharpe = eval_result.get("sharpe", 0)

        # 构建Critic反馈
        feedback = parse_critic_feedback(
            response=json.dumps({
                "ratings": {
                    "theoretical_soundness": 4,
                    "backtest_quality": 3 if sharpe < 1.0 else 4,
                    "robustness": 3,
                    "implementation_quality": 5,
                    "diversification": 3,
                },
                "factual_observations": [
                    f"IC mean: {eval_result.get('ic_mean', 0):.4f}",
                    f"Sharpe ratio: {sharpe:.2f}",
                ],
                "concerns": [
                    "Need more diverse factors" if len(state["current_proposals"]) < 3 else "Consider parameter optimization",
                ],
                "actionable_suggestions": [
                    "Try different lookback windows",
                    "Consider volume-based factors",
                ],
                "can_proceed": sharpe > 0.5,
            }),
            alpha_id=alpha_id,
            iteration=state["iteration"],
        )

        # 保存反馈
        store_feedback(
            alpha_id=feedback["alpha_id"],
            iteration=feedback["iteration"],
            ratings=feedback["ratings"],
            factual_observations=feedback["factual_observations"],
            concerns=feedback["concerns"],
            actionable_suggestions=feedback["actionable_suggestions"],
            can_proceed=feedback["can_proceed"],
        )

        alpha["feedback"] = feedback

    return state


def should_continue_iteration(state: MiningState) -> str:
    """决定是否继续迭代"""
    if not state.get("should_continue", True):
        return "finalize"
    return "proposer"


def compliance_fix_node(state: MiningState) -> MiningState:
    """
    合规修复节点。
    对Evaluator判定为不合规的alpha，调用Proposer重新生成。
    """
    logger.info(f"Compliance fix node - Iteration {state['iteration']}")

    # 查找需要修复的alpha
    needs_fix = [a for a in state["current_proposals"] if a.get("needs_fix")]
    if not needs_fix:
        return state

    logger.info(f"Fixing {len(needs_fix)} non-compliant alphas")

    # 对每个不合规alpha进行修复
    for alpha in needs_fix:
        error_msg = alpha.get("compliance_error", "Unknown compliance error")
        error_code = alpha.get("error_code", "COMPLIANCE_ERROR")
        alpha_id = alpha.get("id", "")

        # 简化处理：生成修复后的alpha（实际会通过agent调用LLM生成）
        # 这里基于原始代码做小改动来规避合规问题
        old_code = alpha.get("code", "")
        fixed_code = f"# Fixed compliance issue for: {alpha.get('name', '')}\n{old_code}"

        fixed_alpha = {
            "name": alpha.get("name", "") + "_fixed",
            "code": fixed_code,
            "description": alpha.get("description", "") + f" [fixed after {error_code}]",
            "parameters": alpha.get("parameters", {}),
            "intuition": alpha.get("intuition", ""),
            "improvement_targets": alpha.get("improvement_targets", []),
            "is_compliance_fix": True,
            "parent_id": alpha_id,
            "compliance_error": error_msg,
            "compliance_retry_count": alpha.get("compliance_retry_count", 0),
        }

        # 删除旧的，添加修复后的
        idx = state["current_proposals"].index(alpha)
        alpha_obj = create_alpha(
            name=fixed_alpha["name"],
            code=fixed_alpha["code"],
            description=fixed_alpha["description"],
            parameters=fixed_alpha["parameters"],
            iteration=state["iteration"],
            intuition=fixed_alpha["intuition"],
            improvement_targets=fixed_alpha["improvement_targets"],
        )
        fixed_alpha["id"] = alpha_obj.id
        fixed_alpha["evaluator_status"] = "pending"
        state["current_proposals"][idx] = fixed_alpha
        state["pending_evaluations"].append(alpha_obj.id)

        logger.info(f"Regenerated {alpha_id} -> {alpha_obj.id} (compliance fix)")

    return state


def _should_retry_compliance(state: MiningState) -> str:
    """检查是否需要重试合规检查"""
    needs_fix = [a for a in state["current_proposals"] if a.get("needs_fix")]
    has_pending = [a for a in state["current_proposals"] if a.get("evaluator_status") == "pending"]
    has_retry = [a for a in state["current_proposals"] if a.get("evaluator_status") == "retry"]

    if needs_fix or has_retry:
        return "compliance_fix"
    return "critic"


def finalize_node(state: MiningState) -> MiningState:
    """最终化节点"""
    state["is_complete"] = True

    # 收集最终候选：基线 + 已发现的因子（去重）
    baseline_ids = [f.get("id", "") for f in state.get("baseline_factor_library", [])]
    discovered_ids = state.get("discovered_factors", [])
    all_ids = list(dict.fromkeys(baseline_ids + discovered_ids))  # 保持顺序去重

    state["final_candidates"] = all_ids
    return state


def build_mining_workflow(config: AlphaMiningConfig | None = None) -> Any:
    """
    构建挖掘工作流。

    Args:
        config: 配置对象

    Returns:
        编译后的工作流图
    """
    if config is None:
        config = AlphaMiningConfig()

    workflow = StateGraph(MiningState)

    # 添加节点
    workflow.add_node("leader", leader_node)
    workflow.add_node("proposer", proposer_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("compliance_fix", compliance_fix_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalize", finalize_node)

    # 定义边
    workflow.set_entry_point("leader")

    # Leader决定是否继续
    workflow.add_conditional_edges(
        "leader",
        should_continue_iteration,
        {
            "proposer": "proposer",
            "finalize": "finalize",
        }
    )

    # 顺序执行
    workflow.add_edge("proposer", "evaluator")
    workflow.add_edge("evaluator", "compliance_fix")

    # 合规修复循环：可能回到evaluator再次检查
    workflow.add_conditional_edges(
        "compliance_fix",
        _should_retry_compliance,
        {
            "compliance_fix": "evaluator",
            "critic": "critic",
        }
    )

    workflow.add_edge("critic", "leader")

    # 结束
    workflow.add_edge("finalize", END)

    return workflow.compile()


async def run_mining(
    config: AlphaMiningConfig | None = None,
    baseline_factor_library: list[dict] | None = None,
) -> dict:
    """
    运行完整的挖掘流程。

    Args:
        config: 配置对象
        baseline_factor_library: 基线因子库（从外部输入的已有因子），每项包含
            id, name, code, description, evaluation 等字段

    Returns:
        运行结果
    """
    if config is None:
        config = AlphaMiningConfig()

    # 重置存储
    reset_storage()

    # 设置基线因子库
    baseline = baseline_factor_library or []
    set_baseline_factor_library(baseline)

    # 创建会话
    session = create_session(config.model_dump())
    session_id = session.session_id

    # 构建工作流
    graph = build_mining_workflow(config)

    # 初始化状态
    initial_state: MiningState = {
        "session_id": session_id,
        "iteration": 1,
        "config": {
            "max_iterations": config.iteration.max_iterations,
            "min_proposals_per_iteration": config.iteration.min_proposals_per_iteration,
            "max_proposals_per_iteration": config.iteration.max_proposals_per_iteration,
            "max_proposer_retries": 3,
            "eval_config": {
                "symbols": config.target_assets,
                "start_date": config.eval_period["start_date"],
                "end_date": config.eval_period["end_date"],
            },
        },
        "baseline_factor_library": baseline,
        "current_proposals": [],
        "pending_evaluations": [],
        "leader_decision": None,
        "should_continue": True,
        "optimization_direction": None,
        "context_factors": [],
        "discovered_factors": [],
        "is_complete": False,
        "final_candidates": [],
    }

    # 运行工作流
    final_state = await graph.ainvoke(initial_state)

    # 保存最终状态
    if final_state.get("final_candidates"):
        finalize_mining_session(session_id, final_state["final_candidates"])

    # 收集所有因子（基线 + 新发现）
    all_factor_ids = final_state.get("final_candidates", [])
    all_factors = []

    # 包含基线因子信息
    for f in baseline:
        all_factors.append({
            "id": f.get("id", ""),
            "name": f.get("name", ""),
            "code": f.get("code", ""),
            "description": f.get("description", ""),
            "is_baseline": True,
            "evaluation": f.get("evaluation", {}),
        })

    # 包含新发现因子信息
    for alpha_id in final_state.get("discovered_factors", []):
        alpha = get_factor_by_id(alpha_id)
        if alpha:
            eval_result = get_evaluation_by_alpha_id(alpha_id)
            all_factors.append({
                "id": alpha.id,
                "name": alpha.name,
                "code": alpha.code,
                "description": alpha.description,
                "is_baseline": False,
                "evaluation": eval_result.to_summary() if eval_result else {},
            })

    return {
        "session_id": session_id,
        "iterations": final_state["iteration"],
        "baseline_count": len(baseline),
        "discovered_count": len(final_state.get("discovered_factors", [])),
        "factors": all_factors,
        "final_candidates": all_factor_ids,
        "termination_reason": final_state.get("leader_decision", {}).get("termination_reason"),
    }
