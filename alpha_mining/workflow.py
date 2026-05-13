"""Main workflow for alpha factor mining using LangGraph with real Agent calls."""

import asyncio
import json
import logging
from typing import TypedDict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

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
    get_baseline_factor_library,
    delete_factor,
)
from alpha_mining.schemas.history import IterationHistory
from alpha_mining.schemas.alpha import AlphaExpression
from alpha_mining.schemas.evaluation import AlphaEvaluation
from alpha_mining.agents.leader import build_leader_agent, parse_leader_decision
from alpha_mining.agents.proposer import build_proposer_agent, parse_alpha_proposals
from alpha_mining.agents.critic import build_critic_agent, parse_critic_feedback
from alpha_mining.prompts.leader_prompts import LEADER_ITERATION_PROMPT
from alpha_mining.prompts.proposer_prompts import PROPOSER_USER_PROMPT
from alpha_mining.prompts.critic_prompts import CRITIC_USER_PROMPT


logger = logging.getLogger(__name__)

# Thread pool for parallel LLM calls
_executor = ThreadPoolExecutor(max_workers=10)


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
    selected_factor_id: Optional[str]
    reasoning_for_selection: Optional[str]
    suggestions_to_proposer: list[str]
    factors_to_remove: list[str]

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
            f"- **{f.get('name', 'Unknown')}** (ID: {f.get('id', 'N/A')})\n"
            f"  Description: {f.get('description', '')}\n"
            f"  Code: ```{f.get('code', '')}```\n"
            f"  Metrics: IC={metrics.get('ic_mean', 0):.4f}, IR={metrics.get('ir', 0):.2f}, "
            f"Sharpe={metrics.get('sharpe', 0):.2f}\n"
        )
    return "\n".join(summary_parts)


def _build_discovered_factors_summary() -> str:
    """构建已发现因子摘要"""
    factors = list_factors()
    if not factors:
        return "No discovered factors yet."

    summary_parts = ["## Discovered Factors\n"]
    for f in factors[-10:]:  # 最近10个
        eval_result = get_evaluation_by_alpha_id(f.id)
        eval_info = ""
        if eval_result:
            eval_info = f"IC={eval_result.ic_mean:.4f}, IR={eval_result.ir:.2f}, Sharpe={eval_result.sharpe:.2f}"

        summary_parts.append(
            f"- **{f.name}** (ID: {f.id}, iter {f.iteration}): {eval_info}"
        )

    return "\n".join(summary_parts)


def _build_iteration_context(state: MiningState) -> dict:
    """构建迭代上下文"""
    baseline = state.get("baseline_factor_library", [])
    discovered = list_factors()
    iteration = state["iteration"]

    # 计算已发现因子数量
    discovered_count = len([f for f in discovered if f.iteration < iteration])

    # 获取最近的反馈
    recent_feedbacks = []
    for f in discovered[-3:]:
        feedbacks = get_feedbacks_by_alpha_id(f.id)
        if feedbacks:
            fb = feedbacks[-1]
            recent_feedbacks.append({
                "alpha_id": f.id,
                "alpha_name": f.name,
                "suggestions": fb.actionable_suggestions,
                "expected_match_score": fb.expected_match_score if hasattr(fb, 'expected_match_score') else None,
            })

    # 构建指标概览
    baseline_metrics = []
    for f in baseline[:5]:
        metrics = f.get("evaluation", {})
        baseline_metrics.append({
            "name": f.get("name"),
            "ic_mean": metrics.get("ic_mean", 0),
            "sharpe": metrics.get("sharpe", 0),
        })

    discovered_metrics = []
    for f in discovered[-5:]:
        eval_result = get_evaluation_by_alpha_id(f.id)
        if eval_result:
            discovered_metrics.append({
                "name": f.name,
                "ic_mean": eval_result.ic_mean,
                "sharpe": eval_result.sharpe,
            })

    return {
        "discovered_count": discovered_count,
        "recent_feedbacks": recent_feedbacks,
        "baseline_metrics": baseline_metrics,
        "discovered_metrics": discovered_metrics,
    }


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


async def _call_leader_agent(leader_prompt: str, config: dict) -> dict:
    """异步调用Leader Agent"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_call_leader,
        leader_prompt,
        config
    )


def _sync_call_leader(leader_prompt: str, config: dict) -> dict:
    """同步调用Leader Agent"""
    try:
        agent = build_leader_agent(
            model_name=config.get("leader_model", "gpt-4o"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )
        response = agent.invoke(leader_prompt)
        return parse_leader_decision(response)
    except Exception as e:
        logger.error(f"Leader agent error: {e}")
        return {
            "should_continue": False,
            "reason": f"Leader error: {str(e)}",
            "termination_reason": "Agent error",
        }


async def _call_proposer_agent(
    proposer_prompt: str,
    config: dict,
    n_proposals: int,
) -> list[dict]:
    """异步调用Proposer Agent"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_call_proposer,
        proposer_prompt,
        config,
        n_proposals
    )


def _sync_call_proposer(proposer_prompt: str, config: dict, n_proposals: int) -> list[dict]:
    """同步调用Proposer Agent"""
    try:
        agent = build_proposer_agent(
            model_name=config.get("proposer_model", "gpt-4o-mini"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )
        response = agent.invoke(proposer_prompt)
        alphas = parse_alpha_proposals(response)
        return alphas[:n_proposals]  # 限制数量
    except Exception as e:
        logger.error(f"Proposer agent error: {e}")
        return []


async def _call_critic_agent(
    alpha: dict,
    evaluation: dict,
    baseline: list[dict],
    optimization_direction: str,
    config: dict,
) -> dict:
    """异步调用Critic Agent"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_call_critic,
        alpha,
        evaluation,
        baseline,
        optimization_direction,
        config
    )


def _sync_call_critic(
    alpha: dict,
    evaluation: dict,
    baseline: list[dict],
    optimization_direction: str,
    config: dict,
) -> dict:
    """同步调用Critic Agent"""
    try:
        agent = build_critic_agent(
            model_name=config.get("critic_model", "gpt-4o"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )

        # 构建baseline摘要
        baseline_summary = _build_baseline_summary(baseline)

        # 构建用户prompt
        user_prompt = CRITIC_USER_PROMPT.format(
            alpha_id=alpha.get("id", ""),
            alpha_name=alpha.get("name", ""),
            description=alpha.get("description", ""),
            code=alpha.get("code", ""),
            optimization_rationale=alpha.get("optimization_rationale", ""),
            evaluation_results=json.dumps(evaluation, indent=2),
            baseline_factor_library=baseline_summary,
            optimization_direction=optimization_direction or "General alpha improvement",
        )

        response = agent.invoke(user_prompt)
        return parse_critic_feedback(
            response,
            alpha_id=alpha.get("id", ""),
            iteration=config.get("iteration", 1),
        )
    except Exception as e:
        logger.error(f"Critic agent error: {e}")
        return {
            "alpha_id": alpha.get("id", ""),
            "iteration": config.get("iteration", 1),
            "ratings": {},
            "factual_observations": [f"Critic error: {str(e)}"],
            "concerns": [],
            "actionable_suggestions": [],
            "can_proceed": False,
        }


def leader_node(state: MiningState) -> MiningState:
    """
    Leader决策节点。
    使用真实的Leader Agent进行决策。
    """
    logger.info(f"Leader node - Iteration {state['iteration']}")

    iteration = state["iteration"]
    config = state["config"]
    baseline = state.get("baseline_factor_library", [])

    # 检查是否达到最大迭代次数
    max_iterations = config.get("max_iterations", 20)
    if iteration > max_iterations:
        all_candidates = state.get("discovered_factors", []) + [f.get("id", "") for f in baseline]
        state["should_continue"] = False
        state["leader_decision"] = {
            "should_continue": False,
            "reason": "Maximum iterations reached",
            "termination_reason": "Max iterations reached",
            "final_candidates": list(dict.fromkeys(all_candidates))[:20],
        }
        return state

    # 构建上下文
    context = _build_iteration_context(state)
    baseline_summary = _build_baseline_summary(baseline)
    discovered_summary = _build_discovered_factors_summary()

    # 构建迭代提示词
    leader_prompt = LEADER_ITERATION_PROMPT.format(
        iteration=iteration,
        max_iterations=max_iterations,
        baseline_factor_library=baseline_summary,
        discovered_count=context["discovered_count"],
        new_candidates_count=len(state.get("current_proposals", [])),
        recent_proposals=_build_proposals_summary(state.get("current_proposals", [])),
        recent_feedbacks=json.dumps(context["recent_feedbacks"], indent=2),
        metrics_overview=f"Baseline: {json.dumps(context['baseline_metrics'])}\nDiscovered: {json.dumps(context['discovered_metrics'])}",
    )

    # 调用Leader Agent
    try:
        decision = asyncio.run(_call_leader_agent(leader_prompt, config))
    except RuntimeError:
        # 如果已经在事件循环中，直接await
        decision = asyncio.get_event_loop().run_until_complete(
            _call_leader_agent(leader_prompt, config)
        )

    # 更新状态
    state["should_continue"] = decision.get("should_continue", False)
    state["optimization_direction"] = decision.get("optimization_direction")
    state["leader_decision"] = decision
    state["selected_factor_id"] = decision.get("selected_factor_id")
    state["reasoning_for_selection"] = decision.get("reasoning_for_selection", "")
    state["suggestions_to_proposer"] = decision.get("suggestions_to_proposer", [])
    state["factors_to_remove"] = decision.get("factors_to_remove", [])

    # 处理因子库管理
    if state["factors_to_remove"]:
        for factor_id in state["factors_to_remove"]:
            delete_factor(factor_id)
            if factor_id in state["discovered_factors"]:
                state["discovered_factors"].remove(factor_id)
        logger.info(f"Removed {len(state['factors_to_remove'])} factors from library")

    # 如果是终止决策
    if not state["should_continue"]:
        all_candidates = state.get("discovered_factors", []) + [f.get("id", "") for f in baseline]
        final_candidates = decision.get("final_candidates") or list(dict.fromkeys(all_candidates))
        state["leader_decision"]["final_candidates"] = final_candidates
        state["final_candidates"] = final_candidates

    logger.info(f"Leader decision: continue={state['should_continue']}, direction={state['optimization_direction']}")
    return state


def proposer_node(state: MiningState) -> MiningState:
    """
    Proposer生成节点。
    使用真实的Proposer Agent生成alpha候选。
    """
    logger.info(f"Proposer node - Iteration {state['iteration']}")

    config = state["config"]
    baseline = state.get("baseline_factor_library", [])
    iteration = state["iteration"]

    # 获取要优化的基线因子
    selected_factor = None
    if state.get("selected_factor_id"):
        for f in baseline:
            if f.get("id") == state["selected_factor_id"]:
                selected_factor = f
                break
        if not selected_factor:
            # 在已发现因子中查找
            for f in list_factors():
                if f.id == state["selected_factor_id"]:
                    eval_result = get_evaluation_by_alpha_id(f.id)
                    selected_factor = {
                        "id": f.id,
                        "name": f.name,
                        "description": f.description,
                        "code": f.code,
                        "evaluation": eval_result.to_summary() if eval_result else {},
                    }
                    break

    # 构建优化方向和反馈
    optimization_direction = state.get("optimization_direction", "Improve alpha factors")
    suggestions = state.get("suggestions_to_proposer", [])

    # 构建反馈摘要
    critic_suggestions = ""
    if suggestions:
        critic_suggestions = "## Suggestions from Critic\n"
        for s in suggestions:
            critic_suggestions += f"- {s}\n"

    # 获取提案数量
    n_proposals = config.get("min_proposals_per_iteration", 3)

    # 构建Proposer提示词
    proposer_prompt = PROPOSER_USER_PROMPT.format(
        optimization_direction=optimization_direction,
        selected_factor_id=selected_factor.get("id", "N/A") if selected_factor else "N/A",
        selected_factor_name=selected_factor.get("name", "N/A") if selected_factor else "N/A",
        selected_factor_description=selected_factor.get("description", "") if selected_factor else "",
        selected_factor_code=selected_factor.get("code", "") if selected_factor else "",
        selected_factor_metrics=json.dumps(selected_factor.get("evaluation", {})) if selected_factor else "{}",
        critic_suggestions=critic_suggestions or "No specific suggestions.",
        n=n_proposals,
    )

    # 调用Proposer Agent
    try:
        alphas = asyncio.run(_call_proposer_agent(proposer_prompt, config, n_proposals))
    except RuntimeError:
        alphas = asyncio.get_event_loop().run_until_complete(
            _call_proposer_agent(proposer_prompt, config, n_proposals)
        )

    if not alphas:
        logger.warning("Proposer returned no alphas, using fallback")
        alphas = _generate_fallback_alphas(state, n_proposals)

    # 保存alpha
    state["current_proposals"] = []
    state["pending_evaluations"] = []

    for alpha in alphas:
        alpha_obj = create_alpha(
            name=alpha.get("name", f"Alpha_{iteration}"),
            code=alpha.get("code", ""),
            description=alpha.get("description", ""),
            parameters=alpha.get("parameters", {}),
            iteration=iteration,
            parent_id=selected_factor.get("id") if selected_factor else None,
            intuition=alpha.get("intuition", ""),
            optimization_rationale=alpha.get("optimization_rationale", ""),
            improvement_targets=alpha.get("improvement_targets", []),
        )
        alpha["id"] = alpha_obj.id
        alpha["parent_id"] = selected_factor.get("id") if selected_factor else None
        state["current_proposals"].append(alpha)
        state["pending_evaluations"].append(alpha_obj.id)

    logger.info(f"Proposer generated {len(state['current_proposals'])} alphas")
    return state


def _generate_fallback_alphas(state: MiningState, n: int) -> list[dict]:
    """生成备用alpha（当Agent调用失败时）"""
    iteration = state["iteration"]
    return [
        {
            "name": f"FallbackAlpha_{iteration}_{i+1}",
            "code": f"(close - close.rolling({20+i*10}).mean()) / close.rolling({20+i*10}).std()",
            "description": f"Fallback mean reversion factor with window {20+i*10}",
            "parameters": {"window": 20+i*10},
            "intuition": "Captures mean reversion behavior",
            "optimization_rationale": "Fallback due to proposer error",
            "improvement_targets": [],
        }
        for i in range(n)
    ]


def evaluator_node(state: MiningState) -> MiningState:
    """
    Evaluator节点。
    调用评估接口获取alpha的量化指标。
    """
    logger.info(f"Evaluator node - Iteration {state['iteration']}, evaluating {len(state['current_proposals'])} alphas")

    max_retries = state["config"].get("max_proposer_retries", 3)

    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id")
        if not alpha_id:
            continue

        # 调用Evaluator
        result = call_evaluator.invoke({
            "alpha_description": alpha.get("description", ""),
            "alpha_code": alpha.get("code", ""),
            "parameters": alpha.get("parameters", {}),
            "eval_config": state["config"].get("eval_config", {}),
        })

        status = result.get("status")
        alpha["compliance_status"] = "passed" if status == "success" else "failed"
        alpha["evaluator_status"] = status

        if status == "success":
            metrics = result.get("metrics", {})
            store_evaluation(alpha_id, metrics)
            alpha["evaluation"] = metrics

            # 标记为已发现
            if alpha_id not in state.get("discovered_factors", []):
                state.setdefault("discovered_factors", []).append(alpha_id)
                add_discovered_alpha(alpha_id)
        else:
            alpha["evaluation"] = None
            alpha["error_message"] = result.get("error_message")
            alpha["error_code"] = result.get("error_code")

            # 合规错误处理
            if result.get("error_code") == "COMPLIANCE_ERROR":
                alpha["needs_fix"] = True
                alpha["compliance_retry_count"] = alpha.get("compliance_retry_count", 0) + 1
                if alpha["compliance_retry_count"] >= max_retries:
                    alpha["evaluator_status"] = "skipped"
                    logger.warning(f"Alpha {alpha_id} failed compliance after {max_retries} retries")
            else:
                alpha["needs_fix"] = False

    return state


async def _evaluate_alpha_async(alpha: dict, eval_config: dict) -> dict:
    """异步评估单个alpha"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_evaluate_alpha,
        alpha,
        eval_config
    )


def _sync_evaluate_alpha(alpha: dict, eval_config: dict) -> dict:
    """同步评估单个alpha"""
    try:
        result = call_evaluator.invoke({
            "alpha_description": alpha.get("description", ""),
            "alpha_code": alpha.get("code", ""),
            "parameters": alpha.get("parameters", {}),
            "eval_config": eval_config,
        })
        return {
            "alpha_id": alpha.get("id"),
            "result": result,
        }
    except Exception as e:
        return {
            "alpha_id": alpha.get("id"),
            "result": {"status": "error", "error_message": str(e)},
        }


async def evaluator_node_parallel(state: MiningState) -> MiningState:
    """
    Evaluator节点（并行版本）。
    并行调用评估接口获取alpha的量化指标。
    """
    logger.info(f"Evaluator node (parallel) - Iteration {state['iteration']}, evaluating {len(state['current_proposals'])} alphas")

    if not state["current_proposals"]:
        return state

    eval_config = state["config"].get("eval_config", {})

    # 并行评估所有alpha
    tasks = [_evaluate_alpha_async(alpha, eval_config) for alpha in state["current_proposals"]]
    results = await asyncio.gather(*tasks)

    # 处理结果
    max_retries = state["config"].get("max_proposer_retries", 3)

    for alpha, result in zip(state["current_proposals"], results):
        alpha_id = alpha.get("id")
        if not alpha_id or not result:
            continue

        result = result.get("result", {})
        status = result.get("status")
        alpha["compliance_status"] = "passed" if status == "success" else "failed"
        alpha["evaluator_status"] = status

        if status == "success":
            metrics = result.get("metrics", {})
            store_evaluation(alpha_id, metrics)
            alpha["evaluation"] = metrics

            if alpha_id not in state.get("discovered_factors", []):
                state.setdefault("discovered_factors", []).append(alpha_id)
                add_discovered_alpha(alpha_id)
        else:
            alpha["evaluation"] = None
            alpha["error_message"] = result.get("error_message")
            alpha["error_code"] = result.get("error_code")

            if result.get("error_code") == "COMPLIANCE_ERROR":
                alpha["needs_fix"] = True
                alpha["compliance_retry_count"] = alpha.get("compliance_retry_count", 0) + 1
                if alpha["compliance_retry_count"] >= max_retries:
                    alpha["evaluator_status"] = "skipped"
            else:
                alpha["needs_fix"] = False

    return state


async def critic_node_parallel(state: MiningState) -> MiningState:
    """
    Critic节点（并行版本）。
    并行调用Critic Agent对每个alpha进行评估。
    """
    logger.info(f"Critic node (parallel) - Iteration {state['iteration']}, critiquing {len(state['current_proposals'])} alphas")

    if not state["current_proposals"]:
        return state

    baseline = state.get("baseline_factor_library", [])
    optimization_direction = state.get("optimization_direction", "")
    config = state["config"]

    # 并行调用Critic
    tasks = []
    for alpha in state["current_proposals"]:
        evaluation = alpha.get("evaluation", {})
        if evaluation:
            tasks.append(_call_critic_agent(alpha, evaluation, baseline, optimization_direction, config))

    if not tasks:
        return state

    feedbacks = await asyncio.gather(*tasks, return_exceptions=True)

    # 保存反馈
    for alpha, feedback in zip(state["current_proposals"], feedbacks):
        if isinstance(feedback, Exception):
            logger.error(f"Critic error for {alpha.get('id')}: {feedback}")
            continue

        alpha_id = alpha.get("id")
        if not alpha_id:
            continue

        # 保存到存储
        store_feedback(
            alpha_id=feedback.get("alpha_id", alpha_id),
            iteration=feedback.get("iteration", state["iteration"]),
            ratings=feedback.get("ratings", {}),
            factual_observations=feedback.get("factual_observations", []),
            concerns=feedback.get("concerns", []),
            actionable_suggestions=feedback.get("actionable_suggestions", []),
            can_proceed=feedback.get("can_proceed", False),
            expected_match_score=feedback.get("expected_match_score", 0.5),
            expected_match_reason=feedback.get("expected_match_reason", ""),
        )

        alpha["feedback"] = feedback

    return state


def critic_node(state: MiningState) -> MiningState:
    """Critic节点（同步版本，用于兼容）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在事件循环中，使用并行版本的简化实现
            return _critic_node_sync(state)
        else:
            return asyncio.run(critic_node_parallel(state))
    except RuntimeError:
        return _critic_node_sync(state)


def _critic_node_sync(state: MiningState) -> MiningState:
    """Critic节点同步实现"""
    logger.info(f"Critic node (sync) - Iteration {state['iteration']}")

    baseline = state.get("baseline_factor_library", [])
    optimization_direction = state.get("optimization_direction", "")

    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id")
        if not alpha_id:
            continue

        evaluation = alpha.get("evaluation", {})
        if not evaluation:
            alpha["feedback"] = {
                "alpha_id": alpha_id,
                "iteration": state["iteration"],
                "ratings": {},
                "factual_observations": ["No evaluation data"],
                "concerns": [],
                "actionable_suggestions": [],
                "can_proceed": False,
            }
            continue

        # 调用Critic Agent
        try:
            feedback = _sync_call_critic(alpha, evaluation, baseline, optimization_direction, state["config"])
        except Exception as e:
            logger.error(f"Critic error: {e}")
            feedback = {
                "alpha_id": alpha_id,
                "iteration": state["iteration"],
                "ratings": {},
                "factual_observations": [f"Critic error: {str(e)}"],
                "concerns": [],
                "actionable_suggestions": [],
                "can_proceed": False,
            }

        # 保存反馈
        store_feedback(
            alpha_id=feedback.get("alpha_id", alpha_id),
            iteration=feedback.get("iteration", state["iteration"]),
            ratings=feedback.get("ratings", {}),
            factual_observations=feedback.get("factual_observations", []),
            concerns=feedback.get("concerns", []),
            actionable_suggestions=feedback.get("actionable_suggestions", []),
            can_proceed=feedback.get("can_proceed", False),
            expected_match_score=feedback.get("expected_match_score", 0.5),
            expected_match_reason=feedback.get("expected_match_reason", ""),
        )

        alpha["feedback"] = feedback

    return state


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

    for alpha in needs_fix:
        error_msg = alpha.get("error_message", "Unknown compliance error")
        error_code = alpha.get("error_code", "COMPLIANCE_ERROR")
        parent_id = alpha.get("parent_id")

        # 调用Proposer修复
        config = state["config"]
        proposer_prompt = f"""## Compliance Error - Fix Required

The following alpha has a compliance error:

**Original Alpha**: {alpha.get('name', '')}
**Error Code**: {error_code}
**Error Message**: {error_msg}

**Parent Factor ID**: {parent_id or 'N/A'}

Please generate a FIXED version of this alpha that resolves the compliance issue.
The factor should maintain the same economic intuition but use compliant code.

Generate the fixed alpha in JSON format with the same fields as before.
"""

        try:
            alphas = asyncio.run(_call_proposer_agent(proposer_prompt, config, 1))
        except RuntimeError:
            alphas = asyncio.get_event_loop().run_until_complete(
                _call_proposer_agent(proposer_prompt, config, 1)
            )

        if alphas:
            fixed = alphas[0]
            # 创建修复后的alpha
            alpha_obj = create_alpha(
                name=fixed.get("name", alpha.get("name", "") + "_fixed"),
                code=fixed.get("code", alpha.get("code", "")),
                description=fixed.get("description", alpha.get("description", "")),
                parameters=fixed.get("parameters", alpha.get("parameters", {})),
                iteration=state["iteration"],
                parent_id=parent_id,
                intuition=fixed.get("intuition", alpha.get("intuition", "")),
                optimization_rationale=fixed.get("optimization_rationale", "Compliance fix"),
            )
            # 更新state中的alpha
            idx = state["current_proposals"].index(alpha)
            fixed["id"] = alpha_obj.id
            fixed["parent_id"] = parent_id
            fixed["is_compliance_fix"] = True
            fixed["original_alpha_id"] = alpha.get("id")
            fixed["evaluator_status"] = "pending"
            state["current_proposals"][idx] = fixed
            state["pending_evaluations"].append(alpha_obj.id)
            logger.info(f"Fixed alpha {alpha.get('id')} -> {alpha_obj.id}")
        else:
            # 修复失败，标记跳过
            idx = state["current_proposals"].index(alpha)
            alpha["evaluator_status"] = "skipped"
            alpha["evaluator_status_reason"] = "Fix failed"
            logger.warning(f"Failed to fix alpha {alpha.get('id')}")

    return state


def _should_retry_compliance(state: MiningState) -> str:
    """检查是否需要重试合规检查"""
    needs_fix = [a for a in state["current_proposals"] if a.get("needs_fix")]
    if needs_fix:
        return "compliance_fix"
    return "critic"


def _should_continue(state: MiningState) -> str:
    """决定是否继续迭代"""
    if not state.get("should_continue", True):
        return "finalize"
    return "proposer"


def finalize_node(state: MiningState) -> MiningState:
    """最终化节点"""
    state["is_complete"] = True

    # 收集最终候选
    baseline_ids = [f.get("id", "") for f in state.get("baseline_factor_library", [])]
    discovered_ids = state.get("discovered_factors", [])
    all_ids = list(dict.fromkeys(baseline_ids + discovered_ids))

    state["final_candidates"] = state.get("final_candidates") or all_ids
    return state


def build_mining_workflow(
    config: AlphaMiningConfig | None = None,
    use_parallel: bool = True,
) -> Any:
    """
    构建挖掘工作流。

    Args:
        config: 配置对象
        use_parallel: 是否使用并行处理（默认True）

    Returns:
        编译后的工作流图
    """
    if config is None:
        config = AlphaMiningConfig()

    workflow = StateGraph(MiningState)

    # 添加节点
    workflow.add_node("leader", leader_node)
    workflow.add_node("proposer", proposer_node)
    workflow.add_node("evaluator", evaluator_node_parallel if use_parallel else evaluator_node)
    workflow.add_node("compliance_fix", compliance_fix_node)
    workflow.add_node("critic", critic_node_parallel if use_parallel else critic_node)
    workflow.add_node("finalize", finalize_node)

    # 定义边
    workflow.set_entry_point("leader")

    # Leader决定是否继续
    workflow.add_conditional_edges(
        "leader",
        _should_continue,
        {
            "proposer": "proposer",
            "finalize": "finalize",
        }
    )

    # 顺序执行
    workflow.add_edge("proposer", "evaluator")
    workflow.add_edge("evaluator", "compliance_fix")

    # 合规修复循环
    workflow.add_conditional_edges(
        "compliance_fix",
        _should_retry_compliance,
        {
            "compliance_fix": "evaluator",
            "critic": "critic",
        }
    )

    workflow.add_edge("critic", "leader")
    workflow.add_edge("finalize", END)

    return workflow.compile()


async def run_mining(
    config: AlphaMiningConfig | None = None,
    baseline_factor_library: list[dict] | None = None,
    use_parallel: bool = True,
) -> dict:
    """
    运行完整的挖掘流程。

    Args:
        config: 配置对象
        baseline_factor_library: 基线因子库
        use_parallel: 是否使用并行处理

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
    graph = build_mining_workflow(config, use_parallel=use_parallel)

    # 构建API配置
    api_config = {
        "api_base": config.model.api_base,
        "api_key": config.model.api_key,
        "leader_model": config.model.leader_model,
        "proposer_model": config.model.proposer_model,
        "critic_model": config.model.critic_model,
    }

    # 初始化状态
    initial_state: MiningState = {
        "session_id": session_id,
        "iteration": 1,
        "config": {
            **api_config,
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
        "selected_factor_id": None,
        "reasoning_for_selection": None,
        "suggestions_to_proposer": [],
        "factors_to_remove": [],
        "discovered_factors": [],
        "is_complete": False,
        "final_candidates": [],
    }

    # 运行工作流
    final_state = await graph.ainvoke(initial_state)

    # 保存最终状态
    if final_state.get("final_candidates"):
        finalize_mining_session(session_id, final_state["final_candidates"])

    # 收集所有因子
    all_factors = []

    for f in baseline:
        all_factors.append({
            "id": f.get("id", ""),
            "name": f.get("name", ""),
            "code": f.get("code", ""),
            "description": f.get("description", ""),
            "is_baseline": True,
            "evaluation": f.get("evaluation", {}),
        })

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
        "iterations": final_state.get("iteration", 0),
        "baseline_count": len(baseline),
        "discovered_count": len(final_state.get("discovered_factors", [])),
        "factors": all_factors,
        "final_candidates": final_state.get("final_candidates", []),
        "termination_reason": final_state.get("leader_decision", {}).get("termination_reason"),
    }
