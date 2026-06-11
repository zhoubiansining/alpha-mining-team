"""Main workflow for alpha factor mining using LangGraph with real Agent calls."""

import asyncio
import json
import logging
import os
from typing import TypedDict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
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
    delete_factor_record,
)
from alpha_mining.schemas.history import IterationHistory
from alpha_mining.schemas.alpha import AlphaExpression
from alpha_mining.schemas.evaluation import AlphaEvaluation
from alpha_mining.agents.leader import build_leader_agent, parse_leader_decision
from alpha_mining.agents.proposer import build_proposer_agent, parse_alpha_proposals
from alpha_mining.agents.critic import build_critic_agent, parse_critic_feedback
from alpha_mining.agents.curator import build_curator_agent, parse_curator_decision
from alpha_mining.prompts.leader_prompts import LEADER_ITERATION_PROMPT
from alpha_mining.prompts.proposer_prompts import PROPOSER_USER_PROMPT
from alpha_mining.prompts.critic_prompts import CRITIC_USER_PROMPT
from alpha_mining.prompts.curator_prompts import CURATOR_USER_PROMPT


logger = logging.getLogger(__name__)


def _log_debug_payload(title: str, content: str, limit: int = 1200) -> None:
    """Log a compact snippet of prompt/response content when enabled."""
    if os.getenv("ALPHA_MINING_DEBUG_PROMPTS", "0") not in {"1", "true", "True"}:
        return
    snippet = content if len(content) <= limit else content[:limit] + "...<truncated>"
    logger.info("%s\n%s", title, snippet)

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

    # Curator决策
    curator_decision: Optional[dict]

    # 已发现的因子（探索过程中产生的）
    discovered_factors: list[str]

    # 每轮迭代的轨迹记录（包含所有合法提案及其最终命运）
    all_iterations: list[dict]

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


async def _evaluate_baseline_factor_library(
    baseline: list[dict],
    eval_config: dict,
) -> list[dict]:
    """Pre-evaluate baseline factors before the first Leader decision."""
    if not baseline:
        return []

    logger.info("Pre-evaluating %d baseline factors", len(baseline))
    evaluated_baseline: list[dict] = []

    async def _evaluate_one(index: int, factor: dict) -> dict:
        if factor.get("evaluation"):
            logger.info("Baseline factor already has evaluation | id=%s | name=%s", factor.get("id"), factor.get("name"))
            return factor

        factor_id = factor.get("id", f"baseline-{index}")
        logger.info("Pre-evaluating baseline factor | id=%s | name=%s", factor_id, factor.get("name", ""))
        result = await _evaluate_alpha_async(
            {
                "id": factor_id,
                "name": factor.get("name", ""),
                "description": factor.get("description", ""),
                "code": factor.get("code", ""),
                "parameters": factor.get("parameters", {}),
            },
            {
                **eval_config,
                "alpha_id": factor_id,
            },
        )
        payload = result.get("result", {})
        if payload.get("status") == "success":
            factor = {
                **factor,
                "evaluation": payload.get("metrics", {}),
            }
            logger.info(
                "Baseline evaluation success | id=%s | ic_mean=%.4f | sharpe=%.4f",
                factor_id,
                factor["evaluation"].get("ic_mean", 0.0),
                factor["evaluation"].get("sharpe", 0.0),
            )
        else:
            factor = {
                **factor,
                "evaluation": factor.get("evaluation", {}),
                "baseline_evaluation_error": payload.get("error_message"),
                "baseline_evaluation_error_code": payload.get("error_code"),
            }
            logger.warning(
                "Baseline evaluation failed | id=%s | error_code=%s | message=%s",
                factor_id,
                payload.get("error_code"),
                payload.get("error_message"),
            )
        return factor

    tasks = [_evaluate_one(index, factor) for index, factor in enumerate(baseline)]
    results = await asyncio.gather(*tasks)
    evaluated_baseline.extend(results)
    return evaluated_baseline


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
    discovered_ids = state.get("discovered_factors", [])
    discovered = [f for f in list_factors() if f.id in discovered_ids]

    # 已发现因子以存储层为准；不要依赖轮次过滤，否则本轮刚评估成功的因子在下一次Leader决策中会被漏掉。
    discovered_count = len(discovered)

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

    curator_decision = state.get("curator_decision") or {}
    return {
        "discovered_count": discovered_count,
        "recent_feedbacks": recent_feedbacks,
        "baseline_metrics": baseline_metrics,
        "discovered_metrics": discovered_metrics,
        "curator_summary": curator_decision.get("library_summary", "No curation yet."),
    }


def _find_factor_reference(reference: str | None, baseline: list[dict]) -> dict | None:
    """Find a factor by id first, then by name, across baseline and discovered factors."""
    if not reference:
        return None

    normalized_reference = reference.strip().lower()

    for factor in baseline:
        if factor.get("id") == reference or factor.get("name", "").strip().lower() == normalized_reference:
            return factor

    for factor in list_factors():
        if factor.id == reference or factor.name.strip().lower() == normalized_reference:
            eval_result = get_evaluation_by_alpha_id(factor.id)
            return {
                "id": factor.id,
                "name": factor.name,
                "description": factor.description,
                "code": factor.code,
                "evaluation": eval_result.to_summary() if eval_result else {},
            }

    return None


def _resolve_candidate_ids(references: list[str], baseline: list[dict]) -> list[str]:
    """Resolve a list of factor name/id references to actual UUIDs.

    Leader may output factor names or IDs as final_candidates.
    This normalizes them to UUIDs so downstream code can match them against
    factor.id fields.
    """
    resolved = []
    for ref in references:
        factor = _find_factor_reference(ref, baseline)
        if factor:
            resolved.append(factor.get("id") if isinstance(factor, dict) else factor.id)
        else:
            resolved.append(ref)  # Keep as-is if unresolvable (fallback)
    return list(dict.fromkeys(resolved))  # deduplicate, preserve order


def _sync_discovered_state(state: MiningState) -> None:
    """Synchronize state-level discovered IDs from storage."""
    stored_ids = [factor.id for factor in list_factors()]
    merged = list(dict.fromkeys([*state.get("discovered_factors", []), *stored_ids]))
    state["discovered_factors"] = merged


def _advance_iteration(state: MiningState) -> MiningState:
    """Advance to the next Leader planning round after Critic finishes."""
    _sync_discovered_state(state)
    state["iteration"] += 1
    logger.info(
        "Iteration advanced | next_iteration=%d | discovered=%d",
        state["iteration"],
        len(state.get("discovered_factors", [])),
    )
    return state


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


def _extract_response_text(response: Any) -> str:
    """Extract assistant text from LangGraph/LangChain response shapes."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    if isinstance(response, dict):
        messages = response.get("messages")
        if messages:
            return _extract_response_text(messages[-1])
        for key in ("output", "content", "text"):
            if key in response:
                return _extract_response_text(response[key])
        return json.dumps(response, ensure_ascii=False)
    return str(response)


def _invoke_deep_agent_text(agent: Any, prompt: str) -> str:
    """Invoke a compiled deepagents graph and return final assistant text."""
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return _extract_response_text(response)


def _invoke_subagent_config_text(agent_config: dict, prompt: str) -> str:
    """Invoke a SubAgent config's underlying chat model directly."""
    model = agent_config["model"]
    messages = [
        SystemMessage(content=agent_config.get("system_prompt", "")),
        HumanMessage(content=prompt),
    ]
    response = model.invoke(messages)
    return _extract_response_text(response)


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
        logger.info(
            "Calling Leader agent | model=%s | prompt_chars=%d",
            config.get("leader_model", "gpt-4o"),
            len(leader_prompt),
        )
        _log_debug_payload("[Leader Prompt]", leader_prompt)
        agent = build_leader_agent(
            model_name=config.get("leader_model", "gpt-4o"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )
        response_text = _invoke_deep_agent_text(agent, leader_prompt)
        logger.info("Leader agent returned %d chars", len(response_text))
        _log_debug_payload("[Leader Response]", response_text)
        return parse_leader_decision(response_text)
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
        logger.info(
            "Calling Proposer agent | model=%s | target_proposals=%d | prompt_chars=%d",
            config.get("proposer_model", "gpt-4o-mini"),
            n_proposals,
            len(proposer_prompt),
        )
        _log_debug_payload("[Proposer Prompt]", proposer_prompt)
        agent = build_proposer_agent(
            model_name=config.get("proposer_model", "gpt-4o-mini"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )
        response_text = _invoke_subagent_config_text(agent, proposer_prompt)
        logger.info("Proposer agent returned %d chars", len(response_text))
        _log_debug_payload("[Proposer Response]", response_text)
        alphas = parse_alpha_proposals(response_text)
        logger.info("Proposer parsed %d alpha candidates", len(alphas))
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
        logger.info(
            "Calling Critic agent | model=%s | alpha_id=%s | prompt_chars=%d",
            config.get("critic_model", "gpt-4o"),
            alpha.get("id", ""),
            len(optimization_direction or "") + len(json.dumps(evaluation, ensure_ascii=False)) + len(alpha.get("name", "")),
        )
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

        _log_debug_payload(f"[Critic Prompt | alpha_id={alpha.get('id', '')}]", user_prompt)

        response = _invoke_subagent_config_text(agent, user_prompt)
        logger.info("Critic agent returned %d chars for alpha_id=%s", len(response), alpha.get("id", ""))
        _log_debug_payload(f"[Critic Response | alpha_id={alpha.get('id', '')}]", response)
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


def _sync_call_curator(curator_prompt: str, config: dict) -> dict:
    """同步调用Curator Agent"""
    try:
        logger.info(
            "Calling Curator agent | model=%s | prompt_chars=%d",
            config.get("critic_model", "gpt-4o"),
            len(curator_prompt),
        )
        _log_debug_payload("[Curator Prompt]", curator_prompt)
        agent = build_curator_agent(
            model_name=config.get("critic_model", "gpt-4o"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )
        response = _invoke_subagent_config_text(agent, curator_prompt)
        logger.info("Curator agent returned %d chars", len(response))
        _log_debug_payload("[Curator Response]", response)
        return parse_curator_decision(response)
    except Exception as e:
        logger.error(f"Curator agent error: {e}")
        return {
            "admitted_factors": [],
            "admission_reasons": {},
            "rejected_factors": [],
            "rejection_reasons": {},
            "factors_to_remove": [],
            "removal_reasons": {},
            "library_summary": f"Curator error: {e}",
            "quality_assessment": "Error",
        }


async def _call_curator_agent(curator_prompt: str, config: dict) -> dict:
    """异步调用Curator Agent"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_call_curator,
        curator_prompt,
        config,
    )


def leader_node(state: MiningState) -> MiningState:
    """
    Leader决策节点。
    使用真实的Leader Agent进行决策。
    """
    logger.info(
        "Leader node start | session=%s | iteration=%d | baseline=%d | discovered=%d | proposals=%d",
        state.get("session_id", ""),
        state["iteration"],
        len(state.get("baseline_factor_library", [])),
        len(state.get("discovered_factors", [])),
        len(state.get("current_proposals", [])),
    )

    iteration = state["iteration"]
    config = state["config"]
    baseline = state.get("baseline_factor_library", [])
    _sync_discovered_state(state)

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
        curator_summary=context.get("curator_summary", "No curation yet."),
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
    selected_reference = decision.get("selected_factor_id")
    selected_factor = _find_factor_reference(selected_reference, baseline)
    if selected_reference and selected_factor:
        state["selected_factor_id"] = selected_factor.get("id")
        if selected_reference != state["selected_factor_id"]:
            logger.info(
                "Normalized Leader selected factor reference | raw=%s | resolved_id=%s | name=%s",
                selected_reference,
                state["selected_factor_id"],
                selected_factor.get("name"),
            )
    else:
        state["selected_factor_id"] = selected_reference
        if selected_reference:
            logger.warning("Leader selected factor could not be resolved | raw=%s", selected_reference)
    state["reasoning_for_selection"] = decision.get("reasoning_for_selection", "")
    state["suggestions_to_proposer"] = decision.get("suggestions_to_proposer", [])

    # 如果是终止决策
    if not state["should_continue"]:
        all_candidates = state.get("discovered_factors", []) + [f.get("id", "") for f in baseline]
        raw_candidates = decision.get("final_candidates") or list(dict.fromkeys(all_candidates))
        # Resolve names to UUIDs — Leader may output factor names instead of IDs
        resolved = _resolve_candidate_ids(raw_candidates, baseline)
        logger.info(
            "Resolved final_candidates | raw=%s | resolved=%s",
            raw_candidates, resolved,
        )
        state["leader_decision"]["final_candidates"] = resolved
        state["final_candidates"] = resolved

    logger.info(
        "Leader decision | continue=%s | selected_factor=%s | final_candidates=%s",
        state["should_continue"],
        state.get("selected_factor_id"),
        state.get("final_candidates") if not state["should_continue"] else [],
    )
    return state


def proposer_node(state: MiningState) -> MiningState:
    """
    Proposer生成节点。
    使用真实的Proposer Agent生成alpha候选。
    """
    logger.info(
        "Proposer node start | session=%s | iteration=%d | selected_factor=%s | suggestions=%d",
        state.get("session_id", ""),
        state["iteration"],
        state.get("selected_factor_id"),
        len(state.get("suggestions_to_proposer", [])),
    )

    config = state["config"]
    baseline = state.get("baseline_factor_library", [])
    iteration = state["iteration"]

    # 获取要优化的基线因子
    selected_factor = None
    if state.get("selected_factor_id"):
        selected_factor = _find_factor_reference(state.get("selected_factor_id"), baseline)

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

    logger.info(
        "Proposer generated %d alphas | ids=%s",
        len(state["current_proposals"]),
        [alpha.get("id") for alpha in state["current_proposals"]],
    )
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
    logger.info(
        "Evaluator node start | session=%s | iteration=%d | proposals=%d",
        state.get("session_id", ""),
        state["iteration"],
        len(state["current_proposals"]),
    )

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
            "eval_config": {
                **state["config"].get("eval_config", {}),
                "alpha_id": alpha_id,
            },
        })

        status = result.get("status")
        alpha["compliance_status"] = "passed" if status == "success" else "failed"
        alpha["evaluator_status"] = status

        if status == "success":
            metrics = result.get("metrics", {})
            store_evaluation(alpha_id, metrics)
            alpha["evaluation"] = metrics
            logger.info("Evaluator success | alpha_id=%s | ic_mean=%.4f | sharpe=%.4f", alpha_id, metrics.get("ic_mean", 0.0), metrics.get("sharpe", 0.0))
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
            logger.warning(
                "Evaluator error | alpha_id=%s | error_code=%s | message=%s",
                alpha_id,
                alpha.get("error_code"),
                alpha.get("error_message"),
            )

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
            "eval_config": {
                **eval_config,
                "alpha_id": alpha.get("id"),
            },
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
    logger.info(
        "Evaluator node (parallel) start | session=%s | iteration=%d | proposals=%d",
        state.get("session_id", ""),
        state["iteration"],
        len(state["current_proposals"]),
    )

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
            logger.info("Evaluator success | alpha_id=%s | ic_mean=%.4f | sharpe=%.4f", alpha_id, metrics.get("ic_mean", 0.0), metrics.get("sharpe", 0.0))
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
            logger.warning(
                "Evaluator error | alpha_id=%s | error_code=%s | message=%s",
                alpha_id,
                alpha.get("error_code"),
                alpha.get("error_message"),
            )

    return state


async def critic_node_parallel(state: MiningState) -> MiningState:
    """
    Critic节点（并行版本）。
    并行调用Critic Agent对每个alpha进行评估。
    """
    logger.info(
        "Critic node (parallel) start | session=%s | iteration=%d | proposals=%d",
        state.get("session_id", ""),
        state["iteration"],
        len(state["current_proposals"]),
    )

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
        logger.info(
            "Critic feedback stored | alpha_id=%s | can_proceed=%s | expected_match_score=%.3f",
            alpha_id,
            feedback.get("can_proceed"),
            feedback.get("expected_match_score", 0.0),
        )

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
    logger.info(
        "Critic node (sync) start | session=%s | iteration=%d | proposals=%d",
        state.get("session_id", ""),
        state["iteration"],
        len(state["current_proposals"]),
    )

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
        logger.info(
            "Critic feedback stored | alpha_id=%s | can_proceed=%s | expected_match_score=%.3f",
            alpha_id,
            feedback.get("can_proceed"),
            feedback.get("expected_match_score", 0.0),
        )

    return state


def curator_node(state: MiningState) -> MiningState:
    """
    Curator节点 — 后Critic因子库守门人。
    评审本轮所有新因子，决定哪些进入因子库，哪些旧因子需要删除。
    替代了原来Leader的因子库管理职责。
    """
    logger.info(
        "Curator node start | session=%s | iteration=%d | proposals=%d",
        state.get("session_id", ""),
        state["iteration"],
        len(state["current_proposals"]),
    )

    if not state["current_proposals"]:
        _advance_iteration(state)
        state["curator_decision"] = {
            "admitted_factors": [],
            "rejected_factors": [],
            "factors_to_remove": [],
            "library_summary": "No proposals to curate.",
            "quality_assessment": "No new factors this round.",
        }
        return state

    config = state["config"]
    baseline = state.get("baseline_factor_library", [])
    iteration = state["iteration"]

    # 构建新候选因子摘要（含Critic反馈和回测指标）
    new_candidates_parts = []
    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id", "")
        evaluation = alpha.get("evaluation") or {}
        feedback = alpha.get("feedback") or {}

        ratings = feedback.get("ratings", {})
        avg_rating = sum(ratings.values()) / len(ratings) if ratings else 0

        new_candidates_parts.append(
            f"### {alpha.get('name', 'Unknown')} (ID: {alpha_id})\n"
            f"- Description: {alpha.get('description', '')}\n"
            f"- Optimization Rationale: {alpha.get('optimization_rationale', '')}\n"
            f"- Backtest: IC={evaluation.get('ic_mean', 0):.4f}, Sharpe={evaluation.get('sharpe', 0):.4f}, "
            f"IR={evaluation.get('ir', 0):.2f}, MaxDD={evaluation.get('max_drawdown', 0):.2%}\n"
            f"- Critic: can_proceed={feedback.get('can_proceed', False)}, "
            f"match_score={feedback.get('expected_match_score', 0):.2f}, "
            f"avg_rating={avg_rating:.1f}\n"
            f"- Critic Concerns: {'; '.join(feedback.get('concerns', [])) or 'None'}\n"
            f"- Critic Suggestions: {'; '.join(feedback.get('actionable_suggestions', [])) or 'None'}\n"
        )

    new_candidates_summary = "\n".join(new_candidates_parts) if new_candidates_parts else "No new candidates."

    # 构建现有因子库摘要
    existing_library_parts = []
    for factor in list_factors():
        eval_result = get_evaluation_by_alpha_id(factor.id)
        eval_info = ""
        if eval_result:
            eval_info = f"IC={eval_result.ic_mean:.4f}, Sharpe={eval_result.sharpe:.4f}, IR={eval_result.ir:.2f}"
        existing_library_parts.append(f"- {factor.name} (ID: {factor.id}, iter {factor.iteration}): {eval_info}")

    existing_library_summary = "\n".join(existing_library_parts) if existing_library_parts else "No existing factors."

    # 构建Curator提示词
    curator_prompt = CURATOR_USER_PROMPT.format(
        iteration=iteration,
        max_iterations=config.get("max_iterations", 20),
        new_candidates_summary=new_candidates_summary,
        existing_library_summary=existing_library_summary,
        baseline_summary=_build_baseline_summary(baseline),
    )

    # 调用Curator
    try:
        decision = asyncio.run(_call_curator_agent(curator_prompt, config))
    except RuntimeError:
        decision = asyncio.get_event_loop().run_until_complete(
            _call_curator_agent(curator_prompt, config)
        )

    state["curator_decision"] = decision

    # 执行Curator决策：准入新因子
    admitted = set(decision.get("admitted_factors", []))
    rejected = set(decision.get("rejected_factors", []))
    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id", "")
        if not alpha_id:
            continue
        if alpha_id in admitted:
            if alpha_id not in state.get("discovered_factors", []):
                state.setdefault("discovered_factors", []).append(alpha_id)
                add_discovered_alpha(alpha_id)
            logger.info(
                "Curator admitted factor | id=%s | name=%s | reason=%s",
                alpha_id,
                alpha.get("name", ""),
                decision.get("admission_reasons", {}).get(alpha_id, ""),
            )
        elif alpha_id in rejected:
            logger.info(
                "Curator rejected factor | id=%s | name=%s | reason=%s",
                alpha_id,
                alpha.get("name", ""),
                decision.get("rejection_reasons", {}).get(alpha_id, ""),
            )
        else:
            # 未被明确提到的因子：如Critic评价尚可则默认准入
            feedback = alpha.get("feedback") or {}
            if feedback.get("can_proceed") and feedback.get("expected_match_score", 0) >= 0.4:
                if alpha_id not in state.get("discovered_factors", []):
                    state.setdefault("discovered_factors", []).append(alpha_id)
                    add_discovered_alpha(alpha_id)
                logger.info(
                    "Curator default-admitted factor | id=%s | name=%s (not explicitly ruled on)",
                    alpha_id,
                    alpha.get("name", ""),
                )
            else:
                logger.info(
                    "Curator default-rejected factor | id=%s | name=%s (not explicitly ruled on, poor metrics)",
                    alpha_id,
                    alpha.get("name", ""),
                )

    # 执行Curator决策：删除旧因子
    factors_to_remove = decision.get("factors_to_remove", [])
    if factors_to_remove:
        for factor_id in factors_to_remove:
            factor = _find_factor_reference(factor_id, baseline)
            resolved_id = factor.get("id") if factor else factor_id
            delete_factor_record(resolved_id)
            if resolved_id in state["discovered_factors"]:
                state["discovered_factors"].remove(resolved_id)
            logger.info(
                "Curator removed old factor | id=%s | reason=%s",
                resolved_id,
                decision.get("removal_reasons", {}).get(factor_id, ""),
            )

    logger.info(
        "Curator decision | admitted=%d | rejected=%d | removed=%d | library_size=%d",
        len(admitted),
        len(rejected),
        len(factors_to_remove),
        len(state.get("discovered_factors", [])),
    )

    # 记录本轮迭代轨迹：包含所有合法提案及其最终命运
    # 排除不合法的因子（compliance check failed / evaluation missing）
    iteration_proposals = []
    for alpha in state["current_proposals"]:
        alpha_id = alpha.get("id", "")
        if not alpha_id:
            continue
        evaluation = alpha.get("evaluation") or {}
        # 只记录通过合规检查、有实际评估结果的因子
        if not evaluation:
            continue
        feedback = alpha.get("feedback") or {}
        # 判断该因子的最终命运
        if alpha_id in admitted:
            fate = "admitted"
        elif alpha_id in rejected:
            fate = "rejected"
        elif feedback.get("can_proceed") and feedback.get("expected_match_score", 0) >= 0.4:
            fate = "admitted"  # 默认准入
        else:
            fate = "rejected"  # 默认拒绝
        iteration_proposals.append({
            "id": alpha_id,
            "name": alpha.get("name", ""),
            "description": alpha.get("description", ""),
            "code": alpha.get("code", ""),
            "parent_id": alpha.get("parent_id", ""),
            "evaluation": {
                "ic_mean": evaluation.get("ic_mean"),
                "ic_std": evaluation.get("ic_std"),
                "ir": evaluation.get("ir"),
                "sharpe": evaluation.get("sharpe"),
                "max_drawdown": evaluation.get("max_drawdown"),
                "long_short_return": evaluation.get("long_short_return"),
                "win_rate": evaluation.get("win_rate"),
                "turnover": evaluation.get("turnover"),
            },
            "critic": {
                "ratings": feedback.get("ratings", {}),
                "expected_match_score": feedback.get("expected_match_score"),
                "can_proceed": feedback.get("can_proceed"),
                "concerns": feedback.get("concerns", []),
                "actionable_suggestions": feedback.get("actionable_suggestions", []),
            },
            "fate": fate,
        })

    iteration_record = {
        "iteration": state["iteration"],
        "proposals": iteration_proposals,
        "curator_summary": {
            "admitted": list(admitted),
            "rejected": list(rejected),
            "removed": factors_to_remove,
            "library_size_after": len(state.get("discovered_factors", [])),
        },
    }
    state.setdefault("all_iterations", []).append(iteration_record)

    return _advance_iteration(state)


def compliance_fix_node(state: MiningState) -> MiningState:
    """
    合规修复节点。
    对Evaluator判定为不合规的alpha，调用Proposer重新生成。
    """
    logger.info(
        "Compliance fix node start | session=%s | iteration=%d | needs_fix=%d",
        state.get("session_id", ""),
        state["iteration"],
        len([a for a in state["current_proposals"] if a.get("needs_fix")]),
    )

    # 查找需要修复的alpha
    needs_fix = [a for a in state["current_proposals"] if a.get("needs_fix")]
    if not needs_fix:
        return state

    logger.info("Fixing %d non-compliant alphas: %s", len(needs_fix), [a.get("id") for a in needs_fix])

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
    workflow.add_node("curator", curator_node)
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

    # Critic → Curator → Leader (factor library management now handled by Curator)
    workflow.add_edge("critic", "curator")
    workflow.add_edge("curator", "leader")
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

    # 先对基线因子进行预回测，确保Leader拥有真实初始指标
    baseline = baseline_factor_library or []
    eval_config = {
        "symbols": config.target_assets,
        "start_date": config.eval_period["start_date"],
        "end_date": config.eval_period["end_date"],
    }
    baseline = await _evaluate_baseline_factor_library(baseline, eval_config)

    # 设置基线因子库
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
                **eval_config,
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
        "curator_decision": None,
        "discovered_factors": [],
        "all_iterations": [],
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

    leader_decision = final_state.get("leader_decision") or {}

    return {
        "session_id": session_id,
        "iterations": final_state.get("iteration", 0),
        "baseline_count": len(baseline),
        "discovered_count": len(final_state.get("discovered_factors", [])),
        "factors": all_factors,
        "final_candidates": final_state.get("final_candidates", []),
        "termination_reason": leader_decision.get("termination_reason"),
        "iteration_history": final_state.get("all_iterations", []),
    }
