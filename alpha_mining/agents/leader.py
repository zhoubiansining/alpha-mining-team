"""Leader agent implementation."""

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent, SubAgent

from alpha_mining.prompts.leader_prompts import (
    LEADER_SYSTEM_PROMPT,
    LEADER_ITERATION_PROMPT,
)
from alpha_mining.tools.storage_tools import (
    get_factor_library,
    get_iteration_history,
)


def get_default_llm():
    """获取默认的LLM实例，配置OpenAI兼容API"""
    return ChatOpenAI(
        model=os.getenv("LEADER_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0.0,
    )


def build_leader_agent(
    llm: ChatOpenAI | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Any:
    """
    构建Leader智能体。

    Args:
        llm: 预配置的LLM实例
        model_name: 模型名称
        api_base: API基础URL
        api_key: API密钥

    Returns:
        Deep Agent实例
    """
    if llm is None:
        llm = get_default_llm()
        if model_name:
            llm.model_name = model_name
        if api_base:
            llm.base_url = api_base
        if api_key:
            llm.api_key = api_key

    # 定义子智能体
    proposer_agent: SubAgent = {
        "name": "proposer",
        "description": "Generate new alpha factor candidates based on Leader's directives",
        "system_prompt": _get_proposer_system_prompt(),
        "tools": [
            get_factor_library,
        ],
    }

    critic_agent: SubAgent = {
        "name": "critic",
        "description": "Criticize alpha candidates based on backtest results",
        "system_prompt": _get_critic_system_prompt(),
        "tools": [
            get_factor_library,
        ],
    }

    agent = create_deep_agent(
        model=llm,
        system_prompt=LEADER_SYSTEM_PROMPT,
        subagents=[proposer_agent, critic_agent],
    )

    return agent


def _get_proposer_system_prompt() -> str:
    """获取Proposer的系统提示词"""
    from alpha_mining.prompts.proposer_prompts import PROPOSER_SYSTEM_PROMPT
    return PROPOSER_SYSTEM_PROMPT


def _get_critic_system_prompt() -> str:
    """获取Critic的系统提示词"""
    from alpha_mining.prompts.critic_prompts import CRITIC_SYSTEM_PROMPT
    return CRITIC_SYSTEM_PROMPT


def parse_leader_decision(response: str) -> dict:
    """
    解析Leader的决策响应。

    Args:
        response: LLM的原始响应

    Returns:
        解析后的决策字典
    """
    try:
        # 尝试直接解析JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 返回默认值
        return {
            "should_continue": False,
            "reason": "Failed to parse leader decision",
            "optimization_direction": None,
            "focus_areas": [],
            "selected_for_context": [],
            "reasoning_for_selection": "",
            "suggestions_to_proposer": [],
            "final_candidates": None,
            "termination_reason": "Parse error",
        }
