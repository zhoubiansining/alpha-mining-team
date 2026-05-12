"""Critic agent implementation."""

import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from deepagents import SubAgent

from alpha_mining.prompts.critic_prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_PROMPT,
)
from alpha_mining.tools.storage_tools import (
    save_critic_feedback,
    get_factor_library,
)


def get_default_llm():
    """获取默认的LLM实例，配置OpenAI兼容API"""
    return ChatOpenAI(
        model=os.getenv("CRITIC_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0.0,
    )


def build_critic_agent(
    llm: ChatOpenAI | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> SubAgent:
    """
    构建Critic智能体。

    Args:
        llm: 预配置的LLM实例
        model_name: 模型名称
        api_base: API基础URL
        api_key: API密钥

    Returns:
        SubAgent配置
    """
    if llm is None:
        llm = get_default_llm()
        if model_name:
            llm.model_name = model_name
        if api_base:
            llm.base_url = api_base
        if api_key:
            llm.api_key = api_key

    return {
        "name": "critic",
        "description": "Criticize alpha candidates based on backtest results",
        "system_prompt": CRITIC_SYSTEM_PROMPT,
        "tools": [save_critic_feedback, get_factor_library],
        "model": llm,
    }


def parse_critic_feedback(response: str, alpha_id: str, iteration: int) -> dict:
    """
    解析Critic的反馈响应。

    Args:
        response: LLM的原始响应
        alpha_id: 被评价的alpha ID
        iteration: 当前迭代轮次

    Returns:
        解析后的反馈字典
    """
    if not response or not response.strip():
        return _default_feedback(alpha_id, iteration)

    response = response.strip()

    # 尝试直接解析JSON
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return _build_feedback_from_dict(result, alpha_id, iteration)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_matches = re.findall(code_block_pattern, response)
    for match in code_matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict):
                return _build_feedback_from_dict(result, alpha_id, iteration)
        except json.JSONDecodeError:
            continue

    # 尝试提取JSON对象
    # 使用贪婪匹配找到第一个 { ... } 对象
    json_pattern = r'\{[\s\S]*\}'
    json_matches = re.findall(json_pattern, response)
    for match in json_matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict) and len(result) > 0:
                return _build_feedback_from_dict(result, alpha_id, iteration)
        except json.JSONDecodeError:
            continue

    # 返回默认值
    return _default_feedback(alpha_id, iteration)


def _build_feedback_from_dict(data: dict, alpha_id: str, iteration: int) -> dict:
    """从字典构建反馈对象"""
    return {
        "alpha_id": alpha_id,
        "iteration": iteration,
        "ratings": data.get("ratings", {
            "theoretical_soundness": 3,
            "backtest_quality": 3,
            "robustness": 3,
            "implementation_quality": 3,
            "diversification": 3,
        }),
        "factual_observations": data.get("factual_observations", []),
        "concerns": data.get("concerns", []),
        "actionable_suggestions": data.get("actionable_suggestions", []),
        "can_proceed": data.get("can_proceed", False),
    }


def _default_feedback(alpha_id: str, iteration: int) -> dict:
    """返回默认反馈"""
    return {
        "alpha_id": alpha_id,
        "iteration": iteration,
        "ratings": {
            "theoretical_soundness": 3,
            "backtest_quality": 3,
            "robustness": 3,
            "implementation_quality": 3,
            "diversification": 3,
        },
        "factual_observations": ["Failed to parse critic feedback"],
        "concerns": ["Parse error"],
        "actionable_suggestions": [],
        "can_proceed": False,
    }
