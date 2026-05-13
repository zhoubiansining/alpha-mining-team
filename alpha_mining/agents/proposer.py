"""Proposer agent implementation."""

import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from deepagents import SubAgent

from alpha_mining.prompts.proposer_prompts import (
    PROPOSER_SYSTEM_PROMPT,
    PROPOSER_USER_PROMPT,
)
from alpha_mining.tools.storage_tools import (
    save_alpha,
    get_factor_library,
)


def get_default_llm():
    """获取默认的LLM实例，配置OpenAI兼容API"""
    return ChatOpenAI(
        model=os.getenv("PROPOSER_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0.0,
    )


def build_proposer_agent(
    llm: ChatOpenAI | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> SubAgent:
    """
    构建Proposer智能体。

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
        "name": "proposer",
        "description": "Generate new alpha factor candidates based on Leader's directives",
        "system_prompt": PROPOSER_SYSTEM_PROMPT,
        "tools": [save_alpha, get_factor_library],
        "model": llm,
    }


def parse_alpha_proposals(response: str) -> list[dict]:
    """
    解析Proposer的alpha提案响应。

    Args:
        response: LLM的原始响应

    Returns:
        解析后的alpha列表
    """
    if not response or not response.strip():
        return []

    # 清理响应文本
    response = response.strip()

    # 尝试直接解析JSON数组
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return _validate_alpha_list(result)
    except json.JSONDecodeError:
        pass

    # 尝试解析单个JSON对象
    try:
        result = json.loads(response)
        if isinstance(result, dict) and "name" in result and "code" in result:
            return [result]
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON数组
    # 查找 [...] 模式
    array_pattern = r'\[[\s\S]*\]'
    array_matches = list(re.finditer(array_pattern, response))
    for match in reversed(array_matches):  # 从后往前找最后一个
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                validated = _validate_alpha_list(result)
                if validated:
                    return validated
        except json.JSONDecodeError:
            continue

    # 尝试提取JSON对象（带代码块或不带）
    # 先尝试提取 ```json ... ``` 块
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_matches = list(re.finditer(code_block_pattern, response))
    for match in reversed(code_matches):
        try:
            result = json.loads(match.group(1))
            if isinstance(result, dict) and "name" in result:
                return [result]
        except json.JSONDecodeError:
            continue

    # 尝试提取裸的JSON对象（支持嵌套大括号）
    # 使用贪婪匹配从 name 到 code 的完整范围
    obj_pattern = r'\{[\s\S]*?"name"\s*:\s*"?[\w\-\s]+"?[\s\S]*?"code"\s*:\s*"[^"]*"[\s\S]*?\}'
    matches = re.findall(obj_pattern, response)
    results = []
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "name" in obj and "code" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    if results:
        return results

    # 最后尝试提取任何以 { 开头、包含 name 和 code 的有效JSON对象
    try:
        first_brace = response.index("{")
        last_brace = response.rindex("}")
        candidate = response[first_brace:last_brace + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "name" in obj and "code" in obj:
            return [obj]
    except (ValueError, json.JSONDecodeError):
        pass

    return results


def _validate_alpha_list(items: list) -> list[dict]:
    """验证并过滤alpha列表"""
    results = []
    for item in items:
        if isinstance(item, dict) and "name" in item and "code" in item:
            # 确保必需字段存在
            alpha = {
                "name": item.get("name", ""),
                "code": item.get("code", ""),
                "description": item.get("description", ""),
                "parameters": item.get("parameters", {}),
                "intuition": item.get("intuition", ""),
                "optimization_rationale": item.get("optimization_rationale", ""),
                "improvement_targets": item.get("improvement_targets", []),
            }
            results.append(alpha)
    return results
