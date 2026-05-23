"""Evaluator interface tools.

This module provides the HTTP interface to the backtesting framework.
"""

import os
from typing import Any

import httpx

from langchain_core.tools import tool

DEFAULT_EVALUATOR_ENDPOINT = "http://localhost:8000/evaluate"
DEFAULT_EVALUATOR_TIMEOUT = 300.0


@tool
def call_evaluator(
    alpha_description: str,
    alpha_code: str,
    parameters: dict,
    eval_config: dict,
) -> dict:
    """
    调用Evaluator获取alpha的量化评估结果。

    这是Evaluator的外部接口，默认POST到back_test服务的/evaluate端点。

    Args:
        alpha_description: alpha的设计理念和数学描述
        alpha_code: 完整的Python/numpy计算表达式
        parameters: 参数配置，如 {"window": 20, "threshold": 0.5}
        eval_config: 评估配置，包含:
            - symbols: 标的列表
            - start_date: 开始日期
            - end_date: 结束日期
            - rebalance_freq: 调仓频率

    Returns:
        包含评估结果的字典，status为"success"或"error"
    """
    eval_config = eval_config or {}
    endpoint = eval_config.get("evaluator_endpoint") or os.getenv(
        "EVALUATOR_ENDPOINT",
        DEFAULT_EVALUATOR_ENDPOINT,
    )
    timeout = float(eval_config.get("evaluator_timeout") or os.getenv(
        "EVALUATOR_TIMEOUT",
        DEFAULT_EVALUATOR_TIMEOUT,
    ))

    payload = {
        "alpha_id": eval_config.get("alpha_id", "test-alpha-id"),
        "alpha_description": alpha_description,
        "alpha_code": alpha_code,
        "parameters": parameters or {},
        "eval_config": {
            key: value
            for key, value in (eval_config or {}).items()
            if key not in {"evaluator_endpoint", "evaluator_timeout", "alpha_id"}
        },
    }

    try:
        response = httpx.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as exc:
        return {
            "status": "error",
            "alpha_id": payload["alpha_id"],
            "metrics": None,
            "error_code": "EVALUATOR_HTTP_ERROR",
            "error_message": str(exc),
            "recoverable": False,
        }
    except ValueError as exc:
        return {
            "status": "error",
            "alpha_id": payload["alpha_id"],
            "metrics": None,
            "error_code": "EVALUATOR_RESPONSE_ERROR",
            "error_message": f"Invalid evaluator JSON response: {exc}",
            "recoverable": False,
        }

    if result.get("status") == "error" and not result.get("error_code"):
        result["error_code"] = _infer_error_code(result.get("error_message", ""))

    return result


def create_evaluator_tool(mock: bool = True) -> Any:
    """
    创建Evaluator工具。

    Args:
        mock: 是否使用mock实现。True时使用模拟数据，False时调用真实回测服务。

    Returns:
        评估工具函数
    """
    return call_evaluator


def _infer_error_code(error_message: str) -> str:
    """Infer a workflow-friendly error code from legacy evaluator errors."""
    message = error_message.lower()
    compliance_keywords = [
        "syntax",
        "class",
        "compute",
        "future",
        "look-ahead",
        "lookahead",
        "unauthorized",
        "invalid",
    ]
    if any(keyword in message for keyword in compliance_keywords):
        return "COMPLIANCE_ERROR"
    return "EVAL_ERROR"
