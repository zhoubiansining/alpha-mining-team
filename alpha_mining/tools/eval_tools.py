"""Evaluator interface tools.

This module provides the interface to the backtesting framework.
The actual evaluation logic is implemented by the backtesting team.
"""

from typing import Any

from langchain_core.tools import tool

from alpha_mining.schemas.evaluation import AlphaEvaluation


@tool
def call_evaluator(
    alpha_description: str,
    alpha_code: str,
    parameters: dict,
    eval_config: dict,
) -> dict:
    """
    调用Evaluator获取alpha的量化评估结果。

    这是Evaluator的外部接口，具体回测逻辑由回测团队实现。
    当前为mock实现，返回模拟数据用于框架测试。

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
    # TODO: 具体实现由回测团队负责
    # 当前返回mock数据用于框架测试
    import random
    import uuid

    # 生成一个mock的alpha_id用于返回
    mock_alpha_id = str(uuid.uuid4())

    # 生成随机的但合理的评估指标
    ic_mean = random.uniform(0.02, 0.08)
    ic_std = random.uniform(0.05, 0.15)
    sharpe = random.uniform(0.5, 2.0)
    max_drawdown = random.uniform(0.05, 0.25)
    turnover = random.uniform(0.2, 0.8)

    evaluation = AlphaEvaluation.create_mock(
        alpha_id=mock_alpha_id,
        ic_mean=ic_mean,
        ic_std=ic_std,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        turnover=turnover,
    )

    return {
        "status": "success",
        "alpha_id": mock_alpha_id,
        "metrics": {
            "ic_mean": evaluation.ic_mean,
            "ic_std": evaluation.ic_std,
            "ir": evaluation.ir,
            "sharpe": evaluation.sharpe,
            "max_drawdown": evaluation.max_drawdown,
            "turnover": evaluation.turnover,
            "long_short_return": evaluation.long_short_return,
            "win_rate": evaluation.win_rate,
        },
        "error_message": None,
    }


def create_evaluator_tool(mock: bool = True) -> Any:
    """
    创建Evaluator工具。

    Args:
        mock: 是否使用mock实现。True时使用模拟数据，False时调用真实回测服务。

    Returns:
        评估工具函数
    """
    if mock:
        return call_evaluator
    else:
        # TODO: 实现真实回测服务调用
        raise NotImplementedError("Real evaluator not yet implemented")
