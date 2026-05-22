from pydantic import BaseModel
from typing import Dict, Any, Optional

class EvaluateRequest(BaseModel):
    alpha_id: Optional[str] = "test-alpha-id"
    alpha_description: Optional[str] = ""
    alpha_code: str
    parameters: Dict[str, Any] = {}
    eval_config: Dict[str, Any] = {}

class Metrics(BaseModel):
    ic_mean: float
    ic_std: float
    ir: float
    sharpe: float
    max_drawdown: float
    turnover: float
    long_short_return: float
    win_rate: float

class EvaluateResponse(BaseModel):
    status: str
    alpha_id: str
    metrics: Optional[Metrics] = None
    error_message: Optional[str] = None