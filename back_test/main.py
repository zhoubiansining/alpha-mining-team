import os

import uvicorn
from fastapi import FastAPI

from back_test.schemas import EvaluateRequest, EvaluateResponse
from back_test.engine import execute_and_evaluate

app = FastAPI(title="Alpha Evaluator Engine")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "back-test"}

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(request: EvaluateRequest):
    status, metrics, error_msg = await execute_and_evaluate(request)
    return EvaluateResponse(
        status=status,
        alpha_id=request.alpha_id or "unknown-id",
        metrics=metrics,
        error_message=error_msg
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BACK_TEST_PORT", "8000")))
