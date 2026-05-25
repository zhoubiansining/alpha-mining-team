"""Shared helpers for baseline factor miners.

The agent pipeline already speaks the factor-library contract used by
`tests.base_factors`. Every baseline produces records in the **same shape**,
so downstream comparison is trivial:

    {
        "id", "name", "code", "description",
        "parameters", "category", "evaluation"
    }

We deliberately keep the field set minimal — anything more would just be
duplicated metadata that drifts as the agents add their own annotations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import httpx


# Header injected at the top of every generated factor class so that the
# class body can use pandas / numpy without re-importing in the spawned
# `exec()` scope inside back_test/engine.py.
FACTOR_TEMPLATE_HEADER = """import numpy as np
import pandas as pd
"""


def build_factor_record(
    *,
    factor_id: str,
    name: str,
    code: str,
    description: str,
    parameters: dict | None = None,
    category: str = "baseline",
) -> dict:
    """Wrap an AlphaFactorTemplate class source into the library record schema."""
    return {
        "id": factor_id,
        "name": name,
        "code": code if code.lstrip().startswith("import") else FACTOR_TEMPLATE_HEADER + code,
        "description": description,
        "parameters": dict(parameters or {}),
        "category": category,
        "evaluation": {},
    }


def call_evaluator_http(
    *,
    record: dict,
    eval_config: dict,
    endpoint: str | None = None,
    timeout: float = 300.0,
) -> dict:
    """POST one factor to back_test /evaluate and return its parsed metrics or error info."""
    endpoint = endpoint or os.getenv(
        "EVALUATOR_ENDPOINT", "http://127.0.0.1:18000/evaluate"
    )
    payload = {
        "alpha_id": record["id"],
        "alpha_description": record.get("description", ""),
        "alpha_code": record["code"],
        "parameters": record.get("parameters", {}),
        "eval_config": {**eval_config, "alpha_id": record["id"]},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, json=payload)
    response.raise_for_status()
    body = response.json()
    if body.get("status") == "success":
        return {"status": "success", "metrics": body.get("metrics", {})}
    return {
        "status": "error",
        "error_code": body.get("error_code", "EVAL_ERROR"),
        "error_message": body.get("error_message", "unknown evaluator error"),
    }


def dump_library(records: Iterable[dict], out_path: str | os.PathLike) -> Path:
    """Persist a list of factor records to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)
    return out_path


def load_library(path: str | os.PathLike) -> list[dict]:
    """Load a previously dumped factor library."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
