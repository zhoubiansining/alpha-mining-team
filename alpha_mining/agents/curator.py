"""Curator agent implementation — post-Critic factor library gatekeeper."""

import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from deepagents import SubAgent

from alpha_mining.prompts.curator_prompts import (
    CURATOR_SYSTEM_PROMPT,
    CURATOR_USER_PROMPT,
)
from alpha_mining.tools.storage_tools import (
    get_factor_library,
    get_baseline_factor_library,
)


def get_default_llm(
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Get default LLM instance configured for OpenAI-compatible API."""
    return ChatOpenAI(
        model=model_name or os.getenv("CRITIC_MODEL", "gpt-4o"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0.0,
    )


def build_curator_agent(
    llm: ChatOpenAI | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> SubAgent:
    """
    Build the Curator agent.

    Args:
        llm: Pre-configured LLM instance
        model_name: Model name
        api_base: API base URL
        api_key: API key

    Returns:
        SubAgent configuration dict
    """
    if llm is None:
        llm = get_default_llm(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
        )

    return {
        "name": "curator",
        "description": "Gatekeeper that decides which new factors enter the library and which old factors to remove, based on Critic feedback and backtest metrics",
        "system_prompt": CURATOR_SYSTEM_PROMPT,
        "tools": [get_factor_library, get_baseline_factor_library],
        "model": llm,
    }


def parse_curator_decision(response: str) -> dict:
    """
    Parse the Curator's decision response.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed decision dict
    """
    if isinstance(response, dict):
        return _normalize_decision(response)

    if not response or not response.strip():
        return _default_decision()

    response = response.strip()

    # Try direct JSON parse
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return _normalize_decision(result)
    except json.JSONDecodeError:
        pass

    # Try ```json ... ``` blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_matches = re.findall(code_block_pattern, response)
    for match in code_matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict):
                return _normalize_decision(result)
        except json.JSONDecodeError:
            continue

    # Try first { to last }
    try:
        first_brace = response.index("{")
        last_brace = response.rindex("}")
        result = json.loads(response[first_brace:last_brace + 1])
        if isinstance(result, dict):
            return _normalize_decision(result)
    except (ValueError, json.JSONDecodeError):
        pass

    return _default_decision()


def _normalize_decision(data: dict) -> dict:
    """Normalize parsed JSON into the expected decision schema."""
    return {
        "admitted_factors": data.get("admitted_factors", []),
        "admission_reasons": data.get("admission_reasons", {}),
        "rejected_factors": data.get("rejected_factors", []),
        "rejection_reasons": data.get("rejection_reasons", {}),
        "factors_to_remove": data.get("factors_to_remove", []),
        "removal_reasons": data.get("removal_reasons", {}),
        "library_summary": data.get("library_summary", ""),
        "quality_assessment": data.get("quality_assessment", ""),
    }


def _default_decision() -> dict:
    """Return a safe default decision (admit nothing, remove nothing)."""
    return {
        "admitted_factors": [],
        "admission_reasons": {},
        "rejected_factors": [],
        "rejection_reasons": {},
        "factors_to_remove": [],
        "removal_reasons": {},
        "library_summary": "Curator parse failed — no changes made.",
        "quality_assessment": "Parse error",
    }
