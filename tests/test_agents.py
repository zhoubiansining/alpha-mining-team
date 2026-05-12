"""Tests for agents parsing functions."""

import pytest
import json

from alpha_mining.agents.proposer import parse_alpha_proposals, _validate_alpha_list
from alpha_mining.agents.critic import parse_critic_feedback


class TestProposerParser:
    """Tests for Proposer response parsing."""

    def test_parse_single_alpha_json(self):
        """Test parsing single alpha JSON."""
        response = json.dumps({
            "name": "Test Alpha",
            "code": "close / open - 1",
            "description": "Simple return",
            "parameters": {},
            "intuition": "Captures returns",
        })

        result = parse_alpha_proposals(response)
        assert len(result) == 1
        assert result[0]["name"] == "Test Alpha"
        assert result[0]["code"] == "close / open - 1"

    def test_parse_multiple_alphas_json(self):
        """Test parsing multiple alphas JSON array."""
        response = json.dumps([
            {
                "name": "Alpha 1",
                "code": "close / open - 1",
                "description": "Return",
                "parameters": {},
                "intuition": "Returns",
            },
            {
                "name": "Alpha 2",
                "code": "volume / ts_mean(volume, 20)",
                "description": "Volume ratio",
                "parameters": {},
                "intuition": "Volume anomaly",
            },
        ])

        result = parse_alpha_proposals(response)
        assert len(result) == 2
        assert result[0]["name"] == "Alpha 1"
        assert result[1]["name"] == "Alpha 2"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns empty list."""
        response = "This is not valid JSON"
        result = parse_alpha_proposals(response)
        assert result == []

    def test_parse_empty_response(self):
        """Test parsing empty response returns empty list."""
        result = parse_alpha_proposals("")
        assert result == []

    def test_parse_partial_json(self):
        """Test parsing partial JSON with extra text."""
        response = """
        Here is my proposal:

        {"name": "Test Alpha", "code": "close / open - 1", "description": "Test", "parameters": {}, "intuition": "Test"}
        """
        result = parse_alpha_proposals(response)
        assert len(result) == 1
        assert result[0]["name"] == "Test Alpha"

    def test_parse_json_code_block(self):
        """Test parsing JSON in code block."""
        response = """
        ```json
        {"name": "Test Alpha", "code": "close / open - 1", "description": "Test", "parameters": {}, "intuition": "Test"}
        ```
        """
        result = parse_alpha_proposals(response)
        assert len(result) == 1
        assert result[0]["name"] == "Test Alpha"

    def test_validate_alpha_list(self):
        """Test validating alpha list."""
        items = [
            {"name": "Alpha 1", "code": "close/open-1", "extra": "data"},
            {"name": "Alpha 2", "code": "volume/mean(volume,20)"},
            {"invalid": "item"},  # Missing name/code
        ]
        result = _validate_alpha_list(items)
        assert len(result) == 2
        assert result[0]["name"] == "Alpha 1"
        assert result[1]["name"] == "Alpha 2"


class TestCriticParser:
    """Tests for Critic response parsing."""

    def test_parse_full_feedback(self):
        """Test parsing complete feedback JSON."""
        response = json.dumps({
            "ratings": {
                "theoretical_soundness": 4,
                "backtest_quality": 3,
                "robustness": 4,
                "implementation_quality": 5,
                "diversification": 3,
            },
            "factual_observations": [
                "IC mean: 0.05",
                "Sharpe ratio: 1.5",
            ],
            "concerns": [
                "May be overfitted",
                "Limited diversification",
            ],
            "actionable_suggestions": [
                "Try different window sizes",
                "Consider volume-based factors",
            ],
            "can_proceed": True,
        })

        result = parse_critic_feedback(response, "alpha-123", 1)

        assert result["alpha_id"] == "alpha-123"
        assert result["iteration"] == 1
        assert result["ratings"]["theoretical_soundness"] == 4
        assert result["can_proceed"] is True
        assert len(result["actionable_suggestions"]) == 2

    def test_parse_minimal_feedback(self):
        """Test parsing minimal feedback JSON."""
        response = json.dumps({
            "can_proceed": False,
            "ratings": {"quality": 3},
        })

        result = parse_critic_feedback(response, "alpha-456", 2)

        assert result["alpha_id"] == "alpha-456"
        assert result["can_proceed"] is False
        # Check defaults are applied
        assert "ratings" in result

    def test_parse_invalid_feedback(self):
        """Test parsing invalid feedback returns defaults."""
        response = "Not valid JSON"

        result = parse_critic_feedback(response, "alpha-789", 3)

        assert result["alpha_id"] == "alpha-789"
        assert result["can_proceed"] is False
        assert "Failed to parse" in result["factual_observations"][0]

    def test_parse_empty_feedback(self):
        """Test parsing empty feedback returns defaults."""
        result = parse_critic_feedback("", "alpha-empty", 4)

        assert result["alpha_id"] == "alpha-empty"
        assert result["can_proceed"] is False

    def test_parse_json_code_block(self):
        """Test parsing JSON in code block."""
        response = """
        ```json
        {
            "ratings": {"theoretical_soundness": 4},
            "can_proceed": true,
            "concerns": ["Good performance"]
        }
        ```
        """
        result = parse_critic_feedback(response, "alpha-codeblock", 5)

        assert result["alpha_id"] == "alpha-codeblock"
        assert result["can_proceed"] is True
        assert len(result["concerns"]) == 1
