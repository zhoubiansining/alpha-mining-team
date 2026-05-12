"""Tests for tools."""

import pytest

from alpha_mining.tools.storage_tools import (
    reset_storage,
    create_alpha,
    store_evaluation,
    store_feedback,
    list_factors,
    get_factor_by_id,
    get_evaluation_by_alpha_id,
    get_feedbacks_by_alpha_id,
    create_session,
    get_session_iterations,
)
from alpha_mining.tools.eval_tools import call_evaluator


class TestStorageTools:
    """Tests for storage tools."""

    def setup_method(self):
        """Reset storage before each test."""
        reset_storage()

    def test_create_alpha(self):
        """Test creating an alpha factor."""
        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            parameters={"window": 20},
            iteration=1,
            intuition="Captures daily returns",
        )

        assert alpha.id is not None
        assert alpha.name == "Test Alpha"
        assert alpha.code == "close / open - 1"

    def test_list_factors_empty(self):
        """Test listing empty factor library."""
        factors = list_factors()
        assert factors == []

    def test_save_and_get_alpha(self):
        """Test saving and retrieving an alpha."""
        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            parameters={"window": 20},
            iteration=1,
        )

        factors = list_factors()
        assert len(factors) == 1
        assert factors[0].name == "Test Alpha"
        assert factors[0].id == alpha.id

    def test_get_factor_by_id(self):
        """Test getting factor by ID."""
        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            parameters={},
            iteration=1,
        )

        retrieved = get_factor_by_id(alpha.id)
        assert retrieved is not None
        assert retrieved.name == "Test Alpha"

    def test_get_factor_by_id_not_found(self):
        """Test getting non-existent factor."""
        factor = get_factor_by_id("non-existent-id")
        assert factor is None

    def test_store_evaluation(self):
        """Test storing evaluation results."""
        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            parameters={},
            iteration=1,
        )

        evaluation = store_evaluation(
            alpha_id=alpha.id,
            metrics={
                "ic_mean": 0.05,
                "ic_std": 0.1,
                "ir": 0.5,
                "sharpe": 1.5,
                "max_drawdown": 0.1,
                "turnover": 0.3,
            },
        )

        assert evaluation.alpha_id == alpha.id
        assert evaluation.ic_mean == 0.05

        # Verify evaluation is retrievable
        retrieved = get_evaluation_by_alpha_id(alpha.id)
        assert retrieved is not None
        assert retrieved.ir == 0.5

    def test_store_feedback(self):
        """Test storing critic feedback."""
        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            parameters={},
            iteration=1,
        )

        feedback = store_feedback(
            alpha_id=alpha.id,
            iteration=1,
            ratings={"quality": 4, "novelty": 3},
            factual_observations=["IC is moderate"],
            concerns=["May be overfitted"],
            actionable_suggestions=["Try different window"],
            can_proceed=True,
        )

        assert feedback.id is not None
        assert feedback.alpha_id == alpha.id

        # Verify feedback is retrievable
        feedbacks = get_feedbacks_by_alpha_id(alpha.id)
        assert len(feedbacks) == 1
        assert feedbacks[0].can_proceed is True

    def test_get_critic_suggestions(self):
        """Test getting critic suggestions for alpha."""
        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            parameters={},
            iteration=1,
        )

        store_feedback(
            alpha_id=alpha.id,
            iteration=1,
            ratings={},
            factual_observations=["Obs 1"],
            concerns=["Concern 1"],
            actionable_suggestions=["Suggestion 1", "Suggestion 2"],
            can_proceed=True,
        )

        feedbacks = get_feedbacks_by_alpha_id(alpha.id)
        assert len(feedbacks) == 1
        assert len(feedbacks[0].actionable_suggestions) == 2

    def test_create_session(self):
        """Test creating a mining session."""
        session = create_session({"max_iterations": 10})
        assert session.session_id is not None
        assert session.is_active is True
        assert session.current_iteration == 0


class TestEvaluatorTools:
    """Tests for evaluator tools."""

    def test_call_evaluator_mock(self):
        """Test mock evaluator call."""
        result = call_evaluator.invoke({
            "alpha_description": "Simple return factor",
            "alpha_code": "close / open - 1",
            "parameters": {"window": 20},
            "eval_config": {
                "symbols": ["A-share"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            },
        })

        assert result["status"] == "success"
        assert "metrics" in result
        assert "ic_mean" in result["metrics"]
        assert "sharpe" in result["metrics"]

    def test_call_evaluator_returns_valid_metrics(self):
        """Test that evaluator returns valid metrics."""
        result = call_evaluator.invoke({
            "alpha_description": "Test alpha",
            "alpha_code": "close / open - 1",
            "parameters": {},
            "eval_config": {},
        })

        assert result["status"] == "success"
        metrics = result["metrics"]

        # Check metrics are within reasonable ranges
        assert -1 <= metrics["ic_mean"] <= 1
        assert 0 <= metrics["ic_std"] <= 1
        assert -5 <= metrics["sharpe"] <= 10
        assert 0 <= metrics["max_drawdown"] <= 1
        assert 0 <= metrics["turnover"] <= 1
