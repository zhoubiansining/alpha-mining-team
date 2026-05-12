"""Tests for data models."""

import pytest
from datetime import datetime

from alpha_mining.schemas.alpha import AlphaExpression
from alpha_mining.schemas.evaluation import AlphaEvaluation
from alpha_mining.schemas.history import (
    CriticFeedback,
    LeaderDecision,
    IterationHistory,
    MiningSession,
)


class TestAlphaExpression:
    """Tests for AlphaExpression model."""

    def test_create_alpha(self):
        """Test basic alpha creation."""
        alpha = AlphaExpression(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            iteration=1,
        )

        assert alpha.id is not None
        assert alpha.name == "Test Alpha"
        assert alpha.code == "close / open - 1"
        assert alpha.iteration == 1
        assert alpha.created_at is not None

    def test_alpha_to_summary(self):
        """Test alpha summary conversion."""
        alpha = AlphaExpression(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            iteration=1,
            intuition="Captures daily returns",
        )

        summary = alpha.to_summary()
        assert summary["name"] == "Test Alpha"
        assert summary["iteration"] == 1
        assert "id" in summary

    def test_alpha_to_dict(self):
        """Test alpha dict conversion."""
        alpha = AlphaExpression(
            name="Test Alpha",
            code="close / open - 1",
            description="Simple return factor",
            iteration=1,
        )

        d = alpha.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "Test Alpha"
        assert "id" in d


class TestAlphaEvaluation:
    """Tests for AlphaEvaluation model."""

    def test_create_evaluation(self):
        """Test basic evaluation creation."""
        eval_result = AlphaEvaluation(
            alpha_id="test-123",
            ic_mean=0.05,
            ic_std=0.1,
            ir=0.5,
            sharpe=1.5,
            max_drawdown=0.1,
            turnover=0.3,
        )

        assert eval_result.alpha_id == "test-123"
        assert eval_result.ic_mean == 0.05
        assert eval_result.ir == 0.5

    def test_create_mock_evaluation(self):
        """Test mock evaluation creation."""
        mock_eval = AlphaEvaluation.create_mock(
            alpha_id="mock-123",
            ic_mean=0.06,
            sharpe=1.2,
        )

        assert mock_eval.alpha_id == "mock-123"
        assert mock_eval.ic_mean == 0.06
        assert mock_eval.ir == pytest.approx(0.06 / mock_eval.ic_std)

    def test_evaluation_summary(self):
        """Test evaluation summary conversion."""
        eval_result = AlphaEvaluation(
            alpha_id="test-123",
            ic_mean=0.05,
            ic_std=0.1,
            ir=0.5,
            sharpe=1.5,
            max_drawdown=0.1,
            turnover=0.3,
        )

        summary = eval_result.to_summary()
        assert summary["ic_mean"] == 0.05
        assert summary["ir"] == 0.5
        assert "alpha_id" in summary


class TestCriticFeedback:
    """Tests for CriticFeedback model."""

    def test_create_feedback(self):
        """Test basic feedback creation."""
        feedback = CriticFeedback(
            id="feedback-123",
            alpha_id="alpha-456",
            iteration=1,
            ratings={"quality": 4, "novelty": 3},
            factual_observations=["IC is moderate"],
            concerns=["May be overfitted"],
            actionable_suggestions=["Try different window"],
            can_proceed=True,
        )

        assert feedback.id == "feedback-123"
        assert feedback.can_proceed is True
        assert len(feedback.actionable_suggestions) == 1

    def test_feedback_summary(self):
        """Test feedback summary conversion."""
        feedback = CriticFeedback(
            id="feedback-123",
            alpha_id="alpha-456",
            iteration=1,
            ratings={"quality": 4},
            concerns=["Concern 1"],
            actionable_suggestions=["Suggestion 1"],
            can_proceed=True,
        )

        summary = feedback.to_summary()
        assert summary["id"] == "feedback-123"
        assert summary["can_proceed"] is True


class TestLeaderDecision:
    """Tests for LeaderDecision model."""

    def test_create_decision_continue(self):
        """Test creating a continue decision."""
        decision = LeaderDecision(
            should_continue=True,
            reason="Quality improving",
            optimization_direction="Focus on momentum",
            focus_areas=["momentum", "volume"],
            selected_for_context=["alpha-1", "alpha-2"],
            suggestions_to_proposer=["Try shorter windows"],
        )

        assert decision.should_continue is True
        assert decision.optimization_direction == "Focus on momentum"
        assert len(decision.selected_for_context) == 2

    def test_create_decision_stop(self):
        """Test creating a stop decision."""
        decision = LeaderDecision(
            should_continue=False,
            reason="Max iterations reached",
            termination_reason="Max iterations reached",
            final_candidates=["alpha-1", "alpha-2", "alpha-3"],
        )

        assert decision.should_continue is False
        assert decision.final_candidates is not None
        assert len(decision.final_candidates) == 3


class TestMiningSession:
    """Tests for MiningSession model."""

    def test_create_session(self):
        """Test basic session creation."""
        session = MiningSession(
            session_id="session-123",
            start_time=datetime.now(),
            config={"max_iterations": 10},
        )

        assert session.session_id == "session-123"
        assert session.current_iteration == 0
        assert session.is_active is True

    def test_session_defaults(self):
        """Test session default values."""
        session = MiningSession(
            session_id="session-123",
            start_time=datetime.now(),
        )

        assert session.iterations == []
        assert session.final_candidates == []
        assert session.end_time is None
