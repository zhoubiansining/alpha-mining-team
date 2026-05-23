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
    delete_factor_record,
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

    def test_delete_factor_record(self):
        """Test deleting a factor through the pure internal function."""
        alpha = create_alpha(
            name="Delete Me",
            code="close / open - 1",
            description="Temporary factor",
            parameters={},
            iteration=1,
        )

        assert delete_factor_record(alpha.id) is True
        assert get_factor_by_id(alpha.id) is None
        assert delete_factor_record(alpha.id) is False


class TestEvaluatorTools:
    """Tests for evaluator tools."""

    def test_call_evaluator_http_success(self, monkeypatch):
        """Test evaluator HTTP contract."""
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "status": "success",
                    "alpha_id": "alpha-1",
                    "metrics": {
                        "ic_mean": 0.05,
                        "ic_std": 0.1,
                        "ir": 0.5,
                        "sharpe": 1.2,
                        "max_drawdown": 0.1,
                        "turnover": 0.3,
                        "long_short_return": 0.08,
                        "win_rate": 0.55,
                    },
                    "error_message": None,
                }

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr("alpha_mining.tools.eval_tools.httpx.post", fake_post)

        result = call_evaluator.invoke({
            "alpha_description": "Simple return factor",
            "alpha_code": "close / open - 1",
            "parameters": {"window": 20},
            "eval_config": {
                "alpha_id": "alpha-1",
                "evaluator_endpoint": "http://testserver/evaluate",
                "evaluator_timeout": 12,
                "universe": "HS300",
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            },
        })

        assert result["status"] == "success"
        assert captured["url"] == "http://testserver/evaluate"
        assert captured["timeout"] == 12
        assert captured["json"]["alpha_id"] == "alpha-1"
        assert captured["json"]["parameters"] == {"window": 20}
        assert captured["json"]["eval_config"] == {
            "universe": "HS300",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        }
        assert "metrics" in result
        assert "ic_mean" in result["metrics"]
        assert "sharpe" in result["metrics"]

    def test_call_evaluator_infers_compliance_error(self, monkeypatch):
        """Test legacy evaluator errors are mapped for workflow retry."""
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "status": "error",
                    "alpha_id": "alpha-bad",
                    "metrics": None,
                    "error_message": "未能在因子代码中发现可执行的 Class 类定义",
                }

        monkeypatch.setattr(
            "alpha_mining.tools.eval_tools.httpx.post",
            lambda url, json, timeout: FakeResponse(),
        )

        result = call_evaluator.invoke({
            "alpha_description": "Test alpha",
            "alpha_code": "close / open - 1",
            "parameters": {},
            "eval_config": {"alpha_id": "alpha-bad"},
        })

        assert result["status"] == "error"
        assert result["error_code"] == "COMPLIANCE_ERROR"
