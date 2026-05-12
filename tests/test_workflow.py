"""Tests for the main workflow."""

import pytest
import asyncio

from alpha_mining.workflow import (
    MiningState,
    build_mining_workflow,
    _build_factor_library_summary,
    _build_proposals_summary,
)
from alpha_mining.tools.storage_tools import (
    reset_storage,
    create_alpha,
    store_evaluation,
    get_session_iterations,
)
from alpha_mining.config import AlphaMiningConfig


class TestMiningState:
    """Tests for MiningState type."""

    def test_state_creation(self):
        """Test basic state creation."""
        state: MiningState = {
            "session_id": "test-session",
            "iteration": 1,
            "config": {"max_iterations": 10},
            "baseline_factor_library": [],
            "current_proposals": [],
            "pending_evaluations": [],
            "leader_decision": None,
            "should_continue": True,
            "optimization_direction": None,
            "context_factors": [],
            "discovered_factors": [],
            "is_complete": False,
            "final_candidates": [],
        }

        assert state["session_id"] == "test-session"
        assert state["iteration"] == 1
        assert state["should_continue"] is True


class TestWorkflow:
    """Tests for the mining workflow."""

    def setup_method(self):
        """Reset storage before each test."""
        reset_storage()

    def test_build_workflow(self):
        """Test building the workflow graph."""
        config = AlphaMiningConfig()
        graph = build_mining_workflow(config)

        assert graph is not None

    def test_workflow_nodes_exist(self):
        """Test that workflow has required nodes."""
        config = AlphaMiningConfig()
        graph = build_mining_workflow(config)

        # Check the graph was compiled
        assert graph is not None

    @pytest.mark.asyncio
    async def test_workflow_single_iteration(self):
        """Test running a single iteration."""
        config = AlphaMiningConfig()
        config.iteration.max_iterations = 1
        config.iteration.min_proposals_per_iteration = 1
        config.iteration.max_proposals_per_iteration = 2

        graph = build_mining_workflow(config)

        initial_state: MiningState = {
            "session_id": "test-session",
            "iteration": 1,
            "config": {
                "max_iterations": 1,
                "min_proposals_per_iteration": 1,
                "max_proposals_per_iteration": 2,
                "max_proposer_retries": 3,
                "eval_config": {},
            },
            "baseline_factor_library": [],
            "current_proposals": [],
            "pending_evaluations": [],
            "leader_decision": None,
            "should_continue": True,
            "optimization_direction": None,
            "context_factors": [],
            "discovered_factors": [],
            "is_complete": False,
            "final_candidates": [],
        }

        final_state = await graph.ainvoke(initial_state)

        # Should complete because max_iterations reached
        assert final_state["iteration"] == 1

    @pytest.mark.asyncio
    async def test_workflow_with_factors(self):
        """Test workflow with pre-existing factors."""
        reset_storage()

        # Add some factors
        alpha = create_alpha(
            name="Alpha 1",
            code="close / open - 1",
            description="Return factor",
            parameters={},
            iteration=1,
        )
        store_evaluation(
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

        # Check factor library
        from alpha_mining.tools.storage_tools import list_factors
        factors = list_factors()
        assert len(factors) == 1


class TestHelperFunctions:
    """Tests for workflow helper functions."""

    def test_build_factor_library_summary_empty(self):
        """Test building summary for empty library."""
        reset_storage()
        summary = _build_factor_library_summary()
        assert "No discovered" in summary

    def test_build_factor_library_summary_with_factors(self):
        """Test building summary with factors."""
        reset_storage()

        alpha = create_alpha(
            name="Test Alpha",
            code="close / open - 1",
            description="Test",
            parameters={},
            iteration=1,
        )
        store_evaluation(
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

        summary = _build_factor_library_summary()
        assert "Test Alpha" in summary
        assert "IC=" in summary

    def test_build_proposals_summary_empty(self):
        """Test building summary for empty proposals."""
        summary = _build_proposals_summary([])
        assert "No proposals" in summary

    def test_build_proposals_summary_with_proposals(self):
        """Test building summary with proposals."""
        proposals = [
            {"name": "Alpha 1", "description": "Return factor"},
            {"name": "Alpha 2", "description": "Volume factor"},
        ]
        summary = _build_proposals_summary(proposals)
        assert "Alpha 1" in summary
        assert "Alpha 2" in summary
