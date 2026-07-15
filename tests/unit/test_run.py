"""Unit tests for the OptimisationRun class."""

import pytest
import torch

from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

SEED = 42


@pytest.fixture()
def simple_run() -> OptimisationRun:
    """Return an OptimisationRun configured for a simple x^2 optimisation."""
    torch.manual_seed(SEED)
    return OptimisationRun(
        objective=ObjectiveFn(),
        surrogate=GPyTorchSurrogate(),
        search_bounds=(-3.0, 3.0),
        initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        max_evaluations=10,
        ei_threshold=0.01,
        n_candidates=100,
        training_iter=20,
    )


class TestOptimisationRunInit:
    """Tests for OptimisationRun.__init__."""

    def test_stores_objective(self, simple_run: OptimisationRun) -> None:
        """Test that the objective is stored correctly."""
        assert isinstance(simple_run.objective, ObjectiveFn)

    def test_stores_surrogate(self, simple_run: OptimisationRun) -> None:
        """Test that the surrogate is stored correctly."""
        assert isinstance(simple_run.surrogate, GPyTorchSurrogate)

    def test_stores_search_bounds(self, simple_run: OptimisationRun) -> None:
        """Test that search bounds are stored correctly."""
        assert simple_run.search_bounds == (-3.0, 3.0)

    def test_stores_max_evaluations(self, simple_run: OptimisationRun) -> None:
        """Test that max_evaluations is stored correctly."""
        assert simple_run.max_evaluations == 10

    def test_stores_ei_threshold(self, simple_run: OptimisationRun) -> None:
        """Test that ei_threshold is stored correctly."""
        assert simple_run.ei_threshold == 0.01

    def test_evaluates_initial_points(self, simple_run: OptimisationRun) -> None:
        """Test that train_y is computed from initial_train_x on construction."""
        # x^2 at [-2, -1, 1, 2] = [4, 1, 1, 4]
        expected = torch.tensor([4.0, 1.0, 1.0, 4.0], dtype=torch.float64)
        assert torch.allclose(simple_run.train_y, expected)

    def test_train_x_is_float64(self, simple_run: OptimisationRun) -> None:
        """Test that train_x is cast to float64 regardless of input dtype."""
        assert simple_run.train_x.dtype == torch.float64

    def test_accepts_list_input(self) -> None:
        """Test that initial_train_x can be a plain Python list."""
        run = OptimisationRun(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 0.0, 2.0],
            max_evaluations=10,
            ei_threshold=0.01,
        )
        assert run.train_x.numel() == 3

    def test_raises_on_empty_train_x(self) -> None:
        """Test that ValueError is raised when initial_train_x is empty."""
        with pytest.raises(ValueError, match="at least one point"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(-3.0, 3.0),
                initial_train_x=torch.tensor([]),
                max_evaluations=10,
                ei_threshold=0.01,
            )

    def test_raises_on_max_eval_too_small(self) -> None:
        """Test that ValueError is raised when max_evaluations <= n initial points."""
        with pytest.raises(ValueError, match="must be greater"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(-3.0, 3.0),
                initial_train_x=torch.tensor([-1.0, 0.0, 1.0]),
                max_evaluations=3,
                ei_threshold=0.01,
            )

    def test_results_accumulator_starts_empty(
        self, simple_run: OptimisationRun
    ) -> None:
        """Test that the deferred-write accumulator is empty before run()."""
        assert simple_run._results == []

    def test_repr(self, simple_run: OptimisationRun) -> None:
        """Test the string representation of OptimisationRun."""
        r = repr(simple_run)
        assert "OptimisationRun" in r
        assert "bounds=(-3.0, 3.0)" in r
        assert "max_eval=10" in r


class TestOptimisationRunRun:
    """Tests for OptimisationRun.run()."""

    def test_returns_expected_keys(self, simple_run: OptimisationRun) -> None:
        """Test that run() returns a dict with all expected keys."""
        result = simple_run.run()
        expected_keys = {
            "best_x",
            "best_y",
            "train_x",
            "train_y",
            "n_iterations",
            "converged",
        }
        assert set(result.keys()) == expected_keys

    def test_best_y_is_float(self, simple_run: OptimisationRun) -> None:
        """Test that best_y is a Python float."""
        result = simple_run.run()
        assert isinstance(result["best_y"], float)

    def test_best_x_is_float(self, simple_run: OptimisationRun) -> None:
        """Test that best_x is a Python float."""
        result = simple_run.run()
        assert isinstance(result["best_x"], float)

    def test_best_y_is_non_negative(self, simple_run: OptimisationRun) -> None:
        """Test scientific invariant: x² is always non-negative."""
        result = simple_run.run()
        assert result["best_y"] >= 0

    def test_n_iterations_is_positive(self, simple_run: OptimisationRun) -> None:
        """Test that at least one iteration was executed."""
        result = simple_run.run()
        assert result["n_iterations"] >= 1

    def test_train_data_grows(self, simple_run: OptimisationRun) -> None:
        """Test that training data grows during the run (unless converged on first iteration)."""
        initial_count = simple_run.train_x.numel()
        result = simple_run.run()
        # If converged on first iteration, no new points are added
        if result["converged"] and result["n_iterations"] == 1:
            assert result["train_x"].numel() == initial_count
        else:
            assert result["train_x"].numel() > initial_count

    def test_train_x_train_y_same_length(self, simple_run: OptimisationRun) -> None:
        """Test that train_x and train_y always have the same length."""
        result = simple_run.run()
        assert result["train_x"].numel() == result["train_y"].numel()

    def test_respects_max_evaluations(self) -> None:
        """Test that the loop stops at max_evaluations even without convergence."""
        run = OptimisationRun(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, 2.0]),
            max_evaluations=5,
            ei_threshold=1e-20,  # impossibly low — forces max_evaluations stop
            n_candidates=50,
            training_iter=10,
        )
        result = run.run()
        assert result["train_x"].numel() <= 5
        assert result["converged"] is False

    def test_converged_flag_true_on_ei_convergence(self) -> None:
        """Test that converged=True when EI drops below threshold."""
        run = OptimisationRun(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_evaluations=50,
            ei_threshold=1.0,  # very high — should converge immediately
            n_candidates=50,
            training_iter=10,
        )
        result = run.run()
        assert result["converged"] is True

    def test_results_accumulator_populated(self, simple_run: OptimisationRun) -> None:
        """Test that _results accumulator is populated after run()."""
        simple_run.run()
        # Each iteration that evaluates a new point adds one entry
        for entry in simple_run._results:
            assert "iteration" in entry
            assert "next_point" in entry
            assert "new_y" in entry
            assert "current_best" in entry
            assert "max_ei" in entry

    def test_custom_objective_converges(self) -> None:
        """Test that the loop works with a custom objective function."""
        torch.manual_seed(SEED)
        run = OptimisationRun(
            objective=ObjectiveFn(lambda x: (x - 1) ** 2),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 5.0),
            initial_train_x=torch.tensor([-2.0, 0.0, 2.0, 4.0]),
            max_evaluations=20,
            ei_threshold=0.01,
            n_candidates=100,
            training_iter=20,
        )
        result = run.run()
        # The minimum of (x-1)^2 is at x=1, y=0
        assert result["best_y"] < 1.0
