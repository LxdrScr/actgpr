"""Plotting utilities for active GPR optimisation."""

import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from actgpr.surrogate import GPyTorchSurrogate


def plot_surrogate(
    surrogate: GPyTorchSurrogate,
    test_x: torch.Tensor,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot the surrogate predictions against training data.

    Parameters
    ----------
    surrogate : GPyTorchSurrogate
        A fitted surrogate model.
    test_x : torch.Tensor of shape (m,)
        The test input points used to compute predictions for plotting.
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
        Set to False when composing multiple plots.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.

    Raises
    ------
    RuntimeError
        If the surrogate has not been fitted or has no training data.
    """
    if (
        surrogate.model is None
        or surrogate.likelihood is None
        or surrogate.train_x is None
        or surrogate.train_y is None
    ):
        raise RuntimeError("The surrogate must be fitted before plotting.")

    preds = surrogate.predict(test_x)
    observed_pred = preds["observed_pred"]
    f_mean = preds["f_mean"]

    assert isinstance(observed_pred, gpytorch.distributions.MultivariateNormal)
    assert isinstance(f_mean, torch.Tensor)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    with torch.no_grad():
        lower, upper = observed_pred.confidence_region()

        ax.plot(
            surrogate.train_x.numpy(),
            surrogate.train_y.numpy(),
            "k*",
            markersize=10,
            label="Training data",
        )
        ax.plot(test_x.numpy(), f_mean.numpy(), "b", label="Mean prediction")
        ax.fill_between(
            test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label="95% CI"
        )
        ax.legend()

    if show:
        plt.show()

    return fig, ax


def plot_acquisition(
    candidates: torch.Tensor,
    ei_scores: torch.Tensor,
    next_point: float | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot the Expected Improvement acquisition landscape.

    Parameters
    ----------
    candidates : torch.Tensor of shape (m,)
        The candidate input points that were scored.
    ei_scores : torch.Tensor of shape (m,)
        The EI score for each candidate point.
    next_point : float or None, optional
        The selected next input point. If provided, a vertical line is drawn.
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
        Set to False when composing multiple plots.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(candidates.numpy(), ei_scores.numpy(), "g", label="Expected Improvement")
    ax.fill_between(candidates.numpy(), 0, ei_scores.numpy(), alpha=0.2, color="g")

    if next_point is not None:
        ax.axvline(
            next_point,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Next point (x={next_point:.2f})",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("EI score")
    ax.legend()

    if show:
        plt.show()

    return fig, ax
