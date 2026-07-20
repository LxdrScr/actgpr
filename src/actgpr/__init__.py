"""Active GPR (Gaussian Process Regression) Optimisation package.

This package finds the minimum of an expensive-to-evaluate scalar objective
function by iteratively fitting a Gaussian Process surrogate and using active
learning.

Exported classes
----------------
OptimisationRun
    Orchestrates the active optimisation loop and MRR artifact writes.
ObjectiveFn
    Wraps the scalar Objective function being minimised.
GPyTorchSurrogate
    Gaussian Process surrogate backend built on GPyTorch.
Acquisition
    Expected Improvement acquisition function.
"""

from actgpr.acquisition import Acquisition
from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

__version__ = "0.1.0"

__all__ = [
    "Acquisition",
    "ObjectiveFn",
    "OptimisationRun",
    "GPyTorchSurrogate",
    "__version__",
]
