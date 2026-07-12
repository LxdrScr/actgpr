# actgpr — Project Architecture & SCE Practices

> Authoritative design document. Update this file when an architecture decision changes.
> Domain terminology is defined in `CONTEXT.md` — use it in all code, docstrings, and logs.

---

## 1. What this package does

`actgpr` finds the minimum of an expensive-to-evaluate scalar **Objective** by
iteratively fitting a Gaussian Process **Surrogate** and using an **Acquisition
function** to pick the most informative next **input point**.

**Loop:**
1. Evaluate Objective at initial input points (provided directly in user configuration)
2. Repeat:
   - Fit Surrogate to all evaluations so far
   - Maximise Acquisition function → choose next input point
   - Evaluate Objective at that input point
3. Until convergence criterion fires (whichever comes first):
   - Posterior std at current best falls below `std_threshold`
   - Total evaluations reach `max_evaluations`
4. Return the current best input point and its Objective value

---

## 2. Tools at a Glance

| Area | Tool |
|---|---|
| Packaging & environments | **Poetry** (`pyproject.toml` + `poetry.lock`) |
| Version control | **Git** + **GitHub** (GitHub Flow) |
| Style | **PEP 8** · auto-formatted with **black** · linted with **ruff** |
| Testing | **pytest** (unit · integration · regression · sanity checks) |
| CI/CD | **GitHub Actions** (lint → test → docs → deploy) |
| Reproducibility | MRR: `config.json` · `meta.json` · `run.log` · fixed seed |
| API docs | **Sphinx** → GitHub Pages |

---

## 3. Project Structure

Follows the **`src/` layout** with **PEP 8** naming throughout. The repository root is `actgpr/`, and the importable package is `actgpr/`.

```
actgpr/                              ← git repo root
│
├── pyproject.toml                   ← single config file (Poetry + black + ruff + pytest)
├── poetry.lock                      ← pinned dependency versions
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
│
├── .github/
│   └── workflows/
│       └── ci.yml                   ← GitHub Actions pipeline
│
├── docs/
│   ├── conf.py                      ← Sphinx config
│   ├── index.rst
│   └── api/                         ← auto-generated API reference
│
├── tests/
│   ├── unit/
│   │   ├── test_mrr.py              ← tests for MRR functionality
│   │   └── test_optimisation_run.py ← unit tests for loop/run configuration
│   ├── integration/
│   │   └── test_integration.py      ← integration tests for the full loop
│   └── regression/
│       └── data/
│           └── quadratic_baseline.csv ← stored reference outputs
│
└── src/
    └── actgpr/                      ← importable package
        ├── __init__.py
        ├── run.py                   ← OptimisationRun (owns loop & MRR write actions)
        ├── mrr.py                   ← MRR module (free functions for artifact writes)
        ├── objective.py             ← Simple objective function (placeholder/empty)
        ├── surrogate.py             ← Surrogate GP model (placeholder/empty; GPyTorch backend)
        └── acquisition.py           ← Acquisition function (placeholder/empty)
```

---

## 4. Architecture Decisions

These decisions were agreed upon during the architecture grilling session:

### Top-level class
- **Name**: `OptimisationRun` (in `src/actgpr/run.py`)
- **Responsibility**: Owns the optimisation loop and coordinates all Minimal Reproducible Run (MRR) writes.

### MRR Module
- **Type**: A Python module (`src/actgpr/mrr.py`) containing free functions (no class, no state).
- **Responsibility**: Handles all file I/O for MRR artifacts. Called directly by `OptimisationRun`.

### User Configuration
- **Format**: JSON (`config.json`), written by the user.
- **Handling**: `OptimisationRun` reads it directly to configure the run (no CLI).
- **Integrity**: The SHA-256 checksum of the user configuration is verified before execution starts.
- **Structure**: Includes run configuration, search bounds, convergence settings, explicit initial input points (no separate sampler), and nested blocks for surrogate/objective settings.
- **Open Item**: The specific configuration schema for surrogate and objective are not yet decided.

### Write Timing & Formats
- **Start of Run**: A copy of `config.json`, a `meta.json` (git commit, package/Python/library versions, platform, and timestamps), and a `manifest.json` (containing the SHA-256 checksums of inputs) are written at startup.
- **During Run**: Per-step events, warnings, and summary statistics (e.g., posterior mean/std and evaluation indices) are appended to a `run.log`.
- **End of Run (Deferred-Write Accumulator)**: Evaluation results are accumulated privately in memory (`_results: list[dict]`) during the loop and written all-at-once at the end to a self-describing HDF5 file (`results.h5`).
- **HDF5 Self-Description**: Configuration parameters (bounds, thresholds) are stored as root group attributes; evaluation datasets (input points, posterior means/stds, objective outputs) are stored in the datasets.

### Output Location
- **Directory**: All outputs from a run are stored in a run-specific subfolder inside a `results/` directory.
- **Run ID Format**: Subfolders are named using the standardized format:
  `YYYY-MM-DD_<experiment>_<key-params>_seed<N>_v<version>_run<NNNN>`

### Convergence
- **Criterion**: The loop terminates when either the maximum Expected Improvement (EI) score across all candidates falls below `ei_threshold` OR the total number of evaluations reaches `max_evaluations` (whichever fires first).

### Surrogate Backend
- **Framework**: `GPyTorch` (selected for its commercial-friendly MIT license and scale/GPU compatibility).
- **Numerical Stability**: 
  - To prevent `NumericalWarning: A not p.d., added jitter` messages from Cholesky decomposition, all inputs (`train_x`, `train_y`, `test_x`) and the GP model parameters are converted to double precision (`float64`).
  - Marginal log likelihood and predictive calculations are wrapped in `gpytorch.settings.cholesky_jitter(1e-4)` to ensure robust numerical operations from the start of the run.

---

## 5. Version Control

- **GitHub Flow:** `main` always works; all changes via feature branches + pull requests.
- Commit messages describe *why*, not *what*.
- Tagged releases: `git tag v0.1.0` — required for FAIR citability.

---

## 6. Testing

| Type | What |
|---|---|
| Unit | Individual methods and functions in isolation (e.g. `OptimisationRun` configuration, `mrr` functions) |
| Integration | Full loop on a simple quadratic objective with fixed seed |
| Regression | Compare loop output against stored CSV baseline (same seed → same result) |
| Sanity | `assert np.all(posterior_std >= 0)` · `assert np.all(np.isfinite(posterior_mean))` |
| Error-handling | `pytest.raises` for bad inputs (wrong type, out-of-bounds, etc.) |

*Fix a random seed in every test. Run with `pytest --cov` for coverage.*

---

## 7. CI/CD (GitHub Actions)

```
push → lint (black + ruff) → test (pytest) → docs (sphinx) → deploy (PyPI on tag)
```

Gating is strictly enforced: any failure (e.g. in lint) terminates the pipeline immediately.

---

## 8. Definition of Done ✓

- [ ] `poetry install` works on a clean machine
- [ ] `pytest` passes (all test types)
- [ ] `black --check` + `ruff` pass
- [ ] Sphinx docs build without warnings
- [ ] `LICENSE` present with SPDX identifier (e.g. MIT)
- [ ] `CITATION.cff` present (minimum citation fields to be added in the future)
- [ ] `CHANGELOG.md` updated with changes
- [ ] `poetry.lock` committed

---

## 9. Open Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Objective structure | **Open** — simple Python function/construct for one or more inputs |
| 2 | Surrogate configuration | **Open** — configuration fields TBD (GPyTorch chosen as backend) |
| 3 | Acquisition function | **Open** — architecture and configuration fields TBD |
| 4 | Dimensionality | **1D first** — scalar input points; nD extension planned |
| 5 | Minimum citation fields | **Planned for the future** — add `cff-version`, `authors`, `title`, `version`, `date-released`, and `url` to `CITATION.cff` |
