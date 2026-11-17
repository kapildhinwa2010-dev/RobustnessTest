# RobustnessTest

RobustnessTest is a small NumPy utility for comparing four matrix-compression approximations and observing how their reconstruction error changes when additive Gaussian noise is applied to the input. It evaluates matrices, not trained models or their predictions. The project metadata declares version `0.1.0`; the available tests are synthetic and do not establish general performance or real-world robustness.

## What is implemented

The module `robustnesstest.py` provides:

- `svd_compress`: truncated SVD reconstruction, with a default rank of 4.
- `quantize`: uniform min–max quantization, with a default of 8 bits (accepted range: 1–16).
- `fourier_compress`: least-squares reconstruction using a constant term and a configurable number of sine/cosine harmonics (default: 4).
- `hybrid_compress`: truncated SVD followed by quantization.
- `benchmark`: applies the four fixed default configurations to a clean matrix and returns relative Frobenius reconstruction error and a `passed` flag. That flag means only that the error is below the code's fixed `0.5` threshold; it is not an independent quality or robustness guarantee.
- `add_noise`, `noise_sweep`, and `robustness_report`: add Gaussian noise and report each method's relative error against the original clean matrix. The report's `most_robust` field selects the method with the smallest change between the first and last supplied noise levels; results depend on the matrix and the supplied noise-level sequence.

Inputs are required to be non-empty, two-dimensional matrices. The project has no command-line interface, dataset, trained-model integration, or built-in evaluation of prediction accuracy.

## Use from a source checkout

There is no CLI. The Python modules can be imported from the repository root when NumPy is available. The project declares Python 3.9 or later and NumPy 1.24 or later. Install the listed dependencies from the checkout:

```bash
python -m pip install -r requirements.txt
```

Then, from the repository root, call the module directly:

```python
import numpy as np
from robustnesstest import benchmark, robustness_report

matrix = np.array([
    [1.0, 0.5, 0.0, -0.5],
    [0.0, 0.5, 1.0, 0.5],
])

for result in benchmark(matrix):
    print(result["method"], result["relative_error"], result["passed"])

report = robustness_report(matrix, sigmas=(0.0, 0.01, 0.05, 0.1))
print(report["most_robust"])
```

The repository declares project metadata in `pyproject.toml`, but a distribution install is not currently verified: `python -m pip install --dry-run .` fails because setuptools detects multiple top-level modules in the flat layout (`activation_compress_helpers` and `robustnesstest`). Use the source checkout rather than relying on `pip install .` until package discovery is configured.

## Tests and automation

On commit `839e6930c02a82ef28b958a514eca7e799dbfbe5`, the test suite was run locally with `python -m pytest -q`: **16 passed**. This result covers the tests in `tests/test_robustnesstest.py`, including validation, synthetic smooth-matrix cases, benchmark output, and basic noise-sweep behavior; it is not a broad empirical study.

The GitHub Actions CI workflow is configured to run pytest and a compile/import smoke check on Python 3.10, 3.11, and 3.12. This README does not assert a remote workflow result. A separate CodeQL workflow is configured for Python analysis.

## Limitations

- The four benchmark methods use fixed defaults; only the individual compression functions expose parameter choices.
- Noise experiments use additive Gaussian noise on matrix entries. They do not cover other noise models, data distributions, trained networks, downstream tasks, or prediction stability.
- The tests use generated matrices, including synthetic smooth matrices. No measured benchmark dataset, comparative performance claim, or general research conclusion is provided.
- A single matrix and noise grid cannot establish which method is generally most robust. The report's ranking is specific to the input and the chosen first/last noise levels.
- The project currently lacks a working setuptools distribution build/install configuration and a CLI.

## License

The repository's `LICENSE` file contains the MIT License, with copyright notice `Copyright (c) 2026 Kapildhinwa2010-dev`. See [LICENSE](LICENSE) for the complete terms.
