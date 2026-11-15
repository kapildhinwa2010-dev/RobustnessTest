# RobustnessTest - How compression error degrades under additive noise
# Part of the Kapildhinwa2010-dev Toolkit.

"""Noise-robustness suite for the four toolkit compression strategies.

Lineage: all toolkit repos. Beyond the plain benchmark (best-case error on
clean matrices), this module sweeps additive Gaussian noise levels and
reports how each method's reconstruction error degrades. NOTE: this is a
matrix-level study; it does not measure robustness of a trained network's
predictions, and results on random matrices may not transfer to real
weights (see README limitations).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "svd_compress",
    "quantize",
    "fourier_compress",
    "hybrid_compress",
    "benchmark",
    "add_noise",
    "noise_sweep",
    "robustness_report",
]


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    t = np.linspace(0.0, 2.0 * np.pi, matrix.shape[1], endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.cos(k * t))
        cols.append(np.sin(k * t))
    basis = np.stack(cols, axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def _methods():
    return {
        "svd_rank4": lambda m: svd_compress(m, rank=4),
        "quantize8": lambda m: quantize(m, bits=8),
        "fourier4": lambda m: fourier_compress(m, num_coeffs=4),
        "hybrid": lambda m: hybrid_compress(m, rank=4, bits=8),
    }


def benchmark(matrix: np.ndarray) -> list:
    """Run all methods on the CLEAN matrix; error vs the matrix itself."""
    matrix = _check(matrix)
    results = []
    norm = float(np.linalg.norm(matrix))
    for name, fn in _methods().items():
        approx = fn(matrix)
        rel_error = float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0
        results.append({
            "method": name,
            "relative_error": rel_error,
            "passed": rel_error < 0.5,
        })
    return results


def add_noise(matrix: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise with standard deviation ``sigma``."""
    matrix = _check(matrix)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return matrix.copy()
    return matrix + rng.normal(0.0, sigma, size=matrix.shape)


def noise_sweep(matrix: np.ndarray, sigmas, seed: int = 0) -> dict:
    """Error of each method on noisy inputs, measured against the CLEAN matrix.

    Returns {method: [{"sigma": s, "relative_error": e}, ...]} in the given
    sigma order. A method that stays flat as sigma grows is noise-robust;
    one whose error tracks sigma is not.
    """
    matrix = _check(matrix)
    sigmas = [float(s) for s in sigmas]
    if not sigmas:
        raise ValueError("sigmas must be non-empty")
    if any(s < 0 for s in sigmas):
        raise ValueError("sigmas must be >= 0")
    rng = np.random.default_rng(seed)
    norm = float(np.linalg.norm(matrix))
    sweep = {}
    for name, fn in _methods().items():
        rows = []
        for sigma in sigmas:
            noisy = add_noise(matrix, sigma, rng)
            approx = fn(noisy)
            rel = float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0
            rows.append({"sigma": sigma, "relative_error": rel})
        sweep[name] = rows
    return sweep


def robustness_report(matrix: np.ndarray,
                      sigmas=(0.0, 0.01, 0.05, 0.1)) -> dict:
    """Summarize the sweep: baseline error, degradation, most robust method.

    "degradation" is relative_error at the largest sigma minus the error at
    sigma 0. Honest caveat: ranking depends on the sigma grid and the matrix
    structure; do not generalize a single matrix's ranking to all weights.
    """
    sweep = noise_sweep(matrix, sigmas)
    summary = {}
    for name, rows in sweep.items():
        baseline = rows[0]["relative_error"]
        worst = rows[-1]["relative_error"]
        summary[name] = {
            "baseline_error": baseline,
            "error_at_max_sigma": worst,
            "degradation": worst - baseline,
        }
    most_robust = min(summary, key=lambda n: summary[n]["degradation"])
    return {"methods": summary, "most_robust": most_robust, "sweep": sweep}