import numpy as np
import pytest

from robustnesstest import (
    add_noise,
    benchmark,
    fourier_compress,
    hybrid_compress,
    noise_sweep,
    quantize,
    robustness_report,
    svd_compress,
)


def _smooth(rows=16, cols=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    amps = rng.standard_normal((rows, 3)) * 0.5
    return amps[:, 0:1] + amps[:, 1:2] * np.sin(t) + amps[:, 2:3] * np.cos(2 * t)


def test_svd_low_rank_exact():
    A = _smooth()
    assert np.linalg.norm(A - svd_compress(A, rank=8)) < 1e-8


def test_svd_rejects_bad_rank():
    with pytest.raises(ValueError):
        svd_compress(np.eye(4), rank=0)


def test_quantize_error_bounded():
    A = _smooth()
    err = np.abs(A - quantize(A, bits=8)).max()
    span = A.max() - A.min()
    assert err <= span / 256 + 1e-12


def test_quantize_rejects_bad_bits():
    with pytest.raises(ValueError):
        quantize(np.ones((2, 2)), bits=0)


def test_fourier_exact_for_smooth():
    A = _smooth()
    err = np.linalg.norm(A - fourier_compress(A, num_coeffs=4)) / np.linalg.norm(A)
    assert err < 1e-6


def test_hybrid_reasonable():
    A = _smooth()
    err = np.linalg.norm(A - hybrid_compress(A, rank=8, bits=12)) / np.linalg.norm(A)
    assert err < 0.05


def test_benchmark_returns_all_methods():
    results = benchmark(_smooth())
    names = {r["method"] for r in results}
    assert names == {"svd_rank4", "quantize8", "fourier4", "hybrid"}


def test_benchmark_entries_have_error_and_pass():
    for entry in benchmark(_smooth()):
        assert 0.0 <= entry["relative_error"]
        assert isinstance(entry["passed"], bool)


def test_benchmark_rejects_1d():
    with pytest.raises(ValueError):
        benchmark(np.ones(5))


def test_benchmark_low_error_for_smooth():
    results = benchmark(_smooth(rows=32, cols=128))
    assert all(r["relative_error"] < 0.5 for r in results)


# --- noise-robustness (RobustnessTest-specific) ---


def test_add_noise_zero_is_copy():
    A = _smooth()
    noisy = add_noise(A, 0.0, np.random.default_rng(0))
    assert np.array_equal(noisy, A)


def test_add_noise_rejects_negative_sigma():
    with pytest.raises(ValueError):
        add_noise(np.eye(4), -0.1, np.random.default_rng(0))


def test_noise_sweep_zero_sigma_matches_benchmark():
    A = _smooth()
    sweep = noise_sweep(A, [0.0])
    bench = {r["method"]: r["relative_error"] for r in benchmark(A)}
    for name, rows in sweep.items():
        assert rows[0]["relative_error"] == pytest.approx(bench[name])


def test_svd_error_grows_with_noise():
    A = _smooth()
    sweep = noise_sweep(A, [0.0, 0.05, 0.2])
    errs = [r["relative_error"] for r in sweep["svd_rank4"]]
    assert errs[2] > errs[0]


def test_noise_sweep_rejects_empty():
    with pytest.raises(ValueError):
        noise_sweep(_smooth(), [])


def test_robustness_report_has_most_robust():
    report = robustness_report(_smooth())
    assert report["most_robust"] in {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    for name, entry in report["methods"].items():
        assert entry["degradation"] >= -1e-12
        assert entry["baseline_error"] >= 0.0