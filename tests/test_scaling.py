import json

import pytest

from scaling import SIZES, fit_scaling_law, iterations_for, main, optimal_parameters, predict_loss


def test_iterations_shrink_as_the_model_grows():
    assert iterations_for(1e9, 1000, 10, 10) == 10_000
    assert iterations_for(1e9, 2000, 10, 10) == 5_000
    assert iterations_for(1.0, 1000, 10, 10) == 1


def test_fit_recovers_a_known_formula():
    true = {"E": 1.2, "A": 30.0, "alpha": 0.5, "B": 400.0, "beta": 0.4}
    points = [
        (n, d, predict_loss(true, n, d))
        for n in (1e4, 3e4, 1e5, 3e5)
        for d in (1e6, 3e6, 1e7, 3e7)
    ]
    fit = fit_scaling_law(points)
    assert fit["alpha"] == pytest.approx(0.5)
    assert fit["beta"] == pytest.approx(0.4)
    assert fit["E"] == pytest.approx(1.2, abs=1e-3)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_fit_rejects_data_with_no_valid_solution():
    # Loss that gets worse with more data needs a negative coefficient.
    points = [(1e4, d, 1 - 1 / d**0.5) for d in (1e2, 1e3, 1e4, 1e5, 1e6)]
    points += [(1e5, d, 1 - 1 / d**0.5) for d in (1e2, 1e3, 1e4)]
    with pytest.raises(ValueError, match="no valid fit"):
        fit_scaling_law(points)


def test_optimal_size_balances_model_and_data():
    # With equal terms, the best split is equal: N = D = sqrt(budget).
    fit = {"E": 1.0, "A": 10.0, "alpha": 0.5, "B": 10.0, "beta": 0.5}
    assert optimal_parameters(fit, 1e10) == pytest.approx(1e5, rel=0.03)


def test_optimal_size_grows_with_budget():
    fit = {"E": 1.0, "A": 30.0, "alpha": 0.4, "B": 200.0, "beta": 0.3}
    assert optimal_parameters(fit, 1e12) > optimal_parameters(fit, 1e11)


def test_sizes_are_in_increasing_order():
    assert SIZES == sorted(SIZES, key=lambda s: s[0] * s[2])


def test_end_to_end_small_budget(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.txt").write_text("ANNA: hello there\nBEN: hi\n\n" * 200)
    (data / "val.txt").write_text("BEN: hello\nANNA: hi there\n\n" * 50)
    out = tmp_path / "out"
    try:
        main(["--data", str(data), "--out", str(out), "--budget-scale", "1e-5"])
    except ValueError as error:
        # Very short runs can be too noisy to fit, but every run must still be saved.
        assert "no valid fit" in str(error)
    assert len(json.loads((out / "results.json").read_text())) == len(SIZES) * 3
