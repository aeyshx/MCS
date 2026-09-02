import numpy as np

from src.analysis import bottleneck_analysis, compute_gap_analysis, validate_synthetic_example
from src.equilibrium import find_equilibrium_multistart


def test_synthetic_validation_reproduces_handbook_equilibrium():
    validation = validate_synthetic_example()
    assert validation["passed"] is True
    assert (validation["actual_A"], validation["actual_B"]) == (2, 1)


def test_bottleneck_analysis_returns_one_signed_result_per_route(synthetic_D, synthetic_route_info):
    profile = np.array([0, 0, 1])
    marginal = bottleneck_analysis(synthetic_D, synthetic_route_info, 3, profile)
    assert set(marginal) == {0, 1}
    assert all(isinstance(value, float) for value in marginal.values())


def test_multistart_preserves_route_order(synthetic_D, synthetic_route_info):
    equilibria, _ = find_equilibrium_multistart(
        synthetic_D, synthetic_route_info, n_drivers=3, n_starts=3, n_routes=2
    )
    assert all(len(counts) == 2 and sum(counts) == 3 for counts in equilibria)


def test_gap_analysis_has_all_route_metrics():
    results = {
        "nash_profile": np.array([0, 0, 1]),
        "lptrp_profile": np.array([0, 1, 1]),
        "optimum_profile": np.array([0, 1, 1]),
    }
    gap = compute_gap_analysis(results, ["A", "B"])
    assert list(gap["route"]) == ["A", "B"]
    assert set(gap.columns) >= {"allocation_gap_total", "gap_closed_by_lptrp", "fraction_closed"}
