"""High-level analyses and report-ready helper calculations."""

import numpy as np
import pandas as pd

from .demand import ROUTES
from .equilibrium import find_equilibrium
from .optimum import find_social_optimum
from .payoffs import (
    all_payoffs,
    all_revenues,
    gini_coefficient,
    passenger_wait_time,
    social_cost,
    total_revenue,
)


def gini(values):
    """Backward-compatible alias for :func:`src.payoffs.gini_coefficient`."""
    return gini_coefficient(values)


def run_full_analysis(
    D,
    route_info,
    n_drivers,
    lptrp_profile,
    alpha=0.7,
    W_norm_base=None,
    V_norm_base=None,
    fuel_cost_multiplier=1.0,
):
    """Compute Nash, LPTRP, and social-optimum comparison metrics."""
    n_routes = D.shape[1]
    if len(lptrp_profile) != n_drivers:
        raise ValueError("lptrp_profile length must equal n_drivers")

    rng = np.random.default_rng(42)
    initial = rng.integers(0, n_routes, size=n_drivers)
    equilibrium, _ = find_equilibrium(
        initial,
        D,
        route_info,
        n_routes=n_routes,
        fuel_cost_multiplier=fuel_cost_multiplier,
    )
    optimum, _ = find_social_optimum(
        D,
        route_info,
        n_drivers,
        alpha,
        1.0,
        1.0,
        fuel_cost_multiplier=fuel_cost_multiplier,
    )

    configs = {"nash": equilibrium, "lptrp": lptrp_profile, "optimum": optimum}
    wait_times = {key: passenger_wait_time(value, D, route_info) for key, value in configs.items()}
    payoff_arrays = {
        key: all_payoffs(value, D, route_info, fuel_cost_multiplier=fuel_cost_multiplier)
        for key, value in configs.items()
    }
    income_variances = {key: float(np.var(value)) for key, value in payoff_arrays.items()}
    ginis = {key: gini_coefficient(value) for key, value in payoff_arrays.items()}
    total_revenues = {
        key: total_revenue(value, D, route_info, fuel_cost_multiplier=fuel_cost_multiplier)
        for key, value in configs.items()
    }

    W_norm = W_norm_base if W_norm_base is not None else max(wait_times.values(), default=1.0)
    V_norm = (
        V_norm_base if V_norm_base is not None else max(income_variances.values(), default=1.0)
    )
    W_norm = W_norm if W_norm > 0 else 1.0
    V_norm = V_norm if V_norm > 0 else 1.0
    social_costs = {
        key: social_cost(
            value,
            D,
            route_info,
            alpha,
            W_norm,
            V_norm,
            fuel_cost_multiplier,
        )
        for key, value in configs.items()
    }

    optimum_cost = social_costs["optimum"]
    price_of_anarchy = social_costs["nash"] / optimum_cost if optimum_cost > 0 else float("inf")
    gap = social_costs["nash"] - optimum_cost
    lptrp_improvement_ratio = (
        (social_costs["nash"] - social_costs["lptrp"]) / gap if gap > 0 else 0.0
    )

    return {
        "nash_profile": equilibrium,
        "optimum_profile": optimum,
        "lptrp_profile": lptrp_profile,
        "social_costs": social_costs,
        "wait_times": wait_times,
        "income_variances": income_variances,
        "ginis": ginis,
        "total_revenues": total_revenues,
        "price_of_anarchy": price_of_anarchy,
        "lptrp_improvement_ratio": lptrp_improvement_ratio,
    }


def bottleneck_analysis(D, route_info, n_drivers, base_profile, alpha=0.7, fuel_cost_multiplier=1.0):
    """Measure the welfare change from reallocating one driver to each route.

    For every route, a driver is taken from the most-populated *other* route.
    A positive value means that the reallocation lowers social cost and that
    the target route is a bottleneck under the baseline allocation.
    """
    del n_drivers  # retained for the public API used by the pipeline
    base_cost = social_cost(
        base_profile,
        D,
        route_info,
        alpha,
        fuel_cost_multiplier=fuel_cost_multiplier,
    )
    counts = np.bincount(base_profile, minlength=D.shape[1])
    marginals = {}
    for route_idx in range(D.shape[1]):
        possible_donors = [idx for idx in range(D.shape[1]) if idx != route_idx and counts[idx] > 0]
        if not possible_donors:
            marginals[route_idx] = 0.0
            continue
        donor_route = max(possible_donors, key=lambda idx: counts[idx])
        modified = base_profile.copy()
        driver_idx = np.where(base_profile == donor_route)[0][0]
        modified[driver_idx] = route_idx
        new_cost = social_cost(
            modified,
            D,
            route_info,
            alpha,
            fuel_cost_multiplier=fuel_cost_multiplier,
        )
        marginals[route_idx] = float(base_cost - new_cost)
    return marginals


def validate_synthetic_example():
    """Validate best-response dynamics with the handbook's 3-driver game."""
    demand = np.zeros((1, 2, 1))
    demand[0, 0, 0] = 90
    demand[0, 1, 0] = 30
    route_info = {
        "length_km": [10.0, 8.0],
        "cycle_time_min": [60, 45],
        "served_nodes": [[0], [0]],
    }
    equilibrium, log = find_equilibrium(
        np.array([0, 0, 0]), demand, route_info, n_routes=2
    )
    counts = np.bincount(equilibrium, minlength=2)
    passed = (counts[0], counts[1]) in {(2, 1), (1, 2)}
    return {
        "test": "3-driver-2-route-synthetic",
        "expected_A": 2,
        "expected_B": 1,
        "actual_A": int(counts[0]),
        "actual_B": int(counts[1]),
        "passed": passed,
        "iterations": len(log),
    }


def compute_gap_analysis(results, route_names=None):
    """Return per-route LPTRP movement relative to Nash and the optimum."""
    n_routes = len(route_names) if route_names is not None else len(ROUTES)
    route_names = route_names or ROUTES[:n_routes]
    nash = np.bincount(results["nash_profile"], minlength=n_routes)
    lptrp = np.bincount(results["lptrp_profile"], minlength=n_routes)
    optimum = np.bincount(results["optimum_profile"], minlength=n_routes)
    rows = []
    for route_idx, route_name in enumerate(route_names):
        gap_total = abs(int(nash[route_idx]) - int(optimum[route_idx]))
        gap_closed = abs(int(lptrp[route_idx]) - int(nash[route_idx]))
        rows.append(
            {
                "route": route_name,
                "nash_drivers": int(nash[route_idx]),
                "lptrp_drivers": int(lptrp[route_idx]),
                "optimum_drivers": int(optimum[route_idx]),
                "allocation_gap_total": gap_total,
                "gap_closed_by_lptrp": gap_closed,
                "fraction_closed": round(gap_closed / gap_total, 3) if gap_total else None,
            }
        )
    return pd.DataFrame(rows)


def revenue_by_route(profile, D, route_info):
    """Return gross daily revenue totals keyed by route index."""
    profile = np.asarray(profile)
    revenues = all_revenues(profile, D, route_info)
    return {
        route_idx: float(revenues[profile == route_idx].sum())
        for route_idx in range(D.shape[1])
    }
