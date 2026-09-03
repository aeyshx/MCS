"""Payoff, revenue, and welfare functions for the jeepney congestion game."""

import numpy as np

from .demand import ROUTES

# Constants derived from the field-data assumptions.
DIESEL_PRICE = 92.66
FUEL_EFFICIENCY_KM_PER_LITER = 5.5
FUEL_COST_PER_KM_PHP = DIESEL_PRICE / FUEL_EFFICIENCY_KM_PER_LITER


def compute_fare(distance_km, discount_ratio=0.20):
    """Return the average PHP fare for a passenger travelling ``distance_km``."""
    extra_km = max(0, distance_km - 4)
    regular_fare = 12.00 + (extra_km * 1.80)
    discount_fare = 9.60 + (extra_km * 1.44)
    return (1 - discount_ratio) * regular_fare + discount_ratio * discount_fare


def route_driver_counts(profile, n_routes=None):
    """Count drivers assigned to each route."""
    if n_routes is None:
        n_routes = len(ROUTES)
    return np.bincount(profile, minlength=n_routes)


def _route_financials(D, route_info, fuel_cost_multiplier=1.0, operating_hours=12):
    """Return total route revenue and per-driver route cost before revenue sharing."""
    n_routes = D.shape[1]
    gross_route_revenue = np.zeros(n_routes, dtype=float)
    driver_cost = np.zeros(n_routes, dtype=float)
    observed_hours = D.shape[2] * 2.0
    extrapolation_factor = operating_hours / observed_hours if observed_hours else 0.0

    for route_idx in range(n_routes):
        length_km = route_info["length_km"][route_idx]
        cycle_min = route_info["cycle_time_min"][route_idx]
        served_nodes = route_info["served_nodes"][route_idx]
        if cycle_min <= 0:
            raise ValueError(f"Route {route_idx} has a non-positive cycle time")

        # Passengers on average ride half the route length (boarding at midpoint
        # of a one-way trip or riding one direction of a return service).
        fare_per_pax = compute_fare(length_km / 2.0, discount_ratio=0.20)
        loops_per_window = 120 / cycle_min
        route_demand = (
            D[served_nodes, route_idx, :].sum(axis=0)
            if served_nodes
            else np.zeros(D.shape[2])
        )
        gross_route_revenue[route_idx] = (
            fare_per_pax * np.sum(route_demand * loops_per_window) * extrapolation_factor
        )
        daily_loops = D.shape[2] * loops_per_window
        driver_cost[route_idx] = (
            FUEL_COST_PER_KM_PHP * fuel_cost_multiplier * length_km * daily_loops * extrapolation_factor
        )

    return gross_route_revenue, driver_cost


def driver_payoff(driver_idx, profile, D, route_info, operating_hours=12, fuel_cost_multiplier=1.0):
    """Expected daily net income (PHP) for one driver."""
    route_idx = profile[driver_idx]
    counts = route_driver_counts(profile, n_routes=D.shape[1])
    gross_route_revenue, driver_cost = _route_financials(
        D, route_info, fuel_cost_multiplier, operating_hours
    )
    return gross_route_revenue[route_idx] / counts[route_idx] - driver_cost[route_idx]


def driver_revenue(driver_idx, profile, D, route_info, operating_hours=12):
    """Expected daily gross revenue (before fuel costs) for one driver."""
    route_idx = profile[driver_idx]
    counts = route_driver_counts(profile, n_routes=D.shape[1])
    gross_route_revenue, _ = _route_financials(D, route_info, operating_hours=operating_hours)
    return gross_route_revenue[route_idx] / counts[route_idx]


def all_payoffs_vectorized(profile, D, route_info, operating_hours=12, fuel_cost_multiplier=1.0):
    """Compute net daily incomes for all drivers with one route-count pass."""
    profile = np.asarray(profile, dtype=int)
    if len(profile) == 0:
        return np.array([], dtype=float)
    counts = route_driver_counts(profile, n_routes=D.shape[1])
    gross_route_revenue, driver_cost = _route_financials(
        D, route_info, fuel_cost_multiplier, operating_hours
    )
    per_driver_revenue = np.divide(
        gross_route_revenue,
        counts,
        out=np.zeros_like(gross_route_revenue),
        where=counts > 0,
    )
    return per_driver_revenue[profile] - driver_cost[profile]


def all_payoffs(profile, D, route_info, operating_hours=12, fuel_cost_multiplier=1.0):
    """Return vectorized net daily incomes for all drivers."""
    return all_payoffs_vectorized(profile, D, route_info, operating_hours, fuel_cost_multiplier)


def all_revenues(profile, D, route_info, operating_hours=12):
    """Return vectorized gross daily revenues for all drivers."""
    profile = np.asarray(profile, dtype=int)
    if len(profile) == 0:
        return np.array([], dtype=float)
    counts = route_driver_counts(profile, n_routes=D.shape[1])
    gross_route_revenue, _ = _route_financials(D, route_info, operating_hours=operating_hours)
    per_driver_revenue = np.divide(
        gross_route_revenue,
        counts,
        out=np.zeros_like(gross_route_revenue),
        where=counts > 0,
    )
    return per_driver_revenue[profile]


def gini_coefficient(incomes):
    """Return the Gini coefficient of a one-dimensional income array."""
    incomes = np.sort(np.abs(np.asarray(incomes, dtype=float)))
    n = len(incomes)
    total = incomes.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * incomes) / (n * total)) - (n + 1) / n


def total_revenue(profile, D, route_info, operating_hours=12, fuel_cost_multiplier=1.0):
    """Return total net operator income (the sum of all driver payoffs)."""
    return float(
        all_payoffs(profile, D, route_info, operating_hours, fuel_cost_multiplier).sum()
    )


def passenger_wait_time_by_window(profile, D, route_info):
    """Return total passenger-minutes waiting for each observed time window."""
    counts = route_driver_counts(profile, n_routes=D.shape[1])
    wait_per_window = []
    for time_idx in range(D.shape[2]):
        total = 0.0
        for route_idx in range(D.shape[1]):
            served_nodes = route_info["served_nodes"][route_idx]
            demand = D[served_nodes, route_idx, time_idx].sum() if served_nodes else 0.0
            if counts[route_idx] == 0:
                total += demand * 60  # finite penalty for an unserved route
            else:
                total += demand * (route_info["cycle_time_min"][route_idx] / counts[route_idx] / 2)
        wait_per_window.append(float(total))
    return wait_per_window


def passenger_wait_time(profile, D, route_info):
    """Return total passenger-minutes of waiting across the observed day."""
    return float(sum(passenger_wait_time_by_window(profile, D, route_info)))


def social_cost(
    profile,
    D,
    route_info,
    alpha=0.7,
    W_norm=1.0,
    V_norm=1.0,
    fuel_cost_multiplier=1.0,
):
    """Return weighted, normalized passenger-wait and income-variance cost."""
    wait_normalizer = W_norm if W_norm > 0 else 1.0
    variance_normalizer = V_norm if V_norm > 0 else 1.0
    wait_component = passenger_wait_time(profile, D, route_info) / wait_normalizer
    payoffs = all_payoffs(profile, D, route_info, fuel_cost_multiplier=fuel_cost_multiplier)
    variance_component = np.var(payoffs) / variance_normalizer
    return alpha * wait_component + (1 - alpha) * variance_component
