"""
tests/test_payoffs.py
=====================
Unit tests for the payoff and social-cost functions in src/payoffs.py.
These tests use a tiny synthetic example (3 drivers, 2 routes) with a
known analytical solution so we can verify the functions precisely.
"""

import numpy as np
import pytest
from src.payoffs import (
    compute_fare,
    route_driver_counts,
    driver_payoff,
    all_payoffs,
    passenger_wait_time,
    social_cost,
)

# ---------------------------------------------------------------------------
# Minimal synthetic fixture
# ---------------------------------------------------------------------------
# 3 drivers, 2 routes
# Route 0: 10 km, 48 min cycle, serves node 0
# Route 1: 20 km, 96 min cycle, serves node 1
ROUTE_INFO_MINI = {
    'length_km':      [2.0, 4.0],   # short routes so fuel cost < fare revenue
    'cycle_time_min': [10,  20],
    'served_nodes':   [[0], [1]],
}

# Demand: 2 nodes x 2 routes x 3 windows
# Only node 0 / route 0 / window 0 has demand = 60 pax
D_MINI = np.zeros((2, 2, 3))
D_MINI[0, 0, 0] = 60.0


# ---------------------------------------------------------------------------
# compute_fare
# ---------------------------------------------------------------------------
class TestComputeFare:
    def test_minimum_distance_returns_base_fare(self):
        """Trip < 4 km should return exactly the base fares blended."""
        fare = compute_fare(2.0, discount_ratio=0.0)
        assert fare == pytest.approx(12.0), "No-discount fare for ≤4 km should be 12.00"

    def test_discount_blends_correctly(self):
        """100% discount passengers should pay 9.60 for short trip."""
        fare = compute_fare(2.0, discount_ratio=1.0)
        assert fare == pytest.approx(9.60)

    def test_extra_km_increases_fare(self):
        """Fare should increase for trips beyond 4 km."""
        fare_4km = compute_fare(4.0, discount_ratio=0.0)
        fare_8km = compute_fare(8.0, discount_ratio=0.0)
        assert fare_8km > fare_4km

    def test_extra_km_rate(self):
        """For 6 km trip (2 km extra) at 0% discount: 12 + 2*1.80 = 15.60"""
        fare = compute_fare(6.0, discount_ratio=0.0)
        assert fare == pytest.approx(15.60)

    def test_mixed_discount_6km(self):
        """20% discount, 6 km: 0.8*15.60 + 0.2*(9.60+2*1.44) = 14.976"""
        expected = 0.8 * 15.60 + 0.2 * (9.60 + 2 * 1.44)
        assert compute_fare(6.0, discount_ratio=0.2) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# route_driver_counts
# ---------------------------------------------------------------------------
class TestRouteDriverCounts:
    def test_counts_match_assignments(self):
        profile = np.array([0, 0, 1])   # 2 on route 0, 1 on route 1
        counts = route_driver_counts(profile, n_routes=2)
        assert counts[0] == 2
        assert counts[1] == 1

    def test_zero_count_for_unserved_route(self):
        profile = np.array([0, 0, 0])
        counts = route_driver_counts(profile, n_routes=2)
        assert counts[1] == 0


# ---------------------------------------------------------------------------
# driver_payoff
# ---------------------------------------------------------------------------
class TestDriverPayoff:
    def test_positive_payoff_on_demand_route(self):
        """A driver on route 0 (where demand exists) should have positive payoff."""
        profile = np.array([0, 0, 1])  # drivers 0,1 on route 0; driver 2 on route 1
        pi = driver_payoff(0, profile, D_MINI, ROUTE_INFO_MINI)
        assert pi > 0, "Driver on route with demand should earn positive income"

    def test_zero_payoff_on_empty_demand_route(self):
        """A driver on route 1 (no demand) should earn ≤ 0 due to fuel cost."""
        profile = np.array([0, 1, 1])
        pi = driver_payoff(1, profile, D_MINI, ROUTE_INFO_MINI)
        assert pi <= 0, "Driver on demand-less route should not profit"

    def test_payoff_decreases_with_more_competitors(self):
        """Adding a competitor on the same route should reduce individual payoff."""
        profile_solo  = np.array([0, 1, 1])   # driver 0 alone on route 0
        profile_compete = np.array([0, 0, 1]) # driver 0 shares route 0
        pi_solo     = driver_payoff(0, profile_solo,    D_MINI, ROUTE_INFO_MINI)
        pi_compete  = driver_payoff(0, profile_compete, D_MINI, ROUTE_INFO_MINI)
        assert pi_solo > pi_compete, "More competitors should reduce individual payoff"


# ---------------------------------------------------------------------------
# passenger_wait_time
# ---------------------------------------------------------------------------
class TestPassengerWaitTime:
    def test_more_drivers_reduces_wait(self):
        """Doubling drivers on a route should halve expected headway and wait."""
        profile_1 = np.array([0, 1, 1])   # 1 driver on route 0
        profile_2 = np.array([0, 0, 1])   # 2 drivers on route 0
        wt_1 = passenger_wait_time(profile_1, D_MINI, ROUTE_INFO_MINI)
        wt_2 = passenger_wait_time(profile_2, D_MINI, ROUTE_INFO_MINI)
        assert wt_1 > wt_2, "More drivers should reduce total wait time"

    def test_unserved_route_incurs_penalty(self):
        """If no driver serves a route with demand, wait time should spike."""
        profile_served   = np.array([0, 1, 1])  # route 0 is served
        profile_unserved = np.array([1, 1, 1])  # nobody on route 0
        wt_served   = passenger_wait_time(profile_served,   D_MINI, ROUTE_INFO_MINI)
        wt_unserved = passenger_wait_time(profile_unserved, D_MINI, ROUTE_INFO_MINI)
        assert wt_unserved > wt_served, "Unserved route should hugely increase wait time"

    def test_zero_demand_zero_wait(self):
        """An all-zero demand tensor should produce zero total wait time."""
        D_empty = np.zeros_like(D_MINI)
        profile = np.array([0, 1, 1])
        wt = passenger_wait_time(profile, D_empty, ROUTE_INFO_MINI)
        assert wt == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# social_cost
# ---------------------------------------------------------------------------
class TestSocialCost:
    def test_social_cost_positive(self):
        profile = np.array([0, 0, 1])
        sc = social_cost(profile, D_MINI, ROUTE_INFO_MINI, alpha=0.7)
        assert sc >= 0

    def test_alpha_zero_ignores_wait_time(self):
        """With alpha=0, social cost equals income variance (normalized)."""
        profile = np.array([0, 0, 1])
        payoffs = all_payoffs(profile, D_MINI, ROUTE_INFO_MINI)
        expected_V = np.var(payoffs)
        sc = social_cost(profile, D_MINI, ROUTE_INFO_MINI, alpha=0.0,
                         W_norm=1.0, V_norm=1.0)
        assert sc == pytest.approx(expected_V, rel=1e-5)

    def test_alpha_one_ignores_income_variance(self):
        """With alpha=1, social cost equals normalized wait time only."""
        profile = np.array([0, 0, 1])
        wt = passenger_wait_time(profile, D_MINI, ROUTE_INFO_MINI)
        sc = social_cost(profile, D_MINI, ROUTE_INFO_MINI, alpha=1.0,
                         W_norm=1.0, V_norm=1.0)
        assert sc == pytest.approx(wt, rel=1e-5)
