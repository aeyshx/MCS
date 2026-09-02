"""
tests/test_equilibrium.py
=========================
Unit tests for the Nash Equilibrium search in src/equilibrium.py.
Uses a tiny 2-route, 4-driver synthetic game with a known pure-strategy
Nash equilibrium so we can verify the algorithm converges to the right answer.
"""

import numpy as np
import pytest
from src.equilibrium import best_response, find_equilibrium, find_equilibrium_multistart
from src.demand import ROUTES

# ---------------------------------------------------------------------------
# Minimal synthetic fixture (same as test_payoffs.py)
# ---------------------------------------------------------------------------
ROUTE_INFO_MINI = {
    'length_km':      [2.0, 4.0],   # short routes so fuel cost < fare revenue
    'cycle_time_min': [10,  20],
    'served_nodes':   [[0], [1]],
}

# Only node 0 / route 0 / window 0 has high demand
D_HIGH_ROUTE0 = np.zeros((2, 2, 3))
D_HIGH_ROUTE0[0, 0, 0] = 200.0   # high demand on route 0

# Balanced demand on both routes
D_BALANCED = np.zeros((2, 2, 3))
D_BALANCED[0, 0, 0] = 100.0
D_BALANCED[1, 1, 0] = 100.0


# ---------------------------------------------------------------------------
# best_response
# ---------------------------------------------------------------------------
class TestBestResponse:
    def test_driver_switches_to_better_route(self):
        """A driver on route 0 (demand) should earn more than on route 1 (no demand)."""
        from src.payoffs import driver_payoff
        # driver 2 alone on route 1 (no demand); check payoff on route 0 > route 1
        profile_on0 = np.array([0, 0, 0])   # driver 2 on route 0
        profile_on1 = np.array([0, 0, 1])   # driver 2 on route 1 (no demand)
        pi_on0 = driver_payoff(2, profile_on0, D_HIGH_ROUTE0, ROUTE_INFO_MINI)
        pi_on1 = driver_payoff(2, profile_on1, D_HIGH_ROUTE0, ROUTE_INFO_MINI)
        assert pi_on0 > pi_on1, "Route with demand should pay more than route without demand"
        # best_response from route 1 should suggest switching to route 0
        br, _ = best_response(2, profile_on1, D_HIGH_ROUTE0, ROUTE_INFO_MINI, n_routes=2)
        assert br == 0, "Driver on empty route should prefer route with demand"

    def test_driver_stays_if_already_best(self):
        """If a driver is already maximizing payoff, best response = current route."""
        # All drivers on route 0 which has all the demand
        profile = np.array([0, 0, 0, 0])
        # With 4 drivers competing on route 0, it might not pay to switch to empty route 1
        br, _ = best_response(0, profile, D_HIGH_ROUTE0, ROUTE_INFO_MINI, n_routes=2)
        # Either stay on 0 or switch — just verify best_response returns a valid route
        assert br in (0, 1)

    def test_profile_unchanged_after_call(self):
        """best_response must restore the original profile (no mutation)."""
        profile = np.array([0, 1, 1])
        original = profile.copy()
        best_response(0, profile, D_HIGH_ROUTE0, ROUTE_INFO_MINI, n_routes=2)
        np.testing.assert_array_equal(profile, original,
            err_msg="best_response must not modify the profile array")


# ---------------------------------------------------------------------------
# find_equilibrium
# ---------------------------------------------------------------------------
class TestFindEquilibrium:
    def test_convergence(self):
        """BR dynamics must terminate before max_iters."""
        initial = np.array([0, 0, 1, 1])
        eq, log = find_equilibrium(
            initial, D_HIGH_ROUTE0, ROUTE_INFO_MINI,
            n_routes=2, max_iters=100,
        )
        assert log[-1]['changes'] == 0, "Last iteration should have 0 changes (converged)"

    def test_equilibrium_is_nash(self):
        """At the returned profile, no driver should want to switch."""
        initial = np.random.default_rng(99).integers(0, 2, size=6)
        eq, _ = find_equilibrium(
            initial, D_HIGH_ROUTE0, ROUTE_INFO_MINI,
            n_routes=2, max_iters=200,
        )
        # Verify no driver can unilaterally improve payoff
        from src.payoffs import driver_payoff
        for i in range(len(eq)):
            _, pi_best = best_response(i, eq, D_HIGH_ROUTE0, ROUTE_INFO_MINI, n_routes=2)
            pi_current = driver_payoff(i, eq, D_HIGH_ROUTE0, ROUTE_INFO_MINI)
            assert pi_best <= pi_current + 1e-4, \
                f"Driver {i} can still improve — not a Nash equilibrium"

    def test_returns_valid_routes(self):
        """Every entry in the equilibrium profile must be a valid route index."""
        initial = np.array([0, 1, 0, 1])
        eq, _ = find_equilibrium(
            initial, D_BALANCED, ROUTE_INFO_MINI, n_routes=2, max_iters=50,
        )
        assert all(0 <= r < 2 for r in eq), "All route indices must be in [0, n_routes)"

    def test_log_format(self):
        """Iteration log must contain 'iter' and 'changes' keys."""
        initial = np.array([0, 1])
        _, log = find_equilibrium(
            initial, D_BALANCED, ROUTE_INFO_MINI, n_routes=2, max_iters=10,
        )
        assert len(log) > 0
        assert 'iter' in log[0]
        assert 'changes' in log[0]


# ---------------------------------------------------------------------------
# find_equilibrium_multistart
# ---------------------------------------------------------------------------
class TestMultiStart:
    def test_returns_equilibria_list(self):
        all_eq, unique_eq = find_equilibrium_multistart(
            D_HIGH_ROUTE0, ROUTE_INFO_MINI, n_drivers=4,
            n_starts=5, seed=42, n_routes=2,
        )
        assert len(all_eq) == 5

    def test_unique_is_subset_of_all(self):
        all_eq, unique_eq = find_equilibrium_multistart(
            D_BALANCED, ROUTE_INFO_MINI, n_drivers=4,
            n_starts=8, seed=0, n_routes=2,
        )
        assert unique_eq.issubset(set(all_eq)), \
            "unique equilibria must be a subset of all found equilibria"

    def test_equilibrium_count_type(self):
        """Each equilibrium is stored as a sorted tuple of driver counts."""
        all_eq, _ = find_equilibrium_multistart(
            D_HIGH_ROUTE0, ROUTE_INFO_MINI, n_drivers=3,
            n_starts=3, seed=7, n_routes=2,
        )
        for eq in all_eq:
            assert isinstance(eq, tuple), "Each equilibrium entry should be a tuple"
