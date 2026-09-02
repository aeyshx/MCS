import numpy as np
from .payoffs import social_cost
from .demand import ROUTES

def find_social_optimum(
    D, route_info, n_drivers, alpha=0.7,
    W_norm=1.0, V_norm=1.0,
    n_iters=20000, T0=1.0, T_min=0.001, seed=42, fuel_cost_multiplier=1.0
):
    '''
    Simulated annealing search for the strategy profile that
    minimizes social cost.
    '''
    rng = np.random.default_rng(seed)
    n_routes = D.shape[1]
    # Sensible initial profile: distribute drivers proportionally to demand
    demand_per_route = D.sum(axis=(0, 2))
    total_demand = demand_per_route.sum()
    proportions = (
        demand_per_route / total_demand
        if total_demand > 0
        else np.full(n_routes, 1 / n_routes)
    )
    initial = rng.choice(n_routes, size=n_drivers, p=proportions)
    current = initial.copy()
    current_cost = social_cost(
        current, D, route_info, alpha, W_norm, V_norm, fuel_cost_multiplier
    )
    best = current.copy()
    best_cost = current_cost
    T = T0
    cooling = (T_min / T0) ** (1 / n_iters)
    for it in range(n_iters):
        # Propose a move: change one driver's route
        i = rng.integers(0, n_drivers)
        new_route = rng.integers(0, n_routes)
        while new_route == current[i]:
            new_route = rng.integers(0, n_routes)
        proposed = current.copy()
        proposed[i] = new_route
        proposed_cost = social_cost(
            proposed, D, route_info, alpha, W_norm, V_norm, fuel_cost_multiplier
        )
        delta = proposed_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = proposed
            current_cost = proposed_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
        T *= cooling
    return best, best_cost
