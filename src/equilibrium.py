import numpy as np
from .payoffs import driver_payoff
from .demand import ROUTES

def best_response(
    driver_idx, profile, D, route_info, n_routes=None, fuel_cost_multiplier=1.0
):
    '''Find the route that maximizes this driver's payoff, others fixed.'''
    if n_routes is None:
        n_routes = len(ROUTES)
    best_route = profile[driver_idx]
    best_pi = driver_payoff(
        driver_idx, profile, D, route_info, fuel_cost_multiplier=fuel_cost_multiplier
    )
    original = profile[driver_idx]
    for r in range(n_routes):
        if r == original:
            continue
        profile[driver_idx] = r
        pi = driver_payoff(
            driver_idx, profile, D, route_info, fuel_cost_multiplier=fuel_cost_multiplier
        )
        if pi > best_pi + 1e-6: # tolerance to avoid oscillation
            best_pi = pi
            best_route = r
    profile[driver_idx] = original # restore for caller
    return best_route, best_pi

def find_equilibrium(
    initial_profile, D, route_info,
    n_routes=None, max_iters=200, seed=42, verbose=False, fuel_cost_multiplier=1.0
):
    '''
    Run best-response dynamics until convergence.
    Returns final profile and iteration log.
    '''
    if n_routes is None:
        n_routes = len(ROUTES)
    rng = np.random.default_rng(seed)
    profile = initial_profile.copy()
    n = len(profile)
    log = []
    for it in range(max_iters):
        order = rng.permutation(n)
        changes = 0
        for i in order:
            br, _ = best_response(
                i, profile, D, route_info, n_routes, fuel_cost_multiplier
            )
            if br != profile[i]:
                profile[i] = br
                changes += 1
        log.append({'iter': it, 'changes': changes})
        if verbose:
            print(f'Iter {it}: {changes} changes')
        if changes == 0:
            break
    return profile, log

def find_equilibrium_multistart(
    D, route_info, n_drivers, n_starts=10, seed=42, n_routes=None,
    fuel_cost_multiplier=1.0,
):
    '''
    Run best-response dynamics from n_starts random initializations
    to check for multiple equilibria.
    '''
    if n_routes is None:
        n_routes = len(ROUTES)
    rng = np.random.default_rng(seed)
    equilibria = []
    for start in range(n_starts):
        initial = rng.integers(0, n_routes, size=n_drivers)
        eq, _ = find_equilibrium(
            initial,
            D,
            route_info,
            n_routes=n_routes,
            seed=seed + start,
            fuel_cost_multiplier=fuel_cost_multiplier,
        )
        # Preserve route order so results can be directly reported by route.
        counts = tuple(np.bincount(eq, minlength=n_routes).tolist())
        equilibria.append(counts)
    unique = set(equilibria)
    return equilibria, unique
