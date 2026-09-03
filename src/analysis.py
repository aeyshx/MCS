import numpy as np
from .equilibrium import find_equilibrium
from .optimum import find_social_optimum
from .payoffs import social_cost, passenger_wait_time, all_payoffs
from .demand import ROUTES

def run_full_analysis(D, route_info, n_drivers, lptrp_profile, alpha=0.7):
    '''
    Compute Nash equilibrium, social optimum, and LPTRP,
    and return all comparison metrics.
    '''
    # First pass: compute unnormalized values for each configuration
    rng = np.random.default_rng(42)
    initial = rng.integers(0, len(ROUTES), size=n_drivers)
    eq, _= find_equilibrium(initial, D, route_info)
    opt, _= find_social_optimum(D, route_info, n_drivers, alpha, 1.0, 1.0)
    
    # Compute normalizers from raw values across three configurations
    configs = {'nash': eq, 'lptrp': lptrp_profile, 'optimum': opt}
    W_values = {k: passenger_wait_time(v, D, route_info) for k, v in configs.items()}
    V_values = {k: np.var(all_payoffs(v, D, route_info)) for k, v in configs.items()}
    W_norm = max(W_values.values())
    V_norm = max(V_values.values())
    
    # Normalized social costs
    costs = {k: social_cost(v, D, route_info, alpha, W_norm, V_norm) for k, v in configs.items()}
    poa = costs['nash'] / costs['optimum']
    if costs['nash'] > costs['optimum']:
        improvement = (costs['nash'] - costs['lptrp']) / (costs['nash'] - costs['optimum'])
    else:
        improvement = 0
        
    return {
        'nash_profile': eq,
        'optimum_profile': opt,
        'lptrp_profile': lptrp_profile,
        'social_costs': costs,
        'wait_times': W_values,
        'income_variances': V_values,
        'price_of_anarchy': poa,
        'lptrp_improvement_ratio': improvement,
    }

def bottleneck_analysis(D, route_info, n_drivers, base_profile, alpha=0.7):
    '''
    For each route, compute the marginal welfare gain from adding one driver
    and the marginal loss from removing one driver.
    '''
    base_cost = social_cost(base_profile, D, route_info, alpha)
    marginals = {}
    for r in range(len(ROUTES)):
        # Add one driver to route r (steal from the largest route)
        largest = np.bincount(base_profile, minlength=len(ROUTES)).argmax()
        if largest == r:
            marginals[r] = 0
            continue
        modified = base_profile.copy()
        # Find a driver on the largest route
        candidates = np.where(base_profile == largest)[0]
        modified[candidates[0]] = r
        new_cost = social_cost(modified, D, route_info, alpha)
        marginals[r] = base_cost - new_cost # positive if adding to r helps
    return marginals
