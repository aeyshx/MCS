import json
import numpy as np
from .demand import ROUTES


def load_lptrp_profile(filepath, n_drivers):
    '''
    Load the LPTRP authorized allocation from a JSON file and return it as
    a driver-assignment array of length n_drivers.

    Each element is the route index (into ROUTES from demand.py) of one driver.
    The profile is truncated or padded with route 0 to match n_drivers exactly.
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)

    assignments = data.get('route_assignments', {})

    profile = []
    for r_idx, r in enumerate(ROUTES):
        count = assignments.get(r, 0)
        profile.extend([r_idx] * count)

    if len(profile) > n_drivers:
        profile = profile[:n_drivers]
    elif len(profile) < n_drivers:
        profile.extend([0] * (n_drivers - len(profile)))

    return np.array(profile)
