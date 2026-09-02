import numpy as np
import pytest


@pytest.fixture
def synthetic_D():
    """Minimal one-node, two-route, one-window demand tensor."""
    demand = np.zeros((1, 2, 1))
    demand[0, 0, 0] = 90
    demand[0, 1, 0] = 30
    return demand


@pytest.fixture
def synthetic_route_info():
    return {
        "length_km": [10.0, 8.0],
        "cycle_time_min": [60, 45],
        "served_nodes": [[0], [0]],
    }
