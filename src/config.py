"""Study-wide definitions and default model assumptions.

This module is the single source of truth for identifiers that appear in raw
data, numerical arrays, analysis tables, and figures.  Keep the ordering of
``NODES``, ``ROUTES``, and ``WINDOWS`` stable: array axes use these positions.
"""

from __future__ import annotations

from typing import Final


NODES: Final[tuple[str, ...]] = (
    "ROTUNDA",
    "PACIFIC MALL",
    "DARAGA MARKET",
    "EMBARCADERO",
    "SM",
)

ROUTES: Final[tuple[str, ...]] = (
    "DIRETSO A",
    "DIRETSO B",
    "RAWIS A",
    "RAWIS B",
    "ARIMBAY",
    "LOOP 1",
    "LOOP 2",
    "EXTERNAL",
)

WINDOWS: Final[tuple[str, ...]] = ("AM", "MID", "PM")
WINDOW_DURATION_MIN: Final[dict[str, int]] = {window: 120 for window in WINDOWS}

# ``EXTERNAL`` is retained during data preparation but excluded from the
# seven-route Legazpi-Daraga allocation model and its result figures.
MODEL_ROUTE_INDICES: Final[tuple[int, ...]] = tuple(range(len(ROUTES) - 1))
MODEL_ROUTES: Final[tuple[str, ...]] = tuple(ROUTES[index] for index in MODEL_ROUTE_INDICES)

DEFAULT_DRIVER_COUNT: Final[int] = 188
DEFAULT_ALPHA: Final[float] = 0.7

# These assumptions are kept here so a future calibration can change them in
# one reviewable place. Their position corresponds exactly to ``ROUTES``.
DEFAULT_ROUTE_INFO: Final[dict[str, list[float] | list[list[int]]]] = {
    "length_km": [12.2, 12.9, 22.28, 21.2, 11.64, 12.77, 13.0, 30.0],
    "cycle_time_min": [59, 62, 107, 102, 56, 61, 62, 144],
    "served_nodes": [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [1, 4],
        [0, 2],
        [0, 2],
        [],
    ],
}
