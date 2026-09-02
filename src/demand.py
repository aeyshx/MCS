import pandas as pd
import numpy as np

# Canonical node and route names matching the real field-survey data.
# These must be kept in sync with src/data_parser.py.
NODES = [
    'ROTUNDA',       # 0 – Ayala / Rotunda area
    'PACIFIC MALL',  # 1
    'DARAGA MARKET', # 2
    'EMBARCADERO',   # 3
    'SM',            # 4
]

ROUTES = [
    'DIRETSO A',  # 0 – LPTRP route 11 (via Lapu-Lapu St)
    'DIRETSO B',  # 1 – LPTRP route 12 (via Capitol)
    'RAWIS A',    # 2 – LPTRP route 13
    'RAWIS B',    # 3 – LPTRP route 14
    'ARIMBAY',    # 4 – LPTRP route 15 (via Tahao Rd)
    'LOOP 1',     # 5 – LPTRP route 16-1
    'LOOP 2',     # 6 – LPTRP route 16-2
    'EXTERNAL',   # 7 – Long-distance (Camalig, Guinobatan, Ligao, etc.)
]

WINDOWS = ['AM', 'MID', 'PM']

# Duration of each observation window in minutes (used to scale boarding rates)
WINDOW_DURATION_MIN = {'AM': 120, 'MID': 120, 'PM': 120}


def load_field_sheets(csv_path='data/processed/boarding_counts.csv'):
    '''Load the normalized boarding-count data produced by data_parser.py.'''
    df = pd.read_csv(csv_path)
    # Expected columns: date, node, window, route, passengers_boarding, obs_duration_min
    return df


def build_demand_tensor(df):
    '''
    Return numpy array D[j, r, t] of total passengers boarding per window.

    j  = node index (0..len(NODES)-1)
    r  = route index (0..len(ROUTES)-1)
    t  = time-window index (0..len(WINDOWS)-1)

    We aggregate by averaging the boarding rate (pax/min) across observation
    days and scaling to the full window duration.
    '''
    D = np.zeros((len(NODES), len(ROUTES), len(WINDOWS)))
    for j_idx, j in enumerate(NODES):
        for r_idx, r in enumerate(ROUTES):
            for t_idx, t in enumerate(WINDOWS):
                subset = df[
                    (df['node'] == j) &
                    (df['route'] == r) &
                    (df['window'] == t)
                ]
                if len(subset) == 0:
                    D[j_idx, r_idx, t_idx] = 0
                    continue
                # Mean boarding rate (pax per minute) scaled to full window
                rate = (
                    subset['passengers_boarding'] / subset['obs_duration_min']
                ).mean()
                D[j_idx, r_idx, t_idx] = rate * WINDOW_DURATION_MIN[t]
    return D


def bootstrap_demand(df, n_bootstrap=200, seed=42):
    '''Bootstrap resample by observation day to get confidence intervals.'''
    rng = np.random.default_rng(seed)
    unique_dates = df['date'].unique()
    samples = np.zeros((n_bootstrap, len(NODES), len(ROUTES), len(WINDOWS)))
    for b in range(n_bootstrap):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        subset = pd.concat([df[df['date'] == d] for d in sampled_dates])
        samples[b] = build_demand_tensor(subset)
    return samples
