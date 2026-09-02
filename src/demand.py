import numpy as np
import pandas as pd

from .config import NODES, ROUTES, WINDOWS, WINDOW_DURATION_MIN


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
