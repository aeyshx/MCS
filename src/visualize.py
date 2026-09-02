import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

from .config import MODEL_ROUTE_INDICES, MODEL_ROUTES, NODES, ROUTES
from .network import nearest_graph_node, route_edges, route_length_km, route_cycle_time_min
from .payoffs import all_payoffs, passenger_wait_time_by_window

# Canonical route colors keyed to the ROUTES list in demand.py
ROUTE_COLORS = {
    'DIRETSO A': '#e41a1c',
    'DIRETSO B': '#377eb8',
    'RAWIS A':   '#4daf4a',
    'RAWIS B':   '#984ea3',
    'ARIMBAY':   '#ff7f00',
    'LOOP 1':    '#a65628',
    'LOOP 2':    '#f781bf',
    'EXTERNAL':  '#999999',
}

# Short labels for axes
ROUTE_SHORT = {r: r.replace(' ', '\n') for r in ROUTES}
CONFIG_COLORS = {'nash': '#d7191c', 'lptrp': '#2c7bb6', 'optimum': '#1a9641'}
HATCH_PATTERNS = {'nash': '///', 'lptrp': '...', 'optimum': ''}

TIME_WINDOW_LABELS = ('AM peak\n(06:00–08:00)', 'Midday\n(11:00–13:00)', 'PM peak\n(16:00–18:00)')


def _available_model_route_indices(D):
    """Return in-scope route columns available in a demand tensor."""
    return tuple(index for index in MODEL_ROUTE_INDICES if index < D.shape[1])


def plot_route_counts(profile, ax=None, title=None, route_indices=MODEL_ROUTE_INDICES):
    '''Bar chart: number of drivers per route.'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    full_counts = np.bincount(profile, minlength=len(ROUTES))
    routes = [ROUTES[index] for index in route_indices]
    counts = full_counts[list(route_indices)]
    colors = [ROUTE_COLORS.get(route, '#888888') for route in routes]
    bars = ax.bar(range(len(routes)), counts, color=colors)
    ax.set_xticks(range(len(routes)))
    ax.set_xticklabels([ROUTE_SHORT[route] for route in routes], fontsize=8)
    ax.set_ylabel('Number of drivers')
    ax.set_xlabel('Route')
    if title:
        ax.set_title(title)
    # Annotate counts on top of each bar
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(cnt), ha='center', va='bottom', fontsize=8)
    return ax


def plot_three_configs(nash, opt, lptrp, save_path=None):
    '''Compare allocations on the seven in-scope Legazpi-Daraga routes.'''
    profiles = {'nash': nash, 'lptrp': lptrp, 'optimum': opt}
    labels = {'nash': 'Nash equilibrium', 'lptrp': 'Proposed LPTRP', 'optimum': 'Social optimum'}
    x = np.arange(len(MODEL_ROUTES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    for position, key in enumerate(('nash', 'lptrp', 'optimum')):
        counts = np.bincount(profiles[key], minlength=len(ROUTES))[list(MODEL_ROUTE_INDICES)]
        bars = ax.bar(
            x + (position - 1) * width,
            counts,
            width,
            label=labels[key],
            color=CONFIG_COLORS[key],
            hatch=HATCH_PATTERNS[key],
        )
        ax.bar_label(bars, padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ROUTE_SHORT[route] for route in MODEL_ROUTES], fontsize=9)
    ax.set_ylabel('Allocated drivers')
    ax.set_xlabel('Modeled jeepney route')
    ax.set_title('Driver Allocation by Configuration')
    ax.set_ylim(0, max(
        np.bincount(profile, minlength=len(ROUTES))[list(MODEL_ROUTE_INDICES)].max()
        for profile in profiles.values()
    ) + 7)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_convergence(log, save_path=None):
    '''Show number of driver switches per iteration during BR dynamics.'''
    fig, ax = plt.subplots(figsize=(8, 5))
    iters   = [entry['iter']    for entry in log]
    changes = [entry['changes'] for entry in log]
    ax.plot(iters, changes, 'o-', color='#2c7bb6', linewidth=1.5, markersize=4)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Drivers who switched route')
    ax.set_title('Best-Response Dynamics: Convergence to Nash Equilibrium')
    ax.set_xticks(iters)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_demand_heatmap(D, save_path=None):
    '''
    Heatmap of total passenger demand per node per time window,
    summed across all routes.  Shape of D: (nodes, routes, windows).
    '''
    # Keep out-of-scope intermunicipal boardings out of corridor model results.
    node_window = D[:, _available_model_route_indices(D), :].sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(node_window, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='Passengers observed')
    ax.set_xticks(range(len(TIME_WINDOW_LABELS)))
    ax.set_xticklabels(TIME_WINDOW_LABELS)
    ax.set_yticks(range(len(NODES)))
    ax.set_yticklabels(NODES)
    ax.set_title('Observed Boarding Demand by Node and Time Window')
    # Annotate cells
    for i in range(len(NODES)):
        for j in range(len(TIME_WINDOW_LABELS)):
            ax.text(j, i, f'{node_window[i, j]:.0f}',
                    ha='center', va='center', fontsize=9,
                    color='black' if node_window[i, j] < node_window.max() * 0.6 else 'white')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_welfare_comparison(social_costs, wait_times, income_variances, save_path=None):
    '''
    Three-panel bar chart comparing Social Cost, Wait Time, and Income Variance
    across Nash, LPTRP, and Social Optimum configurations.
    '''
    configs = ['Nash\nEquilibrium', 'Proposed\nLPTRP', 'Social\nOptimum']
    keys    = ['nash', 'lptrp', 'optimum']
    colors  = [CONFIG_COLORS[key] for key in keys]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Social Cost (normalized)
    sc_vals = [social_costs[k] for k in keys]
    axes[0].bar(configs, sc_vals, color=colors)
    axes[0].set_title('Normalized Social Cost\n(lower = better)')
    axes[0].set_ylabel('Social Cost (dimensionless)')

    # Panel 2: Total Passenger Wait Time
    wt_vals = [wait_times[k] / 60 for k in keys]  # convert to hours
    axes[1].bar(configs, wt_vals, color=colors)
    axes[1].set_title('Total Passenger Wait Time\n(lower = better)')
    axes[1].set_ylabel('Wait time (passenger-hours)')

    # Panel 3: Driver Income Variance
    iv_vals = [income_variances[k] for k in keys]
    axes[2].bar(configs, iv_vals, color=colors)
    axes[2].set_title('Driver Income Variance\n(lower = more equal)')
    axes[2].set_ylabel('Variance (PHP²)')

    for ax, vals in zip(axes, [sc_vals, wt_vals, iv_vals]):
        for bar, v in zip(ax.patches, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Welfare Comparison Across Configurations', fontsize=13)
    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_bottleneck(bottlenecks, save_path=None):
    '''
    Horizontal bar chart of the marginal welfare gain from adding one jeepney
    to each route (positive = undersupplied bottleneck).
    '''
    model_bottlenecks = [
        (route_index, value) for route_index, value in bottlenecks.items()
        if route_index in MODEL_ROUTE_INDICES
    ]
    sorted_bn = sorted(model_bottlenecks, key=lambda x: x[1])
    route_names = [ROUTES[i] for i, _ in sorted_bn]
    values = [v for _, v in sorted_bn]
    colors = ['#1a9641' if v > 0 else '#d7191c' for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(route_names, values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Marginal welfare gain from one additional jeepney')
    ax.set_title('Bottleneck Analysis of Modeled Routes')
    label_offset = max(abs(value) for value in values) * 0.01
    for bar, v in zip(bars, values):
        ax.text(v + (label_offset if v >= 0 else -label_offset),
                bar.get_y() + bar.get_height() / 2,
                f'{v:+.4f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=8)
    ax.set_xlim(min(0, min(values)) * 1.6, max(0, max(values)) * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_sensitivity_poa(sensitivity_results, save_path=None):
    """Plot Price of Anarchy across alpha, demand, and fuel-cost scenarios."""
    labels = [row['scenario'] for row in sensitivity_results]
    poas = [row['price_of_anarchy'] for row in sensitivity_results]
    colors = [
        '#2196f3' if label.startswith('alpha') else '#ff9800'
        if label.startswith('demand') else '#4caf50'
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    markers = ['o' if label.startswith('alpha') else 's' if label.startswith('demand') else '^' for label in labels]
    for indices in ((0, 1, 2), (3, 4), (5, 6)):
        valid_indices = [index for index in indices if index < len(poas)]
        if valid_indices:
            ax.plot(x[valid_indices], np.asarray(poas)[valid_indices], color='#666666', linewidth=1, zorder=1)
    for x_value, poa, color, marker in zip(x, poas, colors, markers):
        ax.scatter(x_value, poa, color=color, marker=marker, s=58, zorder=2)
        ax.annotate(f'{poa:.3f}', (x_value, poa), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='PoA = 1 (no loss)')
    padding = max(0.01, (max(poas) - min(poas)) * 0.25)
    ax.set_ylim(1.0 - padding, max(poas) + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Price of Anarchy')
    ax.set_title('Price of Anarchy Across Sensitivity Scenarios')
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_income_distribution(results, D, route_info, save_path=None):
    """Plot net daily driver-income distributions for the three configurations."""
    configs = [('nash', 'Nash\nEquilibrium'), ('lptrp', 'Proposed\nLPTRP'), ('optimum', 'Social\nOptimum')]
    data = [all_payoffs(results[f'{key}_profile'], D, route_info) for key, _ in configs]
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True, showextrema=True)
    for body, (key, _) in zip(parts['bodies'], configs):
        body.set_facecolor(CONFIG_COLORS[key])
        body.set_edgecolor('black')
        body.set_alpha(0.75)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([label for _, label in configs])
    ax.set_ylabel('Expected daily net income (PHP)')
    ax.set_title('Driver Income Distribution by Configuration')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_poa_bootstrap_ci(poa_samples, poa_observed, save_path=None):
    """Plot the bootstrap sampling distribution and 95% PoA confidence interval."""
    samples = np.asarray(poa_samples, dtype=float)
    if samples.size == 0:
        raise ValueError('At least one PoA bootstrap sample is required')
    lower, upper = np.percentile(samples, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(samples, bins=min(20, max(1, samples.size)), color='#5c6bc0', edgecolor='white', alpha=0.85)
    ax.axvline(lower, color='orange', linestyle='--', linewidth=1.5, label='95% CI bounds')
    ax.axvline(upper, color='orange', linestyle='--', linewidth=1.5)
    ax.axvline(poa_observed, color='red', linewidth=2, label=f'Observed PoA = {poa_observed:.3f}')
    ax.set_xlabel('Price of Anarchy')
    ax.set_ylabel('Bootstrap frequency')
    ax.set_title('Bootstrap Distribution of the Price of Anarchy')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_gini_comparison(ginis, save_path=None):
    """Plot income inequality as a Gini coefficient for each configuration."""
    keys = ['nash', 'lptrp', 'optimum']
    labels = ['Nash\nEquilibrium', 'Proposed\nLPTRP', 'Social\nOptimum']
    values = [ginis[key] for key in keys]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=[CONFIG_COLORS[key] for key in keys], edgecolor='black')
    for bar, key, value in zip(bars, keys, values):
        bar.set_hatch(HATCH_PATTERNS[key])
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    ax.set_ylim(0, max(1.0, max(values) * 1.2))
    ax.set_ylabel('Gini coefficient (0 = equal, 1 = unequal)')
    ax.set_title('Driver Income Equality by Configuration')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_wait_time_by_window(D, results, route_info, save_path=None):
    """Compare passenger-minutes of waiting by observation window."""
    configs = [('nash', 'Nash'), ('lptrp', 'LPTRP'), ('optimum', 'Optimum')]
    x = np.arange(D.shape[2])
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (key, label) in enumerate(configs):
        waits = passenger_wait_time_by_window(results[f'{key}_profile'], D, route_info)
        bars = ax.bar(x + idx * width, waits, width, label=label, color=CONFIG_COLORS[key])
        for bar in bars:
            bar.set_hatch(HATCH_PATTERNS[key])
    ax.set_xticks(x + width)
    ax.set_xticklabels(TIME_WINDOW_LABELS[:D.shape[2]])
    ax.set_ylabel('Total passenger wait time (minutes)')
    ax.set_title('Wait Times by Time Window and Configuration')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_demand_by_node(D, node_lower=None, node_upper=None, save_path=None):
    """Plot demand totals by survey node, optionally with bootstrap intervals."""
    totals = D[:, _available_model_route_indices(D), :].sum(axis=(1, 2))
    lower = totals if node_lower is None else np.asarray(node_lower)
    upper = totals if node_upper is None else np.asarray(node_upper)
    errors = np.vstack([np.maximum(0, totals - lower), np.maximum(0, upper - totals)])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(NODES[:len(totals)], totals, yerr=errors, capsize=4, color='#4575b4', edgecolor='black')
    for bar in bars:
        bar.set_hatch('///')
    ax.set_ylabel('Passengers per observed window')
    ax.set_title('Observed Boarding Demand by Survey Node (Modeled Routes)')
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_multistart_equilibria(equilibrium_counts, save_path=None):
    """Display route-count vectors found from independent Nash initializations."""
    counts = np.asarray(equilibrium_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError('equilibrium_counts must be a two-dimensional route-count array')
    displayed_counts = counts[:, list(MODEL_ROUTE_INDICES)] if counts.shape[1] == len(ROUTES) else counts
    displayed_routes = MODEL_ROUTES if counts.shape[1] == len(ROUTES) else ROUTES[:counts.shape[1]]
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(displayed_counts, aspect='auto', cmap='Blues')
    fig.colorbar(image, ax=ax, label='Drivers assigned')
    ax.set_xticks(np.arange(displayed_counts.shape[1]))
    ax.set_xticklabels([ROUTE_SHORT[route] for route in displayed_routes], fontsize=8)
    ax.set_yticks(np.arange(counts.shape[0]))
    ax.set_yticklabels([f'Start {idx + 1}' for idx in range(counts.shape[0])])
    ax.set_title('Nash Equilibrium Route Counts Across Random Initializations')
    for row in range(displayed_counts.shape[0]):
        for column in range(displayed_counts.shape[1]):
            ax.text(column, row, f'{displayed_counts[row, column]:.0f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_interactive_folium_map(routes_gdf, save_path=None):
    """Create an interactive route-overlay map and optionally save it as HTML."""
    import folium

    all_coords = [coord for _, row in routes_gdf.iterrows() for coord in row.geometry.coords]
    center_lon = float(np.mean([coord[0] for coord in all_coords]))
    center_lat = float(np.mean([coord[1] for coord in all_coords]))
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
    survey_nodes = {
        'ROTUNDA': (13.1990, 123.7470),
        'PACIFIC MALL': (13.1920, 123.7470),
        'DARAGA MARKET': (13.1566, 123.7286),
        'EMBARCADERO': (13.1430, 123.7590),
        'SM': (13.1450, 123.7439),
    }
    for name, (lat, lon) in survey_nodes.items():
        folium.CircleMarker(
            [lat, lon], radius=7, color='black', fill=True, fill_color='#ffea00',
            fill_opacity=0.9, tooltip=name,
        ).add_to(route_map)
    for _, row in routes_gdf.iterrows():
        route_code = row.get('route_code', 'Unknown')
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color=ROUTE_COLORS.get(route_code, '#000000'),
            weight=4,
            opacity=0.8,
            tooltip=f'Route: {route_code}',
        ).add_to(route_map)
    if save_path:
        route_map.save(save_path)
    return route_map


def plot_map_with_routes(G, routes_gdf, save_path=None):
    '''
    Plot the OSM road network as a background, then overlay each jeepney route
    as a road-snapped colored polyline with parallel offsets for overlapping segments.
    Key passenger survey nodes are highlighted with clear landmark badges.
    '''
    from collections import defaultdict
    import matplotlib.lines as mlines
    import networkx as nx

    # ── Helper for calculating parallel offset coordinates ───────────────────
    def _compute_parallel_offset(coords, offset_dist):
        if len(coords) < 2 or offset_dist == 0:
            return coords
        c_arr = np.array(coords)
        dx = np.diff(c_arr[:, 0])
        dy = np.diff(c_arr[:, 1])
        lengths = np.hypot(dx, dy)
        lengths[lengths == 0] = 1e-9

        # Perpendicular normal vectors: (-dy, dx)
        nx_seg = -dy / lengths
        ny_seg = dx / lengths

        vn_x = np.zeros(len(c_arr))
        vn_y = np.zeros(len(c_arr))
        vn_x[0], vn_y[0] = nx_seg[0], ny_seg[0]
        vn_x[-1], vn_y[-1] = nx_seg[-1], ny_seg[-1]

        for i in range(1, len(c_arr) - 1):
            avg_x = (nx_seg[i - 1] + nx_seg[i]) / 2.0
            avg_y = (ny_seg[i - 1] + ny_seg[i]) / 2.0
            norm = np.hypot(avg_x, avg_y)
            if norm > 1e-6:
                vn_x[i] = avg_x / norm
                vn_y[i] = avg_y / norm
            else:
                vn_x[i] = nx_seg[i]
                vn_y[i] = ny_seg[i]

        off_x = c_arr[:, 0] + vn_x * offset_dist
        off_y = c_arr[:, 1] + vn_y * offset_dist
        return list(zip(off_x, off_y))

    # ── 1. Map route waypoints to shortest paths and collect edge sharing ─────
    route_edges_dict = {}
    edge_routes = defaultdict(list)

    for _, row in routes_gdf.iterrows():
        route_code = row['route_code']
        geom = row.geometry
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        node_ids = [nearest_graph_node(G, lon, lat) for lon, lat in coords]
        edges = []
        for n_from, n_to in zip(node_ids[:-1], node_ids[1:]):
            if n_from == n_to:
                continue
            try:
                path = nx.shortest_path(G, n_from, n_to, weight='travel_time')
                edges.extend(zip(path[:-1], path[1:]))
            except nx.NetworkXNoPath:
                continue

        route_edges_dict[route_code] = edges
        for u, v in edges:
            key = tuple(sorted((u, v)))
            if route_code not in edge_routes[key]:
                edge_routes[key].append(route_code)

    # ── 2. Render base road network ──────────────────────────────────────────
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color='#222436',
        edge_linewidth=0.65,
        edge_alpha=0.65,
        show=False,
        close=False,
        bgcolor='#0f111a',
        figsize=(16, 12),
    )

    legend_handles = []
    base_spacing = 0.00030  # ~33 meters in coordinates

    # ── 3. Draw each route with displaced parallel offsets ───────────────────
    for _, row in routes_gdf.iterrows():
        route_code = row['route_code']
        color = ROUTE_COLORS.get(route_code, '#ffffff')
        edges = route_edges_dict.get(route_code, [])

        total_length_m = 0.0
        for u, v in edges:
            key = tuple(sorted((u, v)))
            routes_on_edge = edge_routes[key]
            n_shared = len(routes_on_edge)
            r_idx = routes_on_edge.index(route_code)

            # Center multiple routes across the road segment
            if n_shared == 1:
                offset_dist = 0.0
            else:
                offset_dist = (r_idx - (n_shared - 1) / 2.0) * base_spacing

            edge_data = min(
                G.get_edge_data(u, v).values(),
                key=lambda d: d.get('length', float('inf'))
            )
            total_length_m += edge_data.get('length', 0)

            if 'geometry' in edge_data:
                line_coords = list(edge_data['geometry'].coords)
            else:
                line_coords = [
                    (G.nodes[u]['x'], G.nodes[u]['y']),
                    (G.nodes[v]['x'], G.nodes[v]['y'])
                ]

            # Consistent normal direction regardless of traversal orientation
            if u > v:
                line_coords = list(reversed(line_coords))

            off_line = _compute_parallel_offset(line_coords, offset_dist)
            xs, ys = zip(*off_line)

            ax.plot(
                xs, ys,
                color=color,
                linewidth=2.6,
                alpha=0.92,
                solid_capstyle='round',
                solid_joinstyle='round',
                zorder=5,
            )

        # Legend entry with accurate distance and cycle time
        length_km = total_length_m / 1000.0
        cycle_min = round((length_km / 25.0) * 60.0)
        label = f"{route_code:<10} ({length_km:.1f} km · ~{cycle_min} min)"
        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=3.0, alpha=0.95, label=label)
        )

    # ── 4. Overlay Key Survey Nodes / Urban Hubs ─────────────────────────────
    survey_nodes = {
        'ROTUNDA': (123.7438, 13.1391),
        'PACIFIC MALL': (123.7493, 13.1438),
        'DARAGA MARKET': (123.7160, 13.1500),
        'EMBARCADERO': (123.7590, 13.1430),
        'SM CITY': (123.7439, 13.1450),
    }

    for name, (lon, lat) in survey_nodes.items():
        ax.scatter(lon, lat, s=110, color='#ffea00', edgecolors='#000000', linewidth=1.4, zorder=10)
        ax.scatter(lon, lat, s=32, color='#d50000', edgecolors='none', zorder=11)
        ax.annotate(
            name, (lon, lat),
            xytext=(0, 10), textcoords='offset points',
            fontsize=8.5, fontweight='bold', color='#ffffff',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#1b1e2e', edgecolor='#ffea00', alpha=0.88, linewidth=0.8),
            zorder=12
        )

    # ── 5. Clean Legend, Titles & Viewport Framing ────────────────────────────
    legend = ax.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=8.5,
        title='Jeepney Routes (Displaced Parallel Traces)',
        title_fontsize=10,
        framealpha=0.92,
        facecolor='#141724',
        labelcolor='white',
        edgecolor='#3b4261',
        borderpad=0.8,
    )
    legend.get_title().set_color('#00d2ff')
    legend.get_title().set_fontweight('bold')

    ax.set_title(
        'Legazpi–Daraga Jeepney Transport Route Network\nDisplaced Multi-Route Traces & Passenger Survey Hubs',
        color='white',
        fontsize=14,
        fontweight='bold',
        pad=14,
    )

    # Tighten viewport padding around active corridor
    all_lons = [c[0] for _, r in routes_gdf.iterrows() for c in r.geometry.coords] + [lon for lon, _ in survey_nodes.values()]
    all_lats = [c[1] for _, r in routes_gdf.iterrows() for c in r.geometry.coords] + [lat for _, lat in survey_nodes.values()]
    pad_x = (max(all_lons) - min(all_lons)) * 0.05
    pad_y = (max(all_lats) - min(all_lats)) * 0.05
    ax.set_xlim(min(all_lons) - pad_x, max(all_lons) + pad_x)
    ax.set_ylim(min(all_lats) - pad_y, max(all_lats) + pad_y)

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig
