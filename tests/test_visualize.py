import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.visualize import (
    MODEL_ROUTE_INDICES,
    plot_demand_by_node,
    plot_demand_heatmap,
    plot_gini_comparison,
    plot_multistart_equilibria,
    plot_sensitivity_poa,
    plot_three_configs,
)


def test_enhanced_plots_write_files(tmp_path):
    sensitivity_path = tmp_path / "sensitivity.png"
    gini_path = tmp_path / "gini.png"
    demand_path = tmp_path / "demand.png"
    multistart_path = tmp_path / "multistart.png"

    figures = [
        plot_sensitivity_poa([{"scenario": "alpha=0.3", "price_of_anarchy": 1.2}], sensitivity_path),
        plot_gini_comparison({"nash": 0.2, "lptrp": 0.1, "optimum": 0.05}, gini_path),
        plot_demand_by_node(np.ones((2, 2, 1)), save_path=demand_path),
        plot_multistart_equilibria([(2, 1), (1, 2)], multistart_path),
    ]
    for figure in figures:
        plt.close(figure)
    assert all(path.exists() and path.stat().st_size > 0 for path in (sensitivity_path, gini_path, demand_path, multistart_path))


def test_result_figures_exclude_external_route_and_use_correct_pm_window():
    profiles = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7])
    allocation = plot_three_configs(profiles, profiles, profiles)
    labels = [label.get_text() for label in allocation.axes[0].get_xticklabels()]
    assert 'EXTERNAL' not in labels
    plt.close(allocation)

    demand = np.zeros((5, 8, 3))
    demand[:, 7, :] = 999  # Out-of-scope intermunicipal boardings.
    heatmap = plot_demand_heatmap(demand)
    assert heatmap.axes[0].images[0].get_array().max() == 0
    assert '16:00–18:00' in heatmap.axes[0].get_xticklabels()[2].get_text()
    plt.close(heatmap)
