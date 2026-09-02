import matplotlib.pyplot as plt
import numpy as np

from src.visualize import (
    plot_demand_by_node,
    plot_gini_comparison,
    plot_multistart_equilibria,
    plot_sensitivity_poa,
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
