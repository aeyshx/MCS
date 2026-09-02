"""Run the reproducible Legazpi-Daraga jeepney game-theory analysis pipeline."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (
    bottleneck_analysis,
    compute_gap_analysis,
    revenue_by_route,
    run_full_analysis,
    validate_synthetic_example,
)
from src.data_parser import build_boarding_counts
from src.demand import NODES, ROUTES, bootstrap_demand, build_demand_tensor, load_field_sheets
from src.config import DEFAULT_ALPHA, DEFAULT_DRIVER_COUNT, DEFAULT_ROUTE_INFO
from src.equilibrium import find_equilibrium, find_equilibrium_multistart
from src.lptrp import load_lptrp_profile
from src.network import build_road_graph, load_routes
from src.paths import ProjectPaths
from src.visualize import (
    MODEL_ROUTE_INDICES,
    MODEL_ROUTES,
    plot_bottleneck,
    plot_convergence,
    plot_demand_by_node,
    plot_demand_heatmap,
    plot_gini_comparison,
    plot_income_distribution,
    plot_interactive_folium_map,
    plot_map_with_routes,
    plot_multistart_equilibria,
    plot_poa_bootstrap_ci,
    plot_sensitivity_poa,
    plot_three_configs,
    plot_wait_time_by_window,
    plot_welfare_comparison,
)

PATHS = ProjectPaths.discover()
RAW_DATA_DIR = Path(os.environ["GAME_THEORY_RAW_DIR"]) if "GAME_THEORY_RAW_DIR" in os.environ else PATHS.raw_data
PROCESSED_DATA_PATH = PATHS.processed_data
OUT_FIGURES = PATHS.figures
OUT_MAPS = PATHS.maps
OUT_TABLES = PATHS.tables
OUT_REPORTS = PATHS.reports
ROUTE_INFO = DEFAULT_ROUTE_INFO


def _make_output_directories():
    PATHS.ensure_output_directories()


def _save_figure(fig):
    """Release figure resources after a plot function has written its output."""
    plt.close(fig)


def _write_summary_report(
    path,
    results,
    validation,
    allocation_rows,
    bottleneck_rows,
    demand_rows,
    node_rows,
    sensitivity_results,
    poa_lower,
    poa_upper,
):
    with path.open("w", encoding="utf-8") as report:
        report.write("=" * 60 + "\nLEGAZPI-DARAGA JEEPNEY GAME THEORY MODEL\nResults Summary Report\n")
        report.write("=" * 60 + "\n\nWELFARE METRICS\n" + "-" * 40 + "\n")
        report.write(
            f"{'Config':<12} {'Social Cost':>14} {'Wait Time (min)':>18} "
            f"{'Income Var (PHP²)':>18} {'Gini':>10} {'Net Income':>15}\n"
        )
        for key in ("nash", "lptrp", "optimum"):
            report.write(
                f"{key.upper():<12} {results['social_costs'][key]:>14.6f} "
                f"{results['wait_times'][key]:>18.2f} {results['income_variances'][key]:>18.2f} "
                f"{results['ginis'][key]:>10.4f} {results['total_revenues'][key]:>15.2f}\n"
            )

        report.write("\nKEY INDICATORS\n" + "-" * 40 + "\n")
        report.write(f"Price of Anarchy        : {results['price_of_anarchy']:.4f}\n")
        if poa_lower is not None and poa_upper is not None:
            report.write(f"PoA 95% CI (Bootstrap)  : [{poa_lower:.4f}, {poa_upper:.4f}]\n")
        report.write(f"LPTRP Improvement Ratio : {results['lptrp_improvement_ratio']:.4f}\n")

        report.write("\nVALIDATION (Synthetic 3-Driver Example)\n" + "-" * 40 + "\n")
        report.write("Expected Nash equilibrium: 2 drivers on Route A, 1 driver on Route B\n")
        report.write(
            f"Computed equilibrium     : {validation['actual_A']} drivers on A, "
            f"{validation['actual_B']} drivers on B\n"
        )
        report.write(f"Test passed              : {'YES' if validation['passed'] else 'NO'}\n")
        report.write(f"Iterations to convergence: {validation['iterations']}\n")

        report.write("\nDRIVER ALLOCATION PER ROUTE\n" + "-" * 40 + "\n")
        for row in allocation_rows:
            report.write(
                f"{row['route']:<15} Nash={row['nash_drivers']:>3}  "
                f"LPTRP={row['lptrp_drivers']:>3}  Optimum={row['optimum_drivers']:>3}\n"
            )

        report.write("\nBOTTLENECK ANALYSIS (gain from reallocating +1 jeepney)\n" + "-" * 60 + "\n")
        for row in bottleneck_rows:
            report.write(f"  {row['route']:<15}: {row['marginal_welfare_gain']:+.4f}\n")

        report.write("\nDEMAND BY ROUTE (passengers per observed window)\n" + "-" * 60 + "\n")
        for row in demand_rows:
            report.write(
                f"  {row['route']:<15}: {row['total_pax']:>8.1f} "
                f"[{row['ci_lower']:>7.1f}, {row['ci_upper']:>7.1f}]\n"
            )
        report.write("\nDEMAND BY NODE (passengers per observed window)\n" + "-" * 60 + "\n")
        for row in node_rows:
            report.write(
                f"  {row['node']:<15}: {row['total_pax']:>8.1f} "
                f"[{row['ci_lower']:>7.1f}, {row['ci_upper']:>7.1f}]\n"
            )
        report.write("\nSENSITIVITY ANALYSIS\n" + "-" * 40 + "\n")
        for row in sensitivity_results:
            report.write(f"  {row['scenario']:<20} {row['price_of_anarchy']:.4f}\n")


def run(bootstrap_samples=200, multistart_runs=10, skip_maps=False):
    """Execute every analysis stage and return the primary results dictionary."""
    _make_output_directories()
    # Explicit initialization prevents failed bootstrap work from being mistaken for valid data.
    D_samples = None
    poa_samples = None
    poa_lower = poa_upper = None

    print("Step 0 – Validating equilibrium solver on the synthetic example")
    validation = validate_synthetic_example()
    pd.DataFrame([validation]).to_csv(OUT_TABLES / "table7_validation_synthetic.csv", index=False)

    print("Step 1 – Parsing field-survey CSVs")
    build_boarding_counts(raw_dir=str(RAW_DATA_DIR), out_path=str(PROCESSED_DATA_PATH))
    field_data = load_field_sheets(PROCESSED_DATA_PATH)
    if field_data.empty:
        raise RuntimeError(
            f"No usable boarding observations were parsed from {RAW_DATA_DIR}. "
            "Set GAME_THEORY_RAW_DIR to the directory containing DATA-*.csv files."
        )
    demand = build_demand_tensor(field_data)

    n_drivers = DEFAULT_DRIVER_COUNT
    lptrp_profile = load_lptrp_profile(PATHS.lptrp_profile, n_drivers)
    print("Step 2 – Finding Nash equilibrium and social optimum")
    results = run_full_analysis(demand, ROUTE_INFO, n_drivers, lptrp_profile, alpha=DEFAULT_ALPHA)

    print("Step 3 – Computing bottlenecks, confidence intervals, and sensitivity")
    bottlenecks = bottleneck_analysis(demand, ROUTE_INFO, n_drivers, results["nash_profile"])
    sorted_bottlenecks = sorted(
        (
            (route_index, value)
            for route_index, value in bottlenecks.items()
            if route_index in MODEL_ROUTE_INDICES
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if bootstrap_samples > 0:
        try:
            D_samples = bootstrap_demand(field_data, n_bootstrap=bootstrap_samples, seed=42)
            bootstrap_poa = []
            for sample in D_samples:
                bootstrap_result = run_full_analysis(sample, ROUTE_INFO, n_drivers, lptrp_profile, alpha=DEFAULT_ALPHA)
                bootstrap_poa.append(bootstrap_result["price_of_anarchy"])
            poa_samples = np.asarray(bootstrap_poa)
            poa_lower, poa_upper = np.percentile(poa_samples, [2.5, 97.5])
        except Exception as exc:  # A report remains useful even when optional resampling fails.
            print(f"Warning: bootstrap confidence intervals were unavailable: {exc}")
            D_samples = None
            poa_samples = None

    sensitivity_results = []
    for test_alpha in (0.3, 0.5, 0.9):
        scenario = run_full_analysis(demand, ROUTE_INFO, n_drivers, lptrp_profile, alpha=test_alpha)
        sensitivity_results.append({"scenario": f"alpha={test_alpha}", "price_of_anarchy": scenario["price_of_anarchy"]})
    for demand_multiplier in (0.8, 1.2):
        scenario = run_full_analysis(
            demand * demand_multiplier, ROUTE_INFO, n_drivers, lptrp_profile, alpha=DEFAULT_ALPHA
        )
        sensitivity_results.append({"scenario": f"demand={demand_multiplier}x", "price_of_anarchy": scenario["price_of_anarchy"]})
    for fuel_multiplier in (0.8, 1.2):
        scenario = run_full_analysis(
            demand,
            ROUTE_INFO,
            n_drivers,
            lptrp_profile,
            alpha=DEFAULT_ALPHA,
            fuel_cost_multiplier=fuel_multiplier,
        )
        sensitivity_results.append({"scenario": f"fuel_cost={fuel_multiplier}x", "price_of_anarchy": scenario["price_of_anarchy"]})

    equilibrium_counts, unique_equilibria = find_equilibrium_multistart(
        demand, ROUTE_INFO, n_drivers, n_starts=multistart_runs, seed=42
    )
    print(f"Found {len(unique_equilibria)} unique equilibrium configurations in {multistart_runs} starts.")

    print("Step 4 – Writing tables")
    configs = ("nash", "lptrp", "optimum")
    summary_rows = [
        {
            "configuration": key.upper(),
            "social_cost": results["social_costs"][key],
            "total_wait_min": results["wait_times"][key],
            "income_variance": results["income_variances"][key],
            "gini_coefficient": results["ginis"][key],
            "total_net_income_php": results["total_revenues"][key],
        }
        for key in configs
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_TABLES / "table1_welfare_comparison.csv", index=False)
    bottleneck_rows = [
        {"route": ROUTES[route_idx], "marginal_welfare_gain": value}
        for route_idx, value in sorted_bottlenecks
    ]
    pd.DataFrame(bottleneck_rows).to_csv(OUT_TABLES / "table2_bottleneck_analysis.csv", index=False)
    allocation_rows = [
        {
            "route": route,
            "length_km": ROUTE_INFO["length_km"][route_idx],
            "cycle_time_min": ROUTE_INFO["cycle_time_min"][route_idx],
            "nash_drivers": int((results["nash_profile"] == route_idx).sum()),
            "lptrp_drivers": int((results["lptrp_profile"] == route_idx).sum()),
            "optimum_drivers": int((results["optimum_profile"] == route_idx).sum()),
        }
        for route_idx, route in enumerate(ROUTES)
        if route_idx in MODEL_ROUTE_INDICES
    ]
    pd.DataFrame(allocation_rows).to_csv(OUT_TABLES / "table3_driver_allocation.csv", index=False)

    route_totals = demand.sum(axis=(0, 2))
    node_totals = demand[:, MODEL_ROUTE_INDICES, :].sum(axis=(1, 2))
    if D_samples is not None:
        route_sample_totals = D_samples.sum(axis=(1, 3))
        node_sample_totals = D_samples[:, :, MODEL_ROUTE_INDICES, :].sum(axis=(2, 3))
        route_lower, route_upper = np.percentile(route_sample_totals, [2.5, 97.5], axis=0)
        node_lower, node_upper = np.percentile(node_sample_totals, [2.5, 97.5], axis=0)
    else:
        route_lower = route_upper = route_totals
        node_lower = node_upper = node_totals
    demand_rows = [
        {
            "route": route,
            "am_pax": demand[:, route_idx, 0].sum(),
            "mid_pax": demand[:, route_idx, 1].sum(),
            "pm_pax": demand[:, route_idx, 2].sum(),
            "total_pax": route_totals[route_idx],
            "ci_lower": route_lower[route_idx],
            "ci_upper": route_upper[route_idx],
        }
        for route_idx, route in enumerate(ROUTES)
        if route_idx in MODEL_ROUTE_INDICES
    ]
    pd.DataFrame(demand_rows).to_csv(OUT_TABLES / "table4_demand_by_route.csv", index=False)
    node_rows = [
        {
            "node": node,
            "am_pax": demand[node_idx, MODEL_ROUTE_INDICES, 0].sum(),
            "mid_pax": demand[node_idx, MODEL_ROUTE_INDICES, 1].sum(),
            "pm_pax": demand[node_idx, MODEL_ROUTE_INDICES, 2].sum(),
            "total_pax": node_totals[node_idx],
            "ci_lower": node_lower[node_idx],
            "ci_upper": node_upper[node_idx],
        }
        for node_idx, node in enumerate(NODES)
    ]
    pd.DataFrame(node_rows).to_csv(OUT_TABLES / "table4b_demand_by_node.csv", index=False)
    pd.DataFrame(sensitivity_results).to_csv(OUT_TABLES / "table5_sensitivity_analysis.csv", index=False)
    pd.DataFrame(
        [
            {"start": index, **dict(zip(MODEL_ROUTES, np.asarray(counts)[list(MODEL_ROUTE_INDICES)]))}
            for index, counts in enumerate(equilibrium_counts, start=1)
        ]
    ).to_csv(OUT_TABLES / "table6_multistart_equilibria.csv", index=False)
    compute_gap_analysis(results, ROUTES).query("route != 'EXTERNAL'").to_csv(
        OUT_TABLES / "table8_lptrp_gap_analysis.csv", index=False
    )
    revenue_rows = [{"route": route} for route in MODEL_ROUTES]
    for key in configs:
        revenues = revenue_by_route(results[f"{key}_profile"], demand, ROUTE_INFO)
        for route_idx, row in zip(MODEL_ROUTE_INDICES, revenue_rows):
            row[f"{key}_gross_revenue_php"] = revenues[route_idx]
    pd.DataFrame(revenue_rows).to_csv(OUT_TABLES / "table9_revenue_by_route.csv", index=False)

    print("Step 5 – Generating figures")
    _save_figure(plot_three_configs(results["nash_profile"], results["optimum_profile"], results["lptrp_profile"], OUT_FIGURES / "fig1_driver_allocation_comparison.png"))
    _save_figure(plot_welfare_comparison(results["social_costs"], results["wait_times"], results["income_variances"], OUT_FIGURES / "fig2_welfare_comparison.png"))
    _save_figure(plot_demand_heatmap(demand, OUT_FIGURES / "fig3_demand_heatmap.png"))
    initial = np.random.default_rng(42).integers(0, len(ROUTES), size=n_drivers)
    _, convergence_log = find_equilibrium(initial, demand, ROUTE_INFO)
    _save_figure(plot_convergence(convergence_log, OUT_FIGURES / "fig4_convergence.png"))
    _save_figure(plot_bottleneck(bottlenecks, OUT_FIGURES / "fig5_bottleneck.png"))
    _save_figure(plot_sensitivity_poa(sensitivity_results, OUT_FIGURES / "fig6_sensitivity_poa.png"))
    # Figures 1–6 are the concise results set corresponding to the research questions.

    if not skip_maps:
        print("Step 6 – Generating road-network maps")
        graph = build_road_graph()
        routes = load_routes(PATHS.route_geometries)
        _save_figure(plot_map_with_routes(graph, routes, OUT_MAPS / "map1_road_network_routes.png"))
        plot_interactive_folium_map(routes, OUT_MAPS / "map2_pax_flow_folium.html")

    _write_summary_report(
        OUT_REPORTS / "summary_report.txt",
        results,
        validation,
        allocation_rows,
        bottleneck_rows,
        demand_rows,
        node_rows,
        sensitivity_results,
        poa_lower,
        poa_upper,
    )
    print(f"Pipeline complete. Outputs written to {PATHS.root / 'output'}")
    return results


def main():
    """Parse command-line arguments and execute the complete pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=int(os.environ.get("GAME_THEORY_BOOTSTRAP_SAMPLES", "200")),
        help="number of day-level bootstrap resamples (0 disables bootstrapping)",
    )
    parser.add_argument("--multistart-runs", type=int, default=10)
    parser.add_argument("--skip-maps", action="store_true", help="skip map rendering for a faster analysis-only run")
    arguments = parser.parse_args()
    run(arguments.bootstrap_samples, arguments.multistart_runs, arguments.skip_maps)


if __name__ == "__main__":
    main()
