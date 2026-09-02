# Architecture guide

## Design boundaries

The repository separates raw observations, reproducible model code, and
generated research artefacts. This makes it possible to rerun the study from
the original survey files and to review each transformation independently.

```text
data/raw -> data_parser -> data/processed -> demand -> analysis
                                                |          |
                                                v          v
                                             equilibrium  output/tables, figures, reports
```

## Source modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Canonical labels, route order, and study assumptions. |
| `paths.py` | Repository-relative locations; prevents dependence on the current directory. |
| `data_parser.py` | Converts semi-structured `DATA-*.csv` surveys to a normalized table. |
| `demand.py` | Builds the node × route × time demand tensor and bootstrap samples. |
| `payoffs.py` | Driver earnings, passenger wait time, social-cost functions, and inequality metrics. |
| `equilibrium.py` | Best-response dynamics and multistart equilibrium checks. |
| `optimum.py` | Social-planner allocation search. |
| `analysis.py` | Compares scenarios and produces research metrics. |
| `network.py` | Road-network download/cache and route geometry utilities. |
| `visualize.py` | Figure and map generation only; no model decisions. |

`run_pipeline.py` deliberately contains orchestration rather than new model
math. It calls the modules in a named sequence, writes artefacts, and exposes
the `jeepney-game` command.

## Invariants to protect

- The tuple order in `src/config.py` is the coordinate system for every array.
  Changing it requires changing data, configuration, tests, and documentation
  together.
- `EXTERNAL` remains in raw-demand calculations for auditability, but it is
  excluded from the seven-route allocation model and result figures.
- Raw data is immutable. Correct errors by recording a new survey source, not
  by manually changing `data/processed/boarding_counts.csv`.
- `output/` is generated and ignored by Git. A result is reproducible only when
  it can be regenerated from the committed code, inputs, and assumptions.

## Where new work belongs

Put reusable calculations in the narrowest relevant `src/` module, then add a
unit test. Use `dev/` or a notebook for one-off exploration. Once an
exploration becomes part of the research method, move its logic to `src/`, its
test to `tests/`, and its explanation to `docs/`.
