# Jeepney Game Theory Model

Reproducible game-theoretic evaluation of the Legazpi-Daraga Local Public
Transport Route Plan (LPTRP). The model estimates a Nash equilibrium, the
social optimum, and the Price of Anarchy from field-survey boarding counts.

NSTF 2026 · MCS Category

## Start here

Install the project and run the fast, analysis-only workflow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
jeepney-game --bootstrap-samples 10 --skip-maps
```

The same command can be run without installation from the repository root:

```powershell
python run_pipeline.py --bootstrap-samples 10 --skip-maps
```

Run the quality checks before making a change:

```powershell
pytest
ruff check .
```

## Repository map

```text
src/                 Model, data preparation, analysis, and visualization code
tests/               Fast unit tests for model behaviour
notebooks/           Guided, reproducible exploration and interpretation
data/raw/            Immutable digitized field-survey CSV files
data/reference/      LPTRP allocation, route geometry, and reference extracts
data/processed/      Generated normalized boarding-count dataset (not committed)
data/mock/           Locally generated demonstration data (not committed)
output/              Generated tables, figures, maps, and report (not committed)
docs/                Architecture and data-contract documentation
dev/                 Short-lived experiments; promote stable work into tests
```

See [the architecture guide](docs/architecture.md) for module responsibilities
and [the data dictionary](docs/data_dictionary.md) before changing model inputs.

## Reproducible workflow

1. Keep original digitized surveys in `data/raw/`; do not edit them in place.
2. Run the pipeline. It parses the surveys into
   `data/processed/boarding_counts.csv`, then writes all results to `output/`.
3. Inspect tables and `output/reports/summary_report.txt` before interpreting
   figures.
4. Use the notebooks only for exploration and interpretation. Put reusable
   production logic in `src/` and protect it with tests in `tests/`.

The optional `GAME_THEORY_RAW_DIR` environment variable points the parser to a
different directory of `DATA-*.csv` files:

```powershell
$env:GAME_THEORY_RAW_DIR = "C:\path\to\field-surveys"
jeepney-game --skip-maps
```

For a safe demo dataset, generate data outside the field-data directory:

```powershell
python generate_mock_data.py
$env:GAME_THEORY_RAW_DIR = "data/mock"
jeepney-game --bootstrap-samples 10 --skip-maps
```

## Notebooks

Open Jupyter Lab from the repository root:

```powershell
jupyter lab
```

- `01_network.ipynb` checks route geometry and map inputs.
- `02_demand.ipynb` audits source surveys, normalization, and the demand tensor.
- `03_equilibrium.ipynb` connects the game mechanics to allocation diagnostics.
- `04_results.ipynb` runs and interprets the official research pipeline.
- `05_visualization.ipynb` reviews presentation-ready output assets.

## Citation

[Your name]. (2026). *Game-Theoretic Evaluation of the Legazpi-Daraga LPTRP*.
NSTF 2026, MCS Category. [School name], Albay, Philippines.
