# Jeepney Game Theory Model

Game-theoretic evaluation of the Legazpi-Daraga Local Public Transport Route Plan (LPTRP).
NSTF 2026 · MCS Category · Grade 11 STEM

## What it does

The model treats jeepney drivers as players in a non-cooperative congestion game. It estimates a Nash equilibrium, a social optimum, and the Price of Anarchy, then compares each with the proposed LPTRP allocation.

The complete pipeline also produces bootstrap confidence intervals, fuel-cost and demand sensitivity checks, income-equity measures, multistart equilibrium checks, figures, maps, research tables, and a human-readable summary report.

## Quick start

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:

   ```bash
   python run_pipeline.py
   ```

The default uses the repository's `data/` directory, which contains the digitized `DATA-*.csv` field-survey files. For a shorter development run, skip maps and use fewer bootstrap samples:

```bash
python run_pipeline.py --bootstrap-samples 10 --skip-maps
```

## Field data input

Place raw survey files in a directory and set `GAME_THEORY_RAW_DIR` when the directory is not the repository's `data/` folder:

```bash
# PowerShell
$env:GAME_THEORY_RAW_DIR = "C:\path\to\survey-csvs"
python run_pipeline.py
```

The parser reads `DATA-*.csv` files in the field-survey layout and writes the normalized dataset to `data/processed/boarding_counts.csv`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the expected format and canonical labels.

To generate a self-contained mock dataset instead, run:

```bash
python generate_mock_data.py
python run_pipeline.py
```

## Outputs

All generated results go to `output/`:

- `output/figures/` — 12 publication-ready PNG figures.
- `output/maps/` — static road-network map and interactive Folium route map.
- `output/tables/` — nine CSV result tables, plus demand and validation tables.
- `output/reports/summary_report.txt` — a concise research summary with the synthetic validation result.

## Verification

Run the automated tests with:

```bash
pytest
```

## Citation

[Your name]. (2026). *Game-Theoretic Evaluation of the Legazpi-Daraga LPTRP*. NSTF 2026, MCS Category. [School name], Albay, Philippines.
