# Notebook guide

Notebooks are deliberately thin: they call tested functions from `src/` and
read generated tables rather than reimplementing model calculations.

Run Jupyter from the repository root so imports resolve consistently. Start
with `01_explore_field_data.ipynb`; then use
`02_run_and_interpret_model.ipynb` after the data checks look sensible.
