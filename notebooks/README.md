# Notebook sequence

These notebooks follow the learning sequence in the implementation handbook.
They are research walkthroughs, not alternate implementations: all reusable
logic belongs in `src/` and is protected by `tests/`.

| Notebook | Purpose | Typical use |
| --- | --- | --- |
| `01_network.ipynb` | Inspect route geometry and understand the map inputs. | Once, and whenever route geometry changes. |
| `02_demand.ipynb` | Parse raw sheets and audit the demand tensor. | After every data collection or correction. |
| `03_equilibrium.ipynb` | Validate the game mechanics and inspect allocations. | When reviewing modelling assumptions. |
| `04_results.ipynb` | Run the official pipeline and interpret the research tables. | For a final or reproducible study run. |
| `05_visualization.ipynb` | Review and regenerate selected publication figures. | Before a paper, poster, or presentation. |

Launch Jupyter from the repository root with `jupyter lab`. Run notebooks in
order. Each notebook displays its prerequisites and avoids hidden state; it can
be restarted and rerun from the first cell.

For an offline or quick practice run, generate mock data first and set
`GAME_THEORY_RAW_DIR=data/mock`. Do not mix mock outputs with research results.
