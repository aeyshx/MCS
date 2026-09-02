# Data contract and dictionary

## Input locations

| Location | Status | Purpose |
| --- | --- | --- |
| `data/raw/DATA-*.csv` | source | Digitized field-survey observations. |
| `data/reference/lptrp.json` | source | Authorized driver allocation. |
| `data/reference/routes.geojson` | source | Map geometry. |
| `data/reference/DATA-*.csv` | source | Non-survey reference extracts, excluded by the parser. |
| `data/processed/boarding_counts.csv` | generated | Normalized observations used by the model. |

The parser ignores files whose name contains `LPTRP` or `REGISTERED`; those are
reference materials rather than boarding-observation files.

## Normalized boarding-count schema

| Column | Type | Meaning |
| --- | --- | --- |
| `date` | ISO date | Observation date (`YYYY-MM-DD`). |
| `node` | categorical | One of the canonical boarding nodes. |
| `window` | categorical | `AM`, `MID`, or `PM`. |
| `route` | categorical | One of the canonical routes below. |
| `passengers_boarding` | integer | Passengers boarding a stopped jeepney in the observation slot. |
| `obs_duration_min` | integer | Duration of that observed slot in minutes. |

The demand module averages passenger rates by node, route, and window, then
scales the rate to a 120-minute model window. Missing combinations have zero
demand; they are not imputed.

## Canonical labels

Nodes: `ROTUNDA`, `PACIFIC MALL`, `DARAGA MARKET`, `EMBARCADERO`, `SM`.

Routes, in the exact model-array order: `DIRETSO A`, `DIRETSO B`, `RAWIS A`,
`RAWIS B`, `ARIMBAY`, `LOOP 1`, `LOOP 2`, `EXTERNAL`.

Time windows: `AM` (06:00–08:00), `MID` (11:00–13:00), and `PM`
(16:00–18:00).
