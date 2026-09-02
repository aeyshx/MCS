# Data contribution guide

## Raw survey files

Store digitized field-survey files as `DATA-*.csv`. Each observation section is separated by a blank row and has this structure:

1. Node name
2. Observation date (for example, `August 14, 2026`)
3. Time window (for example, `6:00 AM - 8:00 AM`)
4. Header row: `Time, Jeepney Route, Stopped, Passengers, Fullness, Waiting`
5. Observation rows

Only rows where `Stopped` is `Y` contribute boarding counts. The parser normalizes common spelling variants to these canonical values:

- Nodes: `ROTUNDA`, `PACIFIC MALL`, `DARAGA MARKET`, `EMBARCADERO`, `SM`
- Windows: `AM`, `MID`, `PM`
- Routes: `DIRETSO A`, `DIRETSO B`, `RAWIS A`, `RAWIS B`, `ARIMBAY`, `LOOP 1`, `LOOP 2`, `EXTERNAL`

## Normalized output

The parser creates `data/processed/boarding_counts.csv` with:

```text
date,node,window,route,passengers_boarding,obs_duration_min
```

Do not hand-edit generated output. Correct the raw survey file, rerun the pipeline, and retain the original source file for auditability.
