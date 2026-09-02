import json
import random
import csv
import os

# Ensure directories exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data', exist_ok=True)

NODES = ['PacificMall', 'RizalQuezon', 'DaragaMarket', 'Embarcadero', 'LGCT']
WINDOWS = ['AM', 'MID', 'PM']
ROUTES = ['DL', 'AL', 'DRA', 'DRB', 'L1', 'L2', 'CL', 'GL']
DATES = ['2026-08-01', '2026-08-02', '2026-08-03']

print("Generating mock boarding_counts.csv...")
with open('data/processed/boarding_counts.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['date', 'node', 'window', 'route', 'passengers_boarding', 'obs_duration_min'])
    for d in DATES:
        for node in NODES:
            for w in WINDOWS:
                for r in ROUTES:
                    # Random passenger count between 5 and 50
                    pax = random.randint(5, 50)
                    writer.writerow([d, node, w, r, pax, 60])

print("Generating mock routes.geojson...")
features = []
for idx, r in enumerate(ROUTES):
    # Just a small line segment around Legazpi center (13.14, 123.73)
    offset = idx * 0.005
    feature = {
        "type": "Feature",
        "properties": {
            "route_code": r,
            "route_name": f"Mock Route {r}"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [123.73 + offset, 13.14 + offset],
                [123.74 + offset, 13.15 + offset]
            ]
        }
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open('data/routes.geojson', 'w') as f:
    json.dump(geojson, f, indent=2)

# Also generate a mock LPTRP plan
print("Generating mock lptrp.json...")
lptrp = {
    "route_assignments": {
        "DL": 10, "AL": 10, "DRA": 10, "DRB": 10,
        "L1": 10, "L2": 10, "CL": 10, "GL": 10
    }
}
with open('data/lptrp.json', 'w') as f:
    json.dump(lptrp, f, indent=2)

print("Mock data generated successfully!")
