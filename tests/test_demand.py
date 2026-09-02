import csv

from src.data_parser import build_boarding_counts
from src.demand import build_demand_tensor, load_field_sheets


def test_parser_builds_normalized_boarding_counts(tmp_path):
    raw_file = tmp_path / "DATA-PACIFIC.csv"
    with raw_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [
                ["PACIFIC MALL"],
                ["August 14, 2026"],
                ["6:00 AM - 8:00 AM"],
                ["Time", "Jeepney Route", "Stopped", "Passengers", "Fullness", "Waiting"],
                ["06:00", "DIRETSO", "Y", "4", "", ""],
            ]
        )
    output_file = tmp_path / "boarding_counts.csv"
    rows = build_boarding_counts(str(tmp_path), str(output_file))
    assert len(rows) == 1
    field_data = load_field_sheets(output_file)
    demand = build_demand_tensor(field_data)
    assert demand[1, 0, 0] == 32  # 4 passengers in a 15-minute observation, scaled to 120 minutes
