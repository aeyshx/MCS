"""Create a small, canonical survey dataset for pipeline demonstrations.

The generated file is kept out of ``data/raw`` so it cannot be mistaken for
field data. Use it with ``GAME_THEORY_RAW_DIR=data/mock``.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.config import NODES, ROUTES, WINDOWS


WINDOW_LABELS = {
    "AM": "6:00 AM - 8:00 AM",
    "MID": "11:00 AM - 1:00 PM",
    "PM": "4:00 PM - 6:00 PM",
}


def write_mock_survey(output_directory: Path, seed: int = 42) -> Path:
    """Write deterministic, parser-compatible observations and return the file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "DATA-MOCK.csv"
    random_source = random.Random(seed)
    dates = ("August 1, 2026", "August 2, 2026", "August 3, 2026")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for date in dates:
            for node in NODES:
                for window in WINDOWS:
                    writer.writerows(
                        [
                            [node],
                            [date],
                            [WINDOW_LABELS[window]],
                            ["Time", "Jeepney Route", "Stopped", "Passengers", "Fullness", "Waiting"],
                        ]
                    )
                    for route in ROUTES:
                        writer.writerow(["08:00", route, "Y", random_source.randint(5, 50), "", ""])
                    writer.writerow([])

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mock"))
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    output = write_mock_survey(arguments.output_dir, arguments.seed)
    print(f"Mock field-survey file written to {output}")


if __name__ == "__main__":
    main()
