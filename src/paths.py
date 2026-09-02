"""Filesystem locations used by the reproducible analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve repository paths without depending on the current directory."""

    root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        return cls(Path(__file__).resolve().parents[1])

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_data(self) -> Path:
        """Preferred raw-data location, with support for the legacy layout."""
        preferred = self.data / "raw"
        return preferred if any(preferred.glob("DATA-*.csv")) else self.data

    @property
    def processed_data(self) -> Path:
        return self.data / "processed" / "boarding_counts.csv"

    @property
    def reference_data(self) -> Path:
        return self.data / "reference"

    @property
    def lptrp_profile(self) -> Path:
        return self.reference_data / "lptrp.json"

    @property
    def route_geometries(self) -> Path:
        return self.reference_data / "routes.geojson"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def figures(self) -> Path:
        return self.root / "output" / "figures"

    @property
    def maps(self) -> Path:
        return self.root / "output" / "maps"

    @property
    def tables(self) -> Path:
        return self.root / "output" / "tables"

    @property
    def reports(self) -> Path:
        return self.root / "output" / "reports"

    def ensure_output_directories(self) -> None:
        for directory in (self.figures, self.maps, self.tables, self.reports):
            directory.mkdir(parents=True, exist_ok=True)
