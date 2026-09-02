"""Regression tests for the project-level workflow contracts."""

from generate_mock_data import write_mock_survey
from src.config import MODEL_ROUTES, NODES, ROUTES, WINDOWS
from src.data_parser import build_boarding_counts
from src.demand import build_demand_tensor, load_field_sheets
from src.paths import ProjectPaths


def test_model_routes_are_the_non_external_canonical_routes():
    assert MODEL_ROUTES == ROUTES[:-1]
    assert "EXTERNAL" not in MODEL_ROUTES


def test_mock_survey_is_parser_compatible(tmp_path):
    raw_dir = tmp_path / "raw"
    write_mock_survey(raw_dir)
    normalized = tmp_path / "processed" / "boarding_counts.csv"

    rows = build_boarding_counts(raw_dir, normalized)
    demand = build_demand_tensor(load_field_sheets(normalized))

    assert len(rows) == 3 * len(NODES) * len(WINDOWS) * len(ROUTES)
    assert demand.shape == (len(NODES), len(ROUTES), len(WINDOWS))
    assert demand.sum() > 0


def test_project_paths_are_root_relative():
    paths = ProjectPaths.discover()
    assert paths.data == paths.root / "data"
    assert paths.processed_data == paths.data / "processed" / "boarding_counts.csv"
    assert paths.lptrp_profile == paths.reference_data / "lptrp.json"
