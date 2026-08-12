import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_graphrag_mapping_preserves_all_raw_outputs():
    mapping = _load("graphrag/candidate_mapping.json")
    no_invention = _load("graphrag/outputs/no_invention_candidates_20260114.json")
    two_mode = _load("graphrag/outputs/two_mode_candidates_20260119.json")

    expected_ids = {
        candidate["id"] for candidate in no_invention["se_candidates"]
    }
    expected_ids.update(
        candidate["id"]
        for group in (
            "se_candidates_prior_only",
            "se_candidates_context_plus_creativity",
        )
        for candidate in two_mode[group]
    )
    actual_ids = [candidate["output_id"] for candidate in mapping["candidates"]]

    assert len(actual_ids) == 29
    assert len(actual_ids) == len(set(actual_ids))
    assert set(actual_ids) == expected_ids


def test_every_paper_mapping_resolves_uniquely_to_release_registry():
    mapping = _load("graphrag/candidate_mapping.json")
    registry = _load("artifacts/candidates.json")["candidates"]

    by_paper_row = {}
    for candidate in registry:
        for table_key, row in candidate.get("paper_rows", {}).items():
            if table_key not in {"table_1", "table_3"}:
                continue
            key = (int(table_key.removeprefix("table_")), int(row))
            assert key not in by_paper_row
            by_paper_row[key] = candidate

    linked = 0
    for raw_candidate in mapping["candidates"]:
        paper = raw_candidate.get("paper_mapping")
        if paper is None:
            continue
        linked += 1
        key = (int(paper["table"]), int(paper["row"]))
        release_candidate = by_paper_row[key]
        assert paper["release_candidate_id"] == release_candidate["id"]
        assert paper["release_solver_function"] == "udiscovery.equations.evaluate"

    assert linked == 23


def test_selected_proposals_cover_the_paper_rows():
    mapping = _load("graphrag/candidate_mapping.json")
    selected = {
        (entry["paper_mapping"]["table"], entry["paper_mapping"]["row"])
        for entry in mapping["candidates"]
        if (entry.get("paper_mapping") or {}).get("relation") == "selected"
    }

    expected = {(1, row) for row in range(6, 12)}
    expected.update((3, row) for row in range(1, 16))
    assert selected == expected
