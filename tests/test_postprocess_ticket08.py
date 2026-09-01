"""A8: postprocess structural validation (postprocess spec v2 §4-6; ticket
08). Exercises a real run_pipeline.py output, both clean and deliberately
corrupted, since postprocess's whole job is catching exactly this class of
integrity failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_ensemble.postprocess import (
    PostprocessRequest,
    cli_main,
    validate_run,
    write_postprocess_report,
)
from ocr_ensemble.run_pipeline import run_one_page_pipeline
from ocr_ensemble.storage import read_json


@pytest.fixture()
def clean_run(tmp_path: Path) -> Path:
    output_root = tmp_path / "runs" / "test-run"
    run_one_page_pipeline(output_root=output_root)
    return output_root


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def test_clean_run_passes_with_no_findings(clean_run: Path):
    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    assert artifacts.gate_status == "pass"
    assert artifacts.run_completion_status == "complete"
    assert artifacts.findings == ()
    assert artifacts.expected_pair_count == 2
    assert artifacts.observed_result_count == 2


def test_missing_page_model_result_fails_the_gate(clean_run: Path):
    path = clean_run / "page_model_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _rewrite_jsonl(path, rows[:1])

    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    assert artifacts.gate_status == "fail"
    codes = {f.code for f in artifacts.findings}
    assert "missing_page_model_result" in codes


def test_duplicate_page_model_result_fails_the_gate(clean_run: Path):
    path = clean_run / "page_model_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _rewrite_jsonl(path, rows + [rows[0]])

    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    assert artifacts.gate_status == "fail"
    codes = {f.code for f in artifacts.findings}
    assert "duplicate_page_model_result" in codes


def test_orphan_page_model_result_fails_the_gate(clean_run: Path):
    path = clean_run / "page_model_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    orphan = dict(rows[0])
    orphan["page_model_result_id"] = "page_model_result_sha256:" + "f" * 64
    orphan["dispatch_pair_id"] = "dispatch_pair_sha256:" + "e" * 64
    _rewrite_jsonl(path, rows + [orphan])

    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    assert artifacts.gate_status == "fail"
    codes = {f.code for f in artifacts.findings}
    assert "orphan_page_model_result" in codes


def test_ce_reproducibility_mismatch_fails_the_gate(clean_run: Path):
    path = clean_run / "consensus_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["consensus_entropy"] = 0.999
    _rewrite_jsonl(path, rows)

    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    assert artifacts.gate_status == "fail"
    codes = {f.code for f in artifacts.findings}
    assert "ce_reproducibility_mismatch" in codes


def test_orphan_attempt_wrong_dispatch_intent_is_not_silently_accepted(clean_run: Path):
    path = clean_run / "attempt_events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    duplicate_start = dict(rows[0])
    # a second attempt_started for the same attempt_id is impossible under
    # honest dispatch; verifies postprocess doesn't just count events blindly.
    _rewrite_jsonl(path, rows + [duplicate_start])

    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    assert artifacts.gate_status == "fail"
    codes = {f.code for f in artifacts.findings}
    assert "duplicate_attempt_id" in codes


def test_missing_run_manifest_fails_and_is_incomplete(tmp_path: Path):
    empty_root = tmp_path / "runs" / "empty-run"
    empty_root.mkdir(parents=True)
    artifacts = validate_run(PostprocessRequest(run_root=empty_root))
    assert artifacts.gate_status == "fail"
    assert artifacts.run_completion_status == "incomplete"


def test_write_postprocess_report_round_trips_via_atomic_json(clean_run: Path):
    artifacts = validate_run(PostprocessRequest(run_root=clean_run))
    write_postprocess_report(clean_run, artifacts)

    report = read_json(clean_run / "postprocess_report.json")
    assert report is not None
    assert report["gate_status"] == "pass"
    assert report["findings"] == []


def test_cli_main_exits_zero_and_writes_report_on_a_clean_run(clean_run: Path):
    exit_code = cli_main(["--run-root", str(clean_run)])
    assert exit_code == 0
    assert (clean_run / "postprocess_report.json").exists()


def test_cli_main_exits_three_on_integrity_failure(clean_run: Path):
    path = clean_run / "page_model_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _rewrite_jsonl(path, rows[:1])

    exit_code = cli_main(["--run-root", str(clean_run)])
    assert exit_code == 3
