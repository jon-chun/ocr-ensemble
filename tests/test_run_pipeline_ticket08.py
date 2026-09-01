"""Ticket 08: the run orchestrator that lays out runs/<run_id>/ end to end
(the required stores), closing the gap that manifest.py/
ground_truth.py/align.py's consensus+fusion outputs had no disk-persistence
path and no single driver wired them together.
"""

from __future__ import annotations

from pathlib import Path

from ocr_ensemble.run_pipeline import DEFAULT_DATASET_ROOT, cli_main, run_one_page_pipeline
from ocr_ensemble.storage import read_json, read_jsonl

REQUIRED_JSONL_STORES = (
    "dispatch_intents.jsonl",
    "attempt_events.jsonl",
    "page_model_results.jsonl",
    "consensus_results.jsonl",
    "fused_hypotheses.jsonl",
    "benchmark_observations.jsonl",
)
REQUIRED_JSON_STORES = ("run_manifest.json", "effective_gt_snapshot.json")


def test_pipeline_writes_every_required_store(tmp_path: Path):
    output_root = tmp_path / "runs" / "test-run"
    result = run_one_page_pipeline(output_root=output_root)

    assert result.run_id.startswith("run_manifest_sha256:")
    for name in REQUIRED_JSONL_STORES:
        assert (output_root / name).exists(), f"missing {name}"
    for name in REQUIRED_JSON_STORES:
        assert (output_root / name).exists(), f"missing {name}"


def test_pipeline_produces_two_page_model_results_and_a_quorum_2_fusion(tmp_path: Path):
    output_root = tmp_path / "runs" / "test-run"
    run_one_page_pipeline(output_root=output_root)

    results = read_jsonl(output_root / "page_model_results.jsonl")
    assert len(results) == 2
    assert all(r["terminal_outcome"] == "success" for r in results)

    consensus = read_jsonl(output_root / "consensus_results.jsonl")
    assert len(consensus) == 1
    assert consensus[0]["quorum_size"] == 2
    assert consensus[0]["eligible_hypothesis_count"] == 2
    assert consensus[0]["consensus_entropy"] is not None

    fused = read_jsonl(output_root / "fused_hypotheses.jsonl")
    assert len(fused) == 1
    assert fused[0]["consensus_result_id"] == consensus[0]["consensus_result_id"]


def test_pipeline_produces_member_and_fused_benchmark_observations(tmp_path: Path):
    output_root = tmp_path / "runs" / "test-run"
    run_one_page_pipeline(output_root=output_root)

    observations = read_jsonl(output_root / "benchmark_observations.jsonl")
    subject_kinds = [o["subject_kind"] for o in observations]
    assert subject_kinds.count("page_model_result") == 2
    assert subject_kinds.count("fused_hypothesis") == 1

    for obs in observations:
        assert obs["run_manifest_id"]
        assert obs["gt_snapshot_id"]
        assert obs["cer"] is not None


def test_run_manifest_json_round_trips_via_read_json(tmp_path: Path):
    output_root = tmp_path / "runs" / "test-run"
    result = run_one_page_pipeline(output_root=output_root)

    manifest_doc = read_json(output_root / "run_manifest.json")
    assert manifest_doc is not None
    assert manifest_doc["run_manifest_id"] == result.run_id
    assert len(manifest_doc["roster"]) == 2


def test_pipeline_is_deterministic_across_repeated_runs(tmp_path: Path):
    root_a = tmp_path / "run-a"
    root_b = tmp_path / "run-b"
    result_a = run_one_page_pipeline(output_root=root_a)
    result_b = run_one_page_pipeline(output_root=root_b)

    assert result_a.run_id == result_b.run_id

    obs_a = read_jsonl(root_a / "benchmark_observations.jsonl")
    obs_b = read_jsonl(root_b / "benchmark_observations.jsonl")
    ids_a = sorted(o["benchmark_observation_id"] for o in obs_a)
    ids_b = sorted(o["benchmark_observation_id"] for o in obs_b)
    assert ids_a == ids_b


def test_cli_main_runs_end_to_end_and_exits_zero(tmp_path: Path, capsys):
    output_root = tmp_path / "runs" / "cli-run"
    exit_code = cli_main(
        ["--output-root", str(output_root), "--dataset-root", str(DEFAULT_DATASET_ROOT)]
    )
    assert exit_code == 0
    assert (output_root / "run_manifest.json").exists()
    captured = capsys.readouterr()
    assert "run_id:" in captured.out
