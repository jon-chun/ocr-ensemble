"""A9: minimal one-page analyze report (analyze spec v2 §2-3; ticket 08).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_ensemble.align import ConsensusEntropyBandsConfig
from ocr_ensemble.analyze import (
    AnalyzeRequest,
    RunNotAnalyzable,
    analyze_run,
    cli_main,
    render_report,
)
from ocr_ensemble.postprocess import PostprocessRequest, validate_run, write_postprocess_report
from ocr_ensemble.run_pipeline import run_one_page_pipeline


@pytest.fixture()
def validated_run(tmp_path: Path) -> Path:
    output_root = tmp_path / "runs" / "test-run"
    run_one_page_pipeline(output_root=output_root)
    artifacts = validate_run(PostprocessRequest(run_root=output_root))
    write_postprocess_report(output_root, artifacts)
    return output_root


def test_analyze_reports_cer_wer_ce_and_band_for_every_observation(validated_run: Path):
    artifacts = analyze_run(AnalyzeRequest(run_root=validated_run))
    assert artifacts.postprocess_gate_status == "pass"
    assert len(artifacts.observations) == 3

    fused = [o for o in artifacts.observations if o.subject_kind == "fused_hypothesis"]
    assert len(fused) == 1
    assert fused[0].cer is not None
    assert fused[0].wer is not None
    assert fused[0].consensus_entropy is not None
    assert fused[0].confidence_band in ("high_confidence", "medium_confidence", "low_confidence")

    members = [o for o in artifacts.observations if o.subject_kind == "page_model_result"]
    assert len(members) == 2
    # member observations have no CE/band -- that is a fused-level statistic
    for member in members:
        assert member.consensus_entropy is None
        assert member.confidence_band is None


def test_analyze_refuses_a_run_with_no_postprocess_report(tmp_path: Path):
    output_root = tmp_path / "runs" / "unvalidated-run"
    run_one_page_pipeline(output_root=output_root)

    with pytest.raises(RunNotAnalyzable):
        analyze_run(AnalyzeRequest(run_root=output_root))


def test_analyze_refuses_a_run_with_a_failed_postprocess_gate(tmp_path: Path):
    output_root = tmp_path / "runs" / "failed-run"
    run_one_page_pipeline(output_root=output_root)
    write_postprocess_report(
        output_root,
        validate_run(PostprocessRequest(run_root=output_root)),
    )
    # tamper the written report to simulate a failed gate
    import json

    report_path = output_root / "postprocess_report.json"
    doc = json.loads(report_path.read_text())
    doc["gate_status"] = "fail"
    report_path.write_text(json.dumps(doc))

    with pytest.raises(RunNotAnalyzable):
        analyze_run(AnalyzeRequest(run_root=output_root))


def test_band_boundaries_use_the_configured_thresholds(validated_run: Path):
    strict_bands = ConsensusEntropyBandsConfig(
        ce_low_uncertainty_max=0.01, ce_high_uncertainty_min=0.02
    )
    artifacts = analyze_run(AnalyzeRequest(run_root=validated_run, ce_bands=strict_bands))
    fused = [o for o in artifacts.observations if o.subject_kind == "fused_hypothesis"][0]
    # the fixture's real CE (~0.05) now falls above the stricter high threshold
    assert fused.confidence_band == "low_confidence"


def test_render_report_includes_run_id_gate_and_every_row(validated_run: Path):
    artifacts = analyze_run(AnalyzeRequest(run_root=validated_run))
    text = render_report(artifacts)
    assert artifacts.run_manifest_id in text
    assert "pass" in text
    assert text.count("\n") >= 5


def test_cli_main_exits_zero_on_a_validated_run(validated_run: Path, capsys):
    exit_code = cli_main(["--run-root", str(validated_run)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "CER" in captured.out


def test_cli_main_exits_nonzero_when_not_analyzable(tmp_path: Path, capsys):
    output_root = tmp_path / "runs" / "unvalidated-run"
    run_one_page_pipeline(output_root=output_root)
    exit_code = cli_main(["--run-root", str(output_root)])
    assert exit_code == 3
