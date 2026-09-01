"""A9 analyze: read-only reporting over validated run artifacts (analyze
spec v2; ticket 08).

Ticket 08 scope: the minimal one-page report -- CER, WER, CE, and confidence
band for the fixture page, gated on a passing postprocess report (spec §3:
analyze's preconditions include the postprocess report). The full report
(annotation/audit queue summaries, CE-per-model-pair, Scientific Acceptance
Scorecard, run comparison, charts) lands across tickets 12-15; this module's
``analyze_run``/``cli_main`` are written so those tickets extend the same
entry points rather than replace them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ocr_ensemble.align import ConsensusEntropyBandsConfig
from ocr_ensemble.storage import read_json, read_jsonl

DEFAULT_CE_BANDS = ConsensusEntropyBandsConfig(
    ce_low_uncertainty_max=0.10, ce_high_uncertainty_min=0.30
)


class RunNotAnalyzable(Exception):
    """Raised when the run's postprocess report is missing or not
    ``gate_status="pass"`` (analyze spec §3's precondition); analyze never
    reports on a run postprocess has not cleared.
    """


@dataclass(frozen=True)
class ObservationSummary:
    subject_kind: str
    subject_id: str
    cer: float | None
    wer: float | None
    consensus_entropy: float | None
    confidence_band: str | None


@dataclass(frozen=True)
class AnalysisArtifacts:
    run_manifest_id: str
    postprocess_gate_status: str
    observations: tuple[ObservationSummary, ...]


@dataclass(frozen=True)
class AnalyzeRequest:
    run_root: Path
    ce_bands: ConsensusEntropyBandsConfig = DEFAULT_CE_BANDS


def analyze_run(request: AnalyzeRequest) -> AnalysisArtifacts:
    manifest = read_json(request.run_root / "run_manifest.json")
    if manifest is None:
        raise RunNotAnalyzable(f"no run_manifest.json under {request.run_root}")

    report = read_json(request.run_root / "postprocess_report.json")
    if report is None:
        raise RunNotAnalyzable(
            f"no postprocess_report.json under {request.run_root}; run postprocess first"
        )
    if report["gate_status"] != "pass":
        raise RunNotAnalyzable(
            f"postprocess gate_status={report['gate_status']!r}, not 'pass'; "
            "analyze does not report on a failed or unvalidated run"
        )

    observations = read_jsonl(request.run_root / "benchmark_observations.jsonl")
    summaries = []
    for obs in observations:
        ce = obs["consensus_entropy"]
        band = request.ce_bands.band_for(ce) if ce is not None else None
        summaries.append(
            ObservationSummary(
                subject_kind=obs["subject_kind"],
                subject_id=obs["subject_id"],
                cer=obs["cer"],
                wer=obs["wer"],
                consensus_entropy=ce,
                confidence_band=band,
            )
        )

    return AnalysisArtifacts(
        run_manifest_id=manifest["run_manifest_id"],
        postprocess_gate_status=report["gate_status"],
        observations=tuple(summaries),
    )


def render_report(artifacts: AnalysisArtifacts) -> str:
    lines = [
        f"Run: {artifacts.run_manifest_id}",
        f"Postprocess gate: {artifacts.postprocess_gate_status}",
        "",
        f"{'subject_kind':<20}{'CER':>10}{'WER':>10}{'CE':>10}  {'band'}",
    ]
    for obs in artifacts.observations:
        cer_str = f"{obs.cer:.4f}" if obs.cer is not None else "n/a"
        wer_str = f"{obs.wer:.4f}" if obs.wer is not None else "n/a"
        ce_str = f"{obs.consensus_entropy:.4f}" if obs.consensus_entropy is not None else "n/a"
        band_str = obs.confidence_band or "n/a"
        lines.append(f"{obs.subject_kind:<20}{cer_str:>10}{wer_str:>10}{ce_str:>10}  {band_str}")
    return "\n".join(lines) + "\n"


def cli_main(argv: list[str] | None = None) -> int:
    """Ticket 08 scope: one ``--run-root`` pointing at a ``runs/<run_id>/``
    directory ``run_pipeline.py`` (or a future full orchestrator) produced.
    The individually-enumerated per-store flags in the utils-spec's CLI
    contract are deferred to the ticket that widens beyond one directory's
    worth of stores per invocation.
    """
    parser = argparse.ArgumentParser(prog="ocr-ensemble-fix-analyze")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        artifacts = analyze_run(AnalyzeRequest(run_root=args.run_root))
    except RunNotAnalyzable as exc:
        print(f"error: {exc}")
        return 3

    print(render_report(artifacts), end="")
    return 0
