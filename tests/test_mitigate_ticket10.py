"""Ticket 10: local_deterministic Mitigation Strategy (deskew).

Exercises the real ``aiai-ocr-dataset/D-GEO`` fixture family, deliberately
skewed, against the real clean ``D-COMP`` fixture already sealed by ticket
02. No CER scoring pipeline exists yet (a later ticket), so per the ticket
10 brief this test substitutes a quantitative image-level proxy -- the
detected skew angle before/after the transform -- for a CER comparison, and
otherwise proves the two acceptance-critical shape properties: (a) baseline
and mitigated are two distinct, independently addressable ``PageInputVariant``
records that can coexist under one implied run context with no second
manifest object anywhere in the code path, and (b) an input needing no
correction gets no mitigated variant and a recorded reason instead of a
guessed transform.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from ocr_ensemble.mitigate import (
    MAX_CORRECTABLE_SKEW_DEGREES,
    MIN_CORRECTABLE_SKEW_DEGREES,
    DeskewStrategy,
    MitigationOutcome,
)
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "aiai-ocr-dataset"

SKEWED_FIXTURE = "D-GEO/ocr_d-geo_medium-1_20260713.png"
STRAIGHT_FIXTURE = "D-COMP/ocr_d-comp_low-1_20260706.png"


@pytest.fixture()
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "preprocess-out"


def _seal_baseline(fixture_relative_path: str, output_root: Path):
    request = PreprocessRequest(
        dataset_root=DATASET_ROOT,
        output_root=output_root,
        fixture_relative_paths=(fixture_relative_path,),
    )
    artifacts = preprocess_dataset(request)
    return artifacts.evaluation_units[0], artifacts.input_variants[0]


def test_fixtures_exist():
    assert (DATASET_ROOT / SKEWED_FIXTURE).is_file()
    assert (DATASET_ROOT / STRAIGHT_FIXTURE).is_file()


def test_deskew_produces_distinct_mitigated_variant_for_skewed_fixture(
    output_root: Path,
):
    evaluation_unit, baseline = _seal_baseline(SKEWED_FIXTURE, output_root)

    outcome = DeskewStrategy().apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )

    assert isinstance(outcome, MitigationOutcome)
    assert outcome.no_correction_reason is None
    mitigated = outcome.variant
    assert mitigated is not None

    assert mitigated.variant_kind == "mitigated"
    assert mitigated.mitigation_kind == "local_deterministic"
    assert mitigated.page_input_variant_id != baseline.page_input_variant_id
    assert mitigated.page_input_variant_id.startswith("page_input_variant_sha256:")

    assert mitigated.evaluation_unit_id == baseline.evaluation_unit_id
    assert mitigated.source_page_id == baseline.source_page_id
    assert mitigated.decode_provenance_id == baseline.decode_provenance_id

    assert len(mitigated.transform_chain) == 1
    step = mitigated.transform_chain[0]
    assert step.algorithm_id == "deskew_projection_profile_cv2"
    assert step.parameters["detected_skew_degrees"] == outcome.detected_skew_degrees
    assert step.parameters["corrected_skew_degrees"] == outcome.corrected_skew_degrees
    assert step.parameters["output_width_px"] == mitigated.width_px
    assert step.parameters["output_height_px"] == mitigated.height_px
    assert step.parameters["resampling_method"]
    assert step.parameters["rotation_matrix"]


def test_deskew_quantitatively_reduces_skew_versus_baseline(output_root: Path):
    """Stronger-than-CER proxy (ticket 10 fallback: CER scoring does not exist
    yet). The mitigated variant's detected residual skew must be closer to
    zero than the baseline's originally detected skew.
    """
    evaluation_unit, baseline = _seal_baseline(SKEWED_FIXTURE, output_root)
    outcome = DeskewStrategy().apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )

    assert outcome.variant is not None
    assert abs(outcome.detected_skew_degrees) >= MIN_CORRECTABLE_SKEW_DEGREES
    assert abs(outcome.corrected_skew_degrees) < abs(outcome.detected_skew_degrees)


def test_baseline_variant_is_never_overwritten_by_mitigation(output_root: Path):
    evaluation_unit, baseline = _seal_baseline(SKEWED_FIXTURE, output_root)
    baseline_snapshot = dataclasses.replace(baseline)

    outcome = DeskewStrategy().apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )

    assert outcome.variant is not None
    assert baseline == baseline_snapshot
    assert baseline.variant_kind == "natural_baseline"
    assert baseline.mitigation_kind is None

    # baseline's sealed store on disk is untouched by mitigation (mitigation
    # never writes source_pages.jsonl/evaluation_units.jsonl/input_variants.jsonl,
    # only its own new artifact file).
    from ocr_ensemble.storage import read_jsonl

    sealed_baselines = read_jsonl(output_root / "input_variants.jsonl")
    assert len(sealed_baselines) == 1
    assert sealed_baselines[0]["page_input_variant_id"] == baseline.page_input_variant_id
    assert sealed_baselines[0]["variant_kind"] == "natural_baseline"


def test_mitigation_dispatches_without_any_paired_experiment_concept(output_root: Path):
    """local_deterministic dispatches within the same Run Manifest
    as baseline -- no PairedExperiment is created or required. Run Manifest
    and dispatch do not exist yet in this codebase (tickets 03+), so this
    proves the negative available today: nothing in the mitigate module or
    its outcome references a paired-manifest concept, and the same implied
    manifest/run context (identical evaluation_unit_id/source_page_id) is
    shared by both variants with no second manifest object constructed
    anywhere in this code path.
    """
    import ocr_ensemble.mitigate as mitigate_module

    # The module may *discuss* PairedExperiment in prose (explaining what it
    # deliberately does not depend on), so this checks for an actual runtime
    # dependency -- an import or a live module attribute -- not a substring
    # ban on comments/docstrings.
    assert not hasattr(mitigate_module, "PairedExperiment")
    assert "PairedExperiment" not in dir(mitigate_module)
    module_globals_with_paired_experiment = [
        name
        for name, value in vars(mitigate_module).items()
        if "paired" in name.lower() and "experiment" in name.lower()
    ]
    assert module_globals_with_paired_experiment == []

    evaluation_unit, baseline = _seal_baseline(SKEWED_FIXTURE, output_root)
    outcome = DeskewStrategy().apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )
    mitigated = outcome.variant
    assert mitigated is not None
    assert mitigated.evaluation_unit_id == baseline.evaluation_unit_id


def test_deskew_emits_no_variant_and_records_reason_for_already_straight_fixture(
    output_root: Path,
):
    evaluation_unit, baseline = _seal_baseline(STRAIGHT_FIXTURE, output_root)

    outcome = DeskewStrategy().apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )

    assert outcome.variant is None
    assert outcome.no_correction_reason is not None
    assert "straight" in outcome.no_correction_reason or "below" in outcome.no_correction_reason
    assert abs(outcome.detected_skew_degrees) < MIN_CORRECTABLE_SKEW_DEGREES
    assert outcome.corrected_skew_degrees is None


def test_deskew_rejects_a_baseline_that_is_not_natural_baseline_kind(output_root: Path):
    evaluation_unit, baseline = _seal_baseline(SKEWED_FIXTURE, output_root)
    fake_mitigated = dataclasses.replace(
        baseline, variant_kind="mitigated", mitigation_kind="local_deterministic"
    )

    with pytest.raises(ValueError):
        DeskewStrategy().apply(
            baseline_variant=fake_mitigated,
            evaluation_unit=evaluation_unit,
            output_root=output_root,
        )


def test_mitigated_variant_json_serializable_and_id_is_reproducible(output_root: Path):
    evaluation_unit, baseline = _seal_baseline(SKEWED_FIXTURE, output_root)
    strategy = DeskewStrategy()

    first = strategy.apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )
    second = strategy.apply(
        baseline_variant=baseline, evaluation_unit=evaluation_unit, output_root=output_root
    )

    assert first.variant is not None and second.variant is not None
    assert first.variant.page_input_variant_id == second.variant.page_input_variant_id

    serialized = dataclasses.asdict(first.variant)
    json.dumps(serialized)  # must not raise


def test_mitigation_outcome_rejects_both_variant_and_reason_set():
    with pytest.raises(ValueError):
        MitigationOutcome(
            variant=object(),  # type: ignore[arg-type]
            no_correction_reason="both set is invalid",
            detected_skew_degrees=1.0,
            corrected_skew_degrees=0.1,
        )


def test_mitigation_outcome_rejects_neither_variant_nor_reason_set():
    with pytest.raises(ValueError):
        MitigationOutcome(
            variant=None,
            no_correction_reason=None,
            detected_skew_degrees=1.0,
            corrected_skew_degrees=None,
        )
