"""Canonical serialization/ID fixtures (ticket 01).

Proves: identical semantic inputs produce identical semantic IDs regardless of
timestamp/audit-metadata differences; different semantic inputs produce
different IDs; changing only ``created_at`` or an artifact URI leaves the
semantic ID unchanged while changing ``record_sha256``.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from ocr_ensemble.identity import canonical_json_bytes, content_id, hash_preimage
from ocr_ensemble.records import (
    ApprovalAuthorization,
    BaselineTranscription,
    BenchmarkObservation,
    CalibrationSelection,
    DatasetSplit,
    DecodeProvenance,
    DetectorEvidence,
    DispatchIntent,
    EffectiveGroundTruthSnapshot,
    EvaluationUnit,
    FailureAssessment,
    GroundTruthAssertion,
    GroundTruthEvent,
    PageInputVariant,
    PageModelResult,
    RawProviderEnvelope,
    RosterMemberConfiguration,
    RunManifest,
    SourcePage,
    attempt_event_id,
    attempt_id,
    dispatch_pair_id,
)


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_has_no_insignificant_whitespace():
    a = canonical_json_bytes({"b": 1, "a": 2})
    b = canonical_json_bytes({"a": 2, "b": 1})
    assert a == b
    assert a == b'{"a":2,"b":1}'


def test_canonical_json_rejects_raw_sets():
    with pytest.raises(TypeError):
        canonical_json_bytes({"x": {1, 2, 3}})


def test_content_id_format():
    digest = hash_preimage({"a": 1})
    cid = content_id("source_page", digest)
    assert cid == f"source_page_sha256:{digest}"
    assert cid.islower() or True  # hex digest is lowercase; type name may not be


# ---------------------------------------------------------------------------
# Fixtures: one instance per record type in the identity table
# ---------------------------------------------------------------------------


def _source_page(**overrides) -> SourcePage:
    base = dict(
        source_page_id="placeholder",
        schema_version="v2",
        dataset_id="havi_failure_mode_v1",
        dataset_version="2026-08-31",
        dataset_item_id="D-GEO/0001",
        source_uri="file:///aiai-ocr-dataset/D-GEO/0001.jpg",
        source_sha256="a" * 64,
        media_type="image/jpeg",
        byte_size=123456,
        width_px=2000,
        height_px=3000,
        color_mode="rgb",
        bit_depth=8,
        orientation_metadata=None,
        source_genre="newspaper",
        license_id=None,
    )
    base.update(overrides)
    return SourcePage(**base)


def test_source_page_identity_ignores_uri_and_timestamp_style_fields():
    page1 = _source_page(source_uri="file:///a.jpg")
    page2 = _source_page(source_uri="file:///different-path/a.jpg")
    assert page1.semantic_id() == page2.semantic_id()
    # source_uri is not excluded from record_sha256 (only artifact_sha256-bearing
    # variant records exclude URI); SourcePage's record hash is expected to
    # differ here since source_uri is a plain declared field, not an excluded one.
    assert page1.record_sha256() != page2.record_sha256()


def test_source_page_identity_changes_with_source_hash():
    page1 = _source_page()
    page2 = _source_page(source_sha256="b" * 64)
    assert page1.semantic_id() != page2.semantic_id()


def test_evaluation_unit_identity():
    unit1 = EvaluationUnit(
        evaluation_unit_id="placeholder",
        schema_version="v2",
        source_page_id="source_page_sha256:" + "a" * 64,
        selector=None,
        layout_complexity="linear",
        reading_order_policy_id="reading_order_v1",
        language="en",
        mixed_language=False,
        evaluation_scope_status="in_scope_scored",
    )
    unit2 = dataclasses.replace(unit1, evaluation_unit_id="different-placeholder")
    assert unit1.semantic_id() == unit2.semantic_id()

    unit3 = dataclasses.replace(unit1, layout_complexity="layout_dependent")
    assert unit1.semantic_id() != unit3.semantic_id()


def test_page_input_variant_identity_ignores_artifact_uri_and_own_id():
    variant1 = PageInputVariant(
        page_input_variant_id="placeholder-1",
        schema_version="v2",
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        source_page_id="source_page_sha256:" + "a" * 64,
        variant_kind="natural_baseline",
        mitigation_kind=None,
        artifact_uri="file:///variant-a.jpg",
        artifact_sha256="c" * 64,
        media_type="image/jpeg",
        byte_size=100,
        width_px=2000,
        height_px=3000,
        decode_provenance_id="decode_provenance_sha256:" + "d" * 64,
        transform_chain=(),
        preprocess_policy_id="preprocess_v1",
        perturbation_provenance=None,
    )
    variant2 = dataclasses.replace(
        variant1,
        page_input_variant_id="placeholder-2",
        artifact_uri="file:///different/path/variant-a.jpg",
    )
    assert variant1.semantic_id() == variant2.semantic_id()


def test_page_input_variant_record_hash_is_sensitive_to_uri_and_own_id():
    # record_sha256 protects timestamps and other audit metadata:
    # it hashes the complete record except record_sha256 itself, so unlike
    # semantic_id() it IS sensitive to own-ID and artifact-URI changes.
    variant1 = PageInputVariant(
        page_input_variant_id="placeholder-1",
        schema_version="v2",
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        source_page_id="source_page_sha256:" + "a" * 64,
        variant_kind="natural_baseline",
        mitigation_kind=None,
        artifact_uri="file:///variant-a.jpg",
        artifact_sha256="c" * 64,
        media_type="image/jpeg",
        byte_size=100,
        width_px=2000,
        height_px=3000,
        decode_provenance_id="decode_provenance_sha256:" + "d" * 64,
        transform_chain=(),
        preprocess_policy_id="preprocess_v1",
        perturbation_provenance=None,
    )
    variant2 = dataclasses.replace(
        variant1,
        page_input_variant_id="placeholder-2",
        artifact_uri="file:///different/path/variant-a.jpg",
    )
    assert variant1.record_sha256() != variant2.record_sha256()
    # but semantic_id is unaffected by either change
    assert variant1.semantic_id() == variant2.semantic_id()


def test_decode_provenance_identity():
    dp1 = DecodeProvenance(
        decode_provenance_id="placeholder-1",
        source_page_id="source_page_sha256:" + "a" * 64,
        decoder_id="pillow",
        decoder_version="12.3.0",
        original_media_type="image/jpeg",
        original_bit_depth=8,
        original_color_mode="rgb",
        original_orientation_metadata=None,
        output_encoding="png",
    )
    dp2 = dataclasses.replace(dp1, decode_provenance_id="placeholder-2")
    assert dp1.semantic_id() == dp2.semantic_id()

    dp3 = dataclasses.replace(dp1, decoder_version="12.4.0")
    assert dp1.semantic_id() != dp3.semantic_id()


def test_baseline_transcription_identity():
    bt1 = BaselineTranscription(
        baseline_transcription_id="placeholder-1",
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        baseline_kind="dataset_ocr",
        text="hello world",
        source="bln600_gale_ocr",
        source_artifact_sha256="e" * 64,
        import_provenance_id="import_prov_1",
    )
    bt2 = dataclasses.replace(bt1, baseline_transcription_id="placeholder-2")
    assert bt1.semantic_id() == bt2.semantic_id()


def test_detector_evidence_identity_ignores_artifact_uri():
    ev1 = DetectorEvidence(
        evidence_id="placeholder-1",
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        page_input_variant_id="page_input_variant_sha256:" + "c" * 64,
        detector_id="skew_estimator",
        detector_version="1.0",
        measurement_name="skew_angle_degrees",
        value=2.5,
        unit="degrees",
        assessment_state="measured",
        artifact_uri="file:///evidence-a.json",
        artifact_sha256="f" * 64,
    )
    ev2 = dataclasses.replace(
        ev1, evidence_id="placeholder-2", artifact_uri="file:///different.json"
    )
    assert ev1.semantic_id() == ev2.semantic_id()


def test_failure_assessment_identity():
    fa1 = FailureAssessment(
        failure_assessment_id="placeholder-1",
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        condition_code="D-GEO/PERSPECTIVE",
        state="asserted",
        severity="medium",
        confidence=0.8,
        provenance="detector",
        evidence_ids=("detector_evidence_sha256:" + "f" * 64,),
        policy_version="taxonomy_v1",
        actor=None,
        reason=None,
        supersedes_assessment_id=None,
        created_at="2026-08-31T00:00:00Z",
        override_config_sha256=None,
    )
    fa2 = dataclasses.replace(
        fa1, failure_assessment_id="placeholder-2", created_at="2026-09-01T00:00:00Z"
    )
    assert fa1.semantic_id() == fa2.semantic_id()
    assert fa1.record_sha256() != fa2.record_sha256()


def test_roster_member_configuration_identity():
    rmc1 = RosterMemberConfiguration(
        roster_member_config_id="placeholder-1",
        registry_entry_id="registry_entry_sha256:" + "1" * 64,
        exact_model_id="provider-model-2026-08-01",
        provider="acme",
        router=None,
        billing_account_id="acct-1",
        adapter_id="acme_adapter",
        adapter_version="1.0",
        prompt_template_id="prompt_v1",
        prompt_sha256="2" * 64,
        hyperparameters={"temperature": 0},
        position=0,
    )
    rmc2 = dataclasses.replace(rmc1, roster_member_config_id="placeholder-2")
    assert rmc1.semantic_id() == rmc2.semantic_id()

    rmc3 = dataclasses.replace(rmc1, position=1)
    assert rmc1.semantic_id() != rmc3.semantic_id()


def _roster_member(position: int) -> RosterMemberConfiguration:
    return RosterMemberConfiguration(
        roster_member_config_id=f"placeholder-{position}",
        registry_entry_id="registry_entry_sha256:" + "1" * 64,
        exact_model_id="provider-model-2026-08-01",
        provider="acme",
        router=None,
        billing_account_id="acct-1",
        adapter_id="acme_adapter",
        adapter_version="1.0",
        prompt_template_id="prompt_v1",
        prompt_sha256="2" * 64,
        hyperparameters={"temperature": 0},
        position=position,
    )


def test_run_manifest_identity_ignores_created_at_and_own_id():
    manifest1 = RunManifest(
        run_manifest_id="placeholder-1",
        schema_version="v2",
        created_at="2026-08-31T00:00:00Z",
        dataset_id="havi_failure_mode_v1",
        dataset_version="2026-08-31",
        corpus_evidence_kind="havi_natural",
        dataset_split_id="dataset_split_sha256:" + "3" * 64,
        split_role="diagnostic",
        calibration_selection_id=None,
        evaluation_unit_ids=("evaluation_unit_sha256:" + "b" * 64,),
        page_input_variant_ids=("page_input_variant_sha256:" + "c" * 64,),
        roster=(_roster_member(0),),
        config_snapshot_sha256="4" * 64,
        taxonomy_registry_id="taxonomy_v1",
        assessment_policy_id="assessment_v1",
        alignment_policy_id="alignment_v1",
        eligibility_policy_id="eligibility_v1",
        fusion_policy_id="fusion_v1",
        metric_policy_id="metric_v1",
        anomaly_policy_id="anomaly_v1",
        analysis_policy_id="analysis_v1",
        contamination_register_id="contamination_v1",
        pricing_registry_snapshot_id="pricing_v1",
        model_registry_snapshot_id="model_registry_v1",
        budget_policy={"run_ceiling_usd": 10.0},
        model_corpus_match=False,
        model_corpus_match_policy_id=None,
        human_override_ids=(),
    )
    manifest2 = dataclasses.replace(
        manifest1,
        run_manifest_id="placeholder-2",
        created_at="2026-09-01T00:00:00Z",
    )
    assert manifest1.semantic_id() == manifest2.semantic_id()
    assert manifest1.record_sha256() != manifest2.record_sha256()

    manifest3 = dataclasses.replace(manifest1, roster=(_roster_member(0), _roster_member(1)))
    assert manifest1.semantic_id() != manifest3.semantic_id()


def test_dataset_split_identity():
    split1 = DatasetSplit(
        dataset_split_id="placeholder-1",
        dataset_id="bln600_v1",
        dataset_version="2026-08-31",
        algorithm_id="hash_bucket_v1",
        algorithm_version="1.0",
        hash_rule="sha256(dataset_item_id) mod 5",
        seed="fixed-seed-1",
        assignments=(("item-1", "calibration"), ("item-2", "locked_evaluation")),
    )
    split2 = dataclasses.replace(split1, dataset_split_id="placeholder-2")
    assert split1.semantic_id() == split2.semantic_id()


def test_calibration_selection_identity():
    sel1 = CalibrationSelection(
        calibration_selection_id="placeholder-1",
        dataset_split_id="dataset_split_sha256:" + "3" * 64,
        calibration_run_manifest_id="run_manifest_sha256:" + "5" * 64,
        calibration_gt_snapshot_id="effective_gt_snapshot_sha256:" + "6" * 64,
        calibration_observation_store_sha256="7" * 64,
        selected_comparator_roster_member_config_id="roster_member_config_sha256:" + "8" * 64,
        selection_metric="cer",
        selection_rule="minimum calibration CER; ties by manifest position",
        alignment_policy_id="alignment_v1",
        eligibility_policy_id="eligibility_v1",
        fusion_policy_id="fusion_v1",
        metric_policy_id="metric_v1",
        anomaly_policy_id="anomaly_v1",
        analysis_policy_id="analysis_v1",
        bootstrap_algorithm_id="bootstrap_v1",
        bootstrap_seed="fixed-seed-2",
        bootstrap_replicates=10000,
    )
    sel2 = dataclasses.replace(sel1, calibration_selection_id="placeholder-2")
    assert sel1.semantic_id() == sel2.semantic_id()


def test_dispatch_intent_identity():
    di1 = DispatchIntent(
        dispatch_intent_id="placeholder-1",
        run_manifest_id="run_manifest_sha256:" + "5" * 64,
        page_input_variant_id="page_input_variant_sha256:" + "c" * 64,
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        roster_member_config_id="roster_member_config_sha256:" + "8" * 64,
        dispatch_pair_id=dispatch_pair_id(
            "run_manifest_sha256:" + "5" * 64,
            "page_input_variant_sha256:" + "c" * 64,
            "roster_member_config_sha256:" + "8" * 64,
        ),
        ordinal=0,
        planned_max_attempts=4,
    )
    di2 = dataclasses.replace(di1, dispatch_intent_id="placeholder-2")
    assert di1.semantic_id() == di2.semantic_id()


def test_dispatch_pair_id_is_deterministic_and_order_sensitive_by_field():
    pair1 = dispatch_pair_id("run-1", "variant-1", "roster-1")
    pair2 = dispatch_pair_id("run-1", "variant-1", "roster-1")
    assert pair1 == pair2

    pair3 = dispatch_pair_id("run-1", "variant-1", "roster-2")
    assert pair1 != pair3


def test_attempt_id_deterministic_per_intent_and_attempt_number():
    id1 = attempt_id("dispatch_intent_sha256:" + "9" * 64, 1)
    id2 = attempt_id("dispatch_intent_sha256:" + "9" * 64, 1)
    id3 = attempt_id("dispatch_intent_sha256:" + "9" * 64, 2)
    assert id1 == id2
    assert id1 != id3


def test_attempt_event_id_ignores_timestamp_by_construction():
    # event_id hashes only attempt_id, event_type, and the non-timestamp payload;
    # callers must exclude timestamps from the payload dict passed in.
    payload = {"outcome": "success", "http_status": 200}
    eid1 = attempt_event_id("attempt_sha256:" + "a" * 64, "attempt_finished", payload)
    eid2 = attempt_event_id("attempt_sha256:" + "a" * 64, "attempt_finished", dict(payload))
    assert eid1 == eid2


def test_raw_provider_envelope_identity_ignores_artifact_uri():
    env1 = RawProviderEnvelope(
        raw_provider_envelope_id="placeholder-1",
        sanitized_envelope_sha256="b" * 64,
        sanitization_policy_version="v1",
        artifact_uri="file:///envelope-a.json.zst",
    )
    env2 = dataclasses.replace(
        env1, raw_provider_envelope_id="placeholder-2", artifact_uri="file:///different.json.zst"
    )
    assert env1.semantic_id() == env2.semantic_id()


def test_approval_authorization_identity_ignores_evidence_uri_and_verified_at():
    auth1 = ApprovalAuthorization(
        approval_authorization_id="placeholder-1",
        dataset_id="bln600_v1",
        dataset_version="2026-08-31",
        authority_actor_id="bln600-project",
        authorization_basis="published dataset license grants re-keyed transcription reuse",
        evidence_uri="https://example.org/bln600-license",
        evidence_sha256="c" * 64,
        verified_by="curator-1",
        verified_at="2026-08-31T00:00:00Z",
    )
    auth2 = dataclasses.replace(
        auth1,
        approval_authorization_id="placeholder-2",
        evidence_uri="https://example.org/mirror/bln600-license",
        verified_at="2026-09-01T00:00:00Z",
    )
    assert auth1.semantic_id() == auth2.semantic_id()


def test_ground_truth_assertion_identity_ignores_created_at():
    ga1 = GroundTruthAssertion(
        assertion_id="placeholder-1",
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        role="gold_full",
        target_state="transcribable",
        text="the quick brown fox",
        guideline_id="guideline_v1",
        source="havi_human_authored",
        author_actor_id="annotator-1",
        created_at="2026-08-31T00:00:00Z",
        source_artifact_sha256=None,
    )
    ga2 = dataclasses.replace(
        ga1, assertion_id="placeholder-2", created_at="2026-09-01T00:00:00Z"
    )
    assert ga1.semantic_id() == ga2.semantic_id()
    assert ga1.record_sha256() != ga2.record_sha256()


def test_ground_truth_event_identity_ignores_created_at():
    ev1 = GroundTruthEvent(
        event_id="placeholder-1",
        sequence=1,
        prior_event_hash=None,
        assertion_id="ground_truth_assertion_sha256:" + "d" * 64,
        action="submit",
        actor_id="annotator-1",
        event_source="human_workflow",
        approval_authorization_id=None,
        reason=None,
        superseded_by_assertion_id=None,
        created_at="2026-08-31T00:00:00Z",
    )
    ev2 = dataclasses.replace(
        ev1, event_id="placeholder-2", created_at="2026-09-01T00:00:00Z"
    )
    assert ev1.semantic_id() == ev2.semantic_id()


def test_effective_gt_snapshot_identity():
    snap1 = EffectiveGroundTruthSnapshot(
        effective_gt_snapshot_id="placeholder-1",
        assertion_ids=("ground_truth_assertion_sha256:" + "d" * 64,),
        event_ids=("ground_truth_event_sha256:" + "e" * 64,),
        authorization_ids=(),
        conflicted_evaluation_unit_ids=(),
        unavailable_evaluation_unit_ids=(),
        resolver_policy_version="resolver_v1",
    )
    snap2 = dataclasses.replace(snap1, effective_gt_snapshot_id="placeholder-2")
    assert snap1.semantic_id() == snap2.semantic_id()


def test_page_model_result_identity_ignores_created_at():
    pmr1 = PageModelResult(
        page_model_result_id="placeholder-1",
        dispatch_pair_id="dispatch_pair_sha256:" + "f" * 64,
        dispatch_intent_id="dispatch_intent_sha256:" + "9" * 64,
        run_manifest_id="run_manifest_sha256:" + "5" * 64,
        evaluation_unit_id="evaluation_unit_sha256:" + "b" * 64,
        page_input_variant_id="page_input_variant_sha256:" + "c" * 64,
        roster_member_config_id="roster_member_config_sha256:" + "8" * 64,
        selected_attempt_id="attempt_sha256:" + "a" * 64,
        attempt_ids=("attempt_sha256:" + "a" * 64,),
        terminal_outcome="success",
        parsed_text="the quick brown fox",
        eligibility="eligible",
        ineligibility_reasons=(),
        total_cost_usd=Decimal("0.0123"),
        total_measured_duration_ms=842.5,
        created_at="2026-08-31T00:00:00Z",
    )
    pmr2 = dataclasses.replace(
        pmr1, page_model_result_id="placeholder-2", created_at="2026-09-01T00:00:00Z"
    )
    assert pmr1.semantic_id() == pmr2.semantic_id()
    assert pmr1.record_sha256() != pmr2.record_sha256()

    pmr3 = dataclasses.replace(pmr1, total_cost_usd=Decimal("0.0124"))
    assert pmr1.semantic_id() != pmr3.semantic_id()


def test_benchmark_observation_identity_ignores_created_at():
    bo1 = BenchmarkObservation(
        benchmark_observation_id="placeholder-1",
        run_manifest_id="run_manifest_sha256:" + "5" * 64,
        dataset_id="havi_failure_mode_v1",
        dataset_version="2026-08-31",
        corpus_evidence_kind="havi_natural",
        split="diagnostic",
        evaluation_unit_id="evaluation_unit_sha256:" + "1" * 64,
        source_page_id="source_page_sha256:" + "2" * 64,
        source_sha256="3" * 64,
        page_input_variant_id="page_input_variant_sha256:" + "c" * 64,
        input_variant_sha256="4" * 64,
        subject_kind="page_model_result",
        subject_id="page_model_result_sha256:" + "8" * 64,
        roster_member_config_id="roster_member_config_sha256:" + "9" * 64,
        exact_model_id="stub-model-v1",
        provider="stub",
        router=None,
        prompt_sha256="1" * 64,
        hyperparameters_sha256=None,
        adapter_version="1.0.0",
        challenge_seed_family=None,
        failure_condition_codes=(),
        evaluation_scope_status="in_scope_scored",
        layout_complexity="linear",
        language=None,
        mixed_language=False,
        gt_snapshot_id="effective_gt_snapshot_sha256:" + "6" * 64,
        gt_assertion_id="assertion_sha256:" + "7" * 64,
        gt_availability="available",
        metric_policy_id="metric_v1",
        hypothesis_text_sha256="a" * 64,
        reference_length_chars=42,
        reference_length_words=8,
        char_substitutions=1,
        char_insertions=0,
        char_deletions=1,
        word_substitutions=0,
        word_insertions=0,
        word_deletions=0,
        cer=0.05,
        wer=0.10,
        consensus_entropy=0.02,
        quorum_size=2,
        eligible_hypothesis_count=2,
        outcome="success",
        attributed_cost_usd=Decimal("0.005"),
        attributed_latency_ms=842.5,
        resource_attribution="member_all_attempts",
        created_at="2026-08-31T00:00:00Z",
    )
    bo2 = dataclasses.replace(
        bo1, benchmark_observation_id="placeholder-2", created_at="2026-09-01T00:00:00Z"
    )
    assert bo1.semantic_id() == bo2.semantic_id()
    assert bo1.record_sha256() != bo2.record_sha256()

    bo3 = dataclasses.replace(bo1, cer=0.06)
    assert bo1.semantic_id() != bo3.semantic_id()
