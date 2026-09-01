"""A2 preflight: seal a Run Manifest and its Dispatch Intents.

Ticket 03 scope: one stub roster member against the ticket-02 fixture's
sealed ``natural_baseline`` Page Input Variant. The full A2 contract (real
verified-registry preflight, Dataset Split assignment across an entire
corpus, Calibration Selection binding for locked runs) is a later ticket's
scope; this seals just enough of a real ``RunManifest``/``DispatchIntent``
pair to exercise A3's reserve-then-attempt-then-log lifecycle honestly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ocr_ensemble.records import (
    DatasetSplit,
    DispatchIntent,
    EvaluationUnit,
    PageInputVariant,
    RosterMemberConfiguration,
    RunManifest,
    dispatch_pair_id,
)
from ocr_ensemble.records import seal as _seal
from ocr_ensemble.storage import write_json_atomic, write_jsonl_atomic

SCHEMA_VERSION = "ocr-ensemble-schema/v2"
STUB_ADAPTER_ID = "stub_model_v1"
STUB_PROVIDER = "stub"
STUB_REGISTRY_ENTRY_ID = "registry_entry_sha256:" + "0" * 64  # no live registry yet (ticket 09)


def seal_stub_roster_member(
    *,
    billing_account_id: str = "stub-billing-account",
    prompt_template_id: str = "ocr_transcription_v1",
    prompt_sha256: str = "1" * 64,
    hyperparameters: dict | None = None,
    position: int = 0,
) -> RosterMemberConfiguration:
    return _seal(
        RosterMemberConfiguration(
            roster_member_config_id="placeholder",
            registry_entry_id=STUB_REGISTRY_ENTRY_ID,
            exact_model_id="stub-model-v1",
            provider=STUB_PROVIDER,
            router=None,
            billing_account_id=billing_account_id,
            adapter_id=STUB_ADAPTER_ID,
            adapter_version="1.0.0",
            prompt_template_id=prompt_template_id,
            prompt_sha256=prompt_sha256,
            hyperparameters=hyperparameters or {},
            position=position,
        )
    )


def seal_stub_dataset_split(
    *, evaluation_unit_id: str, dataset_id: str, dataset_version: str
) -> DatasetSplit:
    return _seal(
        DatasetSplit(
            dataset_split_id="placeholder",
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            algorithm_id="ticket03_single_fixture_v1",
            algorithm_version="1.0.0",
            hash_rule="single fixture: always diagnostic",
            seed="not-applicable",
            assignments=((evaluation_unit_id, "diagnostic"),),
        )
    )


def seal_run_manifest(
    *,
    evaluation_unit: EvaluationUnit,
    page_input_variant: PageInputVariant,
    roster: tuple[RosterMemberConfiguration, ...],
    dataset_split: DatasetSplit,
    created_at: str,
    budget_policy: dict,
    dataset_id: str,
    dataset_version: str,
) -> RunManifest:
    return _seal(
        RunManifest(
            run_manifest_id="placeholder",
            schema_version=SCHEMA_VERSION,
            created_at=created_at,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            corpus_evidence_kind="havi_natural",
            dataset_split_id=dataset_split.dataset_split_id,
            split_role="diagnostic",
            calibration_selection_id=None,
            evaluation_unit_ids=(evaluation_unit.evaluation_unit_id,),
            page_input_variant_ids=(page_input_variant.page_input_variant_id,),
            roster=roster,
            config_snapshot_sha256="2" * 64,
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
            budget_policy=budget_policy,
            model_corpus_match=False,
            model_corpus_match_policy_id=None,
            human_override_ids=(),
        )
    )


def seal_dispatch_intents(
    *,
    run_manifest: RunManifest,
    page_input_variant: PageInputVariant,
    evaluation_unit: EvaluationUnit,
    roster: tuple[RosterMemberConfiguration, ...],
    planned_max_attempts: int = 4,  # 1 initial + 3 retries
) -> tuple[DispatchIntent, ...]:
    """Seal exactly one ``DispatchIntent`` per roster member, all written
    before A3 starts. The expected-pair matrix is exactly
    this set of intents -- nothing downstream may infer it as an unchecked
    Cartesian product.
    """
    intents = []
    for ordinal, member in enumerate(roster):
        pair_id = dispatch_pair_id(
            run_manifest.run_manifest_id,
            page_input_variant.page_input_variant_id,
            member.roster_member_config_id,
        )
        intent = _seal(
            DispatchIntent(
                dispatch_intent_id="placeholder",
                run_manifest_id=run_manifest.run_manifest_id,
                page_input_variant_id=page_input_variant.page_input_variant_id,
                evaluation_unit_id=evaluation_unit.evaluation_unit_id,
                roster_member_config_id=member.roster_member_config_id,
                dispatch_pair_id=pair_id,
                ordinal=ordinal,
                planned_max_attempts=planned_max_attempts,
            )
        )
        intents.append(intent)
    return tuple(intents)


def write_run_manifest(output_root: Path, run_manifest: RunManifest) -> None:
    write_json_atomic(output_root / "run_manifest.json", dataclasses.asdict(run_manifest))


def write_dispatch_intents(output_root: Path, intents: tuple[DispatchIntent, ...]) -> None:
    write_jsonl_atomic(
        output_root / "dispatch_intents.jsonl",
        (dataclasses.asdict(intent) for intent in intents),
    )
