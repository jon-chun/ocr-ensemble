"""Canonical immutable record types.

Every type is a frozen dataclass. Each declares ``IDENTITY_FIELDS``, the ordered
field names forming its semantic-identity preimage per the identity registry
table. ``semantic_id()`` hashes only those fields; ``record_sha256()`` hashes the
full serialized record except ``record_sha256`` itself. Adding or removing an
identity field is a schema-version change, so ``IDENTITY_FIELDS``
must never be edited casually — treat it as part of the wire contract.

This module intentionally holds every record type in one place: the identity
table is the single cross-cutting contract every stage (A0-A9) reads,
and splitting it across per-stage modules would let two stages drift out of
sync with each other's field names.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Literal, TypedDict

from ocr_ensemble.identity import canonical_json_bytes, content_id, hash_preimage, sha256_hex


class CanonicalRecord:
    """Mixin providing semantic-id and full-record hashing from ``IDENTITY_FIELDS``.

    Subclasses are frozen dataclasses that declare a class-level
    ``IDENTITY_FIELDS: tuple[str, ...]`` and a class-level ``RECORD_TYPE: str``
    (used as the content-id type prefix, e.g. ``"source_page"``).
    """

    IDENTITY_FIELDS: tuple[str, ...] = ()
    RECORD_TYPE: str = ""

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    def identity_preimage(self) -> dict[str, Any]:
        data = self._as_dict()
        missing = [f for f in self.IDENTITY_FIELDS if f not in data]
        if missing:
            raise AttributeError(
                f"{type(self).__name__}.IDENTITY_FIELDS references undeclared "
                f"fields: {missing}"
            )
        return {k: data[k] for k in self.IDENTITY_FIELDS}

    def semantic_id(self) -> str:
        """Content ID over only the declared identity-field preimage."""
        digest = hash_preimage(self.identity_preimage())
        return content_id(self.RECORD_TYPE, digest)

    def record_sha256(self) -> str:
        """Hash over the complete serialized record except ``record_sha256``
        itself. This intentionally includes the record's own
        ID, artifact URIs, and timestamps — those are exactly the audit
        metadata this hash is meant to protect. Only ``semantic_id()`` excludes
        non-identity fields; ``record_sha256`` excludes nothing else.
        """
        data = self._as_dict()
        filtered = {k: v for k, v in data.items() if k != "record_sha256"}
        return sha256_hex(canonical_json_bytes(filtered))


def seal(record: CanonicalRecord) -> CanonicalRecord:
    """Replace ``record``'s own placeholder ID field with its computed
    semantic ID, matching the ``<field>_id = sha256({...})`` convention used
    throughout the preprocess/manifest sealing stages. Every sealing call
    site constructs a record with its own id field literally set to
    ``"placeholder"`` and passes it here -- one implementation shared by every
    stage rather than re-derived per module.
    """
    semantic_id = record.semantic_id()
    id_field = next(
        f.name
        for f in dataclasses.fields(record)  # type: ignore[arg-type]
        if f.name.endswith("_id") and getattr(record, f.name) == "placeholder"
    )
    return dataclasses.replace(record, **{id_field: semantic_id})  # type: ignore[type-var]


# ---------------------------------------------------------------------------
# A0/A1: dataset, source, variant, and failure contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePage(CanonicalRecord):
    RECORD_TYPE = "source_page"
    IDENTITY_FIELDS = (
        "schema_version",
        "dataset_id",
        "dataset_version",
        "dataset_item_id",
        "source_sha256",
    )

    source_page_id: str
    schema_version: str
    dataset_id: str
    dataset_version: str
    dataset_item_id: str
    source_uri: str
    source_sha256: str
    media_type: str
    byte_size: int
    width_px: int
    height_px: int
    color_mode: str
    bit_depth: int | None
    orientation_metadata: str | None
    source_genre: str | None
    license_id: str | None


@dataclass(frozen=True)
class EvaluationUnit(CanonicalRecord):
    RECORD_TYPE = "evaluation_unit"
    IDENTITY_FIELDS = (
        "schema_version",
        "source_page_id",
        "selector",
        "layout_complexity",
        "reading_order_policy_id",
    )

    evaluation_unit_id: str
    schema_version: str
    source_page_id: str
    selector: dict | None
    layout_complexity: Literal["linear", "layout_dependent", "unknown"]
    reading_order_policy_id: str
    language: str | None
    mixed_language: bool
    evaluation_scope_status: Literal[
        "in_scope_scored", "diagnostic_non_gating", "unsupported_fixture"
    ]


@dataclass(frozen=True)
class TransformStep:
    step_id: str
    algorithm_id: str
    algorithm_version: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class PerturbationProvenance:
    natural_source_page_id: str
    natural_source_sha256: str
    recipe_id: str
    recipe_version: str
    generator_id: str
    generator_version: str
    ordered_parameters: tuple[tuple[str, Any], ...]
    random_seed: str
    severity: str
    intended_family: str
    intended_condition: str | None
    transform_chain: tuple[TransformStep, ...]


@dataclass(frozen=True)
class DecodeProvenance(CanonicalRecord):
    RECORD_TYPE = "decode_provenance"
    # every declared field except own ID, record_sha256, artifact URI, and
    # operational timestamps; artifact SHA-256 is included.
    # DecodeProvenance carries no artifact URI/timestamp fields itself, so its
    # preimage is every field but its own ID.
    IDENTITY_FIELDS = (
        "source_page_id",
        "decoder_id",
        "decoder_version",
        "original_media_type",
        "original_bit_depth",
        "original_color_mode",
        "original_orientation_metadata",
        "output_encoding",
    )

    decode_provenance_id: str
    source_page_id: str
    decoder_id: str
    decoder_version: str
    original_media_type: str
    original_bit_depth: int | None
    original_color_mode: str
    original_orientation_metadata: str | None
    output_encoding: str


@dataclass(frozen=True)
class PageInputVariant(CanonicalRecord):
    RECORD_TYPE = "page_input_variant"
    IDENTITY_FIELDS = (
        "schema_version",
        "evaluation_unit_id",
        "source_page_id",
        "variant_kind",
        "artifact_sha256",
        "media_type",
        "byte_size",
        "width_px",
        "height_px",
        "decode_provenance_id",
        "transform_chain",
        "preprocess_policy_id",
        "perturbation_provenance",
    )

    page_input_variant_id: str
    schema_version: str
    evaluation_unit_id: str
    source_page_id: str
    variant_kind: Literal["natural_baseline", "synthetic_challenge", "mitigated"]
    mitigation_kind: Literal["local_deterministic", "ai_model_enhancement"] | None
    artifact_uri: str
    artifact_sha256: str
    media_type: str
    byte_size: int
    width_px: int
    height_px: int
    decode_provenance_id: str
    transform_chain: tuple[TransformStep, ...]
    preprocess_policy_id: str
    perturbation_provenance: PerturbationProvenance | None


@dataclass(frozen=True)
class BaselineTranscription(CanonicalRecord):
    RECORD_TYPE = "baseline_transcription"
    IDENTITY_FIELDS = (
        "evaluation_unit_id",
        "baseline_kind",
        "text",
        "source",
        "source_artifact_sha256",
        "import_provenance_id",
    )

    baseline_transcription_id: str
    evaluation_unit_id: str
    baseline_kind: Literal["dataset_ocr"]
    text: str
    source: str
    source_artifact_sha256: str
    import_provenance_id: str


@dataclass(frozen=True)
class DetectorEvidence(CanonicalRecord):
    RECORD_TYPE = "detector_evidence"
    IDENTITY_FIELDS = (
        "evaluation_unit_id",
        "page_input_variant_id",
        "detector_id",
        "detector_version",
        "measurement_name",
        "value",
        "unit",
        "assessment_state",
        "artifact_sha256",
    )

    evidence_id: str
    evaluation_unit_id: str
    page_input_variant_id: str
    detector_id: str
    detector_version: str
    measurement_name: str
    value: float | int | str | bool | None
    unit: str | None
    assessment_state: Literal["measured", "not_assessed", "unsupported", "failed"]
    artifact_uri: str | None
    artifact_sha256: str | None


@dataclass(frozen=True)
class FailureAssessment(CanonicalRecord):
    RECORD_TYPE = "failure_assessment"
    IDENTITY_FIELDS = (
        "evaluation_unit_id",
        "condition_code",
        "state",
        "severity",
        "confidence",
        "provenance",
        "evidence_ids",
        "policy_version",
        "actor",
        "reason",
        "supersedes_assessment_id",
        "override_config_sha256",
    )

    failure_assessment_id: str
    evaluation_unit_id: str
    condition_code: str
    state: Literal["asserted", "withdrawn"]
    severity: Literal["none", "low", "medium", "high", "unknown"]
    confidence: float | None
    provenance: Literal["curator", "detector", "dataset", "human_override"]
    evidence_ids: tuple[str, ...]
    policy_version: str
    actor: str | None
    reason: str | None
    supersedes_assessment_id: str | None
    created_at: str
    override_config_sha256: str | None


# ---------------------------------------------------------------------------
# A2: centralized configuration, verified registries, and Run Manifest (§5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RosterMemberConfiguration(CanonicalRecord):
    RECORD_TYPE = "roster_member_config"
    # every declared field except own ID and record_sha256
    IDENTITY_FIELDS = (
        "registry_entry_id",
        "exact_model_id",
        "provider",
        "router",
        "billing_account_id",
        "adapter_id",
        "adapter_version",
        "prompt_template_id",
        "prompt_sha256",
        "hyperparameters",
        "position",
    )

    roster_member_config_id: str
    registry_entry_id: str
    exact_model_id: str
    provider: str
    router: str | None
    billing_account_id: str
    adapter_id: str
    adapter_version: str
    prompt_template_id: str
    prompt_sha256: str
    hyperparameters: dict[str, Any]
    position: int


@dataclass(frozen=True)
class RunManifest(CanonicalRecord):
    RECORD_TYPE = "run_manifest"
    # every declared field except own ID, record_sha256, and created_at
    IDENTITY_FIELDS = (
        "schema_version",
        "dataset_id",
        "dataset_version",
        "corpus_evidence_kind",
        "dataset_split_id",
        "split_role",
        "calibration_selection_id",
        "evaluation_unit_ids",
        "page_input_variant_ids",
        "roster",
        "config_snapshot_sha256",
        "taxonomy_registry_id",
        "assessment_policy_id",
        "alignment_policy_id",
        "eligibility_policy_id",
        "fusion_policy_id",
        "metric_policy_id",
        "anomaly_policy_id",
        "analysis_policy_id",
        "contamination_register_id",
        "pricing_registry_snapshot_id",
        "model_registry_snapshot_id",
        "budget_policy",
        "model_corpus_match",
        "model_corpus_match_policy_id",
        "human_override_ids",
    )

    run_manifest_id: str
    schema_version: str
    created_at: str
    dataset_id: str
    dataset_version: str
    corpus_evidence_kind: Literal["havi_natural", "havi_synthetic", "bln600"]
    dataset_split_id: str
    split_role: Literal["calibration", "locked_evaluation", "diagnostic"]
    calibration_selection_id: str | None
    evaluation_unit_ids: tuple[str, ...]
    page_input_variant_ids: tuple[str, ...]
    roster: tuple[RosterMemberConfiguration, ...]
    config_snapshot_sha256: str
    taxonomy_registry_id: str
    assessment_policy_id: str
    alignment_policy_id: str
    eligibility_policy_id: str
    fusion_policy_id: str
    metric_policy_id: str
    anomaly_policy_id: str
    analysis_policy_id: str
    contamination_register_id: str
    pricing_registry_snapshot_id: str
    model_registry_snapshot_id: str
    budget_policy: dict[str, Any]
    model_corpus_match: bool
    model_corpus_match_policy_id: str | None
    human_override_ids: tuple[str, ...]


    def identity_preimage(self) -> dict[str, Any]:
        preimage = super().identity_preimage()
        # roster contains nested CanonicalRecord dataclasses; reduce to plain
        # dicts of their own identity preimages so nested identity, not nested
        # object identity, drives the parent hash.
        preimage["roster"] = [member.identity_preimage() for member in self.roster]
        return preimage


@dataclass(frozen=True)
class DatasetSplit(CanonicalRecord):
    RECORD_TYPE = "dataset_split"
    IDENTITY_FIELDS = (
        "dataset_id",
        "dataset_version",
        "algorithm_id",
        "algorithm_version",
        "hash_rule",
        "seed",
        "assignments",
    )

    dataset_split_id: str
    dataset_id: str
    dataset_version: str
    algorithm_id: str
    algorithm_version: str
    hash_rule: str
    seed: str
    assignments: tuple[tuple[str, Literal["calibration", "locked_evaluation", "diagnostic"]], ...]


@dataclass(frozen=True)
class CalibrationSelection(CanonicalRecord):
    RECORD_TYPE = "calibration_selection"
    IDENTITY_FIELDS = (
        "dataset_split_id",
        "calibration_run_manifest_id",
        "calibration_gt_snapshot_id",
        "calibration_observation_store_sha256",
        "selected_comparator_roster_member_config_id",
        "selection_metric",
        "selection_rule",
        "alignment_policy_id",
        "eligibility_policy_id",
        "fusion_policy_id",
        "metric_policy_id",
        "anomaly_policy_id",
        "analysis_policy_id",
        "bootstrap_algorithm_id",
        "bootstrap_seed",
        "bootstrap_replicates",
    )

    calibration_selection_id: str
    dataset_split_id: str
    calibration_run_manifest_id: str
    calibration_gt_snapshot_id: str
    calibration_observation_store_sha256: str
    selected_comparator_roster_member_config_id: str
    selection_metric: Literal["cer"]
    selection_rule: str
    alignment_policy_id: str
    eligibility_policy_id: str
    fusion_policy_id: str
    metric_policy_id: str
    anomaly_policy_id: str
    analysis_policy_id: str
    bootstrap_algorithm_id: str
    bootstrap_seed: str
    bootstrap_replicates: int


@dataclass(frozen=True)
class DispatchIntent(CanonicalRecord):
    RECORD_TYPE = "dispatch_intent"
    # manifest, input variant, Evaluation Unit, roster configuration, ordinal,
    # and planned maximum attempts as applicable
    IDENTITY_FIELDS = (
        "run_manifest_id",
        "page_input_variant_id",
        "evaluation_unit_id",
        "roster_member_config_id",
        "ordinal",
        "planned_max_attempts",
    )

    dispatch_intent_id: str
    run_manifest_id: str
    page_input_variant_id: str
    evaluation_unit_id: str
    roster_member_config_id: str
    dispatch_pair_id: str
    ordinal: int
    planned_max_attempts: int


def dispatch_pair_id(
    run_manifest_id: str,
    page_input_variant_id: str,
    roster_member_config_id: str,
) -> str:
    """Hash of the three semantic IDs identifying a Dispatch Pair."""
    digest = hash_preimage(
        {
            "run_manifest_id": run_manifest_id,
            "page_input_variant_id": page_input_variant_id,
            "roster_member_config_id": roster_member_config_id,
        }
    )
    return content_id("dispatch_pair", digest)


# ---------------------------------------------------------------------------
# A3: dispatch attempts (identity table row "Attempt")
# ---------------------------------------------------------------------------


def attempt_id(dispatch_intent_id: str, attempt_number: int) -> str:
    """``attempt_id = hash(dispatch_intent_id, attempt_number)``."""
    digest = hash_preimage(
        {"dispatch_intent_id": dispatch_intent_id, "attempt_number": attempt_number}
    )
    return content_id("attempt", digest)


def attempt_event_id(
    attempt_id_: str,
    event_type: Literal["attempt_started", "attempt_finished"],
    non_timestamp_payload: dict[str, Any],
) -> str:
    """Attempt event IDs hash attempt ID, event type, and the non-timestamp
    event payload."""
    digest = hash_preimage(
        {
            "attempt_id": attempt_id_,
            "event_type": event_type,
            "payload": non_timestamp_payload,
        }
    )
    return content_id("attempt_event", digest)


def dispatch_refused_event_id(dispatch_intent_id: str, reason: str) -> str:
    """``DispatchRefused`` event ID: hashes the intent and refusal reason,
    never the timestamp -- same timestamp-blindness convention as every
    other event ID.
    """
    digest = hash_preimage({"dispatch_intent_id": dispatch_intent_id, "reason": reason})
    return content_id("dispatch_refused_event", digest)


@dataclass(frozen=True)
class RawProviderEnvelope(CanonicalRecord):
    RECORD_TYPE = "raw_provider_envelope"
    IDENTITY_FIELDS = ("sanitized_envelope_sha256", "sanitization_policy_version")

    raw_provider_envelope_id: str
    sanitized_envelope_sha256: str
    sanitization_policy_version: str
    artifact_uri: str


FieldAvailability = Literal["reported", "unsupported", "not_applicable", "unknown"]


@dataclass(frozen=True)
class FieldValue:
    """``value`` is non-null exactly when ``availability="reported"``; null for
    the other three states. Not a ``CanonicalRecord`` --
    it is always embedded inside one, never independently identified.
    """

    availability: FieldAvailability
    value: Any | None

    def __post_init__(self) -> None:
        reported = self.availability == "reported"
        has_value = self.value is not None
        if reported != has_value:
            raise ValueError(
                "FieldValue.value must be non-null exactly when "
                f"availability='reported' (got availability={self.availability!r}, "
                f"value={self.value!r})"
            )


@dataclass(frozen=True)
class RawEnvelopeRef:
    """Reference to a sanitized, content-addressed provider envelope blob.
    Every received provider envelope is retained completely
    as a compressed CAS blob; this is the pointer to it, not the payload.
    """

    envelope_sha256: str
    cas_uri: str
    compressed_byte_size: int
    uncompressed_byte_size: int
    media_type: str
    sanitization_policy_version: str
    preview_utf8: str


ParsedOcrOutputKind = Literal["transcription", "whole_unit_abstention", "parse_failure"]


@dataclass(frozen=True)
class ParsedOcrOutput:
    """Tagged union over the three parse outcomes. ``text`` is
    non-null only for ``kind="transcription"``.
    """

    kind: ParsedOcrOutputKind
    text: str | None
    parser_version: str
    complete: bool

    def __post_init__(self) -> None:
        if self.kind == "transcription":
            if self.text is None:
                raise ValueError("kind='transcription' requires non-null text")
        elif self.text is not None:
            raise ValueError(f"kind={self.kind!r} requires null text")


ApiCallAttemptFinishOutcome = Literal[
    "response_received",
    "transport_error",
    "timeout",
    "rate_limited",
    "content_filtered",
    "provider_rejected",
    "cancelled",
]


@dataclass(frozen=True)
class ApiCallAttemptStarted:
    """Immutable append-only event: a provider contact was authorized and
    about to be issued. Written and fsynced *before* network
    contact, after every applicable ledger has already reserved
    ``maximum_exposure_usd`` -- this ordering is the entire
    "no re-spend on indeterminate attempts" guarantee.
    """

    event_type: Literal["attempt_started"]
    event_id: str
    attempt_id: str
    dispatch_intent_id: str
    attempt_number: int
    provider_idempotency_key: str | None
    reservation_ids: tuple[str, ...]
    maximum_exposure_usd: Decimal
    request_sha256: str
    request_envelope: RawEnvelopeRef
    started_at: str


@dataclass(frozen=True)
class ApiCallAttemptFinished:
    """Immutable append-only event: the terminal outcome of one attempt.
    An attempt with a ``started`` event and no matching
    ``finished`` event is ``indeterminate`` -- there is no
    separate "indeterminate" record; it is the absence of this event.
    """

    event_type: Literal["attempt_finished"]
    event_id: str
    attempt_id: str
    finished_at: str
    outcome: ApiCallAttemptFinishOutcome
    http_status: FieldValue
    provider_finish_reason: str | None
    error_code: str | None
    error_message_sanitized: str | None
    tokens_input: FieldValue
    tokens_output: FieldValue
    tokens_thinking: FieldValue
    provider_duration_ms: FieldValue
    measured_duration_ms: float
    actual_cost_usd: FieldValue
    pricing_snapshot_id: str | None
    raw_envelope: RawEnvelopeRef | None
    parsed_output: ParsedOcrOutput | None


@dataclass(frozen=True)
class DispatchRefused:
    """Immutable append-only event: a Dispatch Pair's batch reservation could
    not be admitted, so no attempt was ever made -- "no member
    in that batch is dispatched" derives ``budget_refused``.

    This behavior (the pair derives ``budget_refused``) was originally
    described without specifying how that disposition is durably recorded
    for a pair that never got an attempt event at all. Ticket 03's original
    ``BudgetRefused`` was raised as a Python exception only; ticket 04 needs
    a persisted fact to read back, so this event closes that gap. Flagged
    for future design review rather than silently treating it as settled.
    """

    event_type: Literal["dispatch_refused"]
    event_id: str
    dispatch_intent_id: str
    dispatch_pair_id: str
    reason: str
    refused_at: str


class VerifiedModelRegistryEntry(TypedDict):
    """Preflight registry entry, verbatim shape. Not a hashed
    canonical record -- ``RunManifest`` references the whole registry only via
    ``model_registry_snapshot_id``, never an individual entry's identity.
    """

    registry_entry_id: str
    roster_member_id: str
    provider: str
    billing_account_id: str
    router: str | None
    exact_model_id: str
    endpoint_or_sdk: str
    adapter_id: str
    adapter_version: str
    supports_image_input: bool
    supported_media_types: list[str]
    max_image_bytes: int | None
    max_width_px: int | None
    max_height_px: int | None
    context_limit_tokens: int
    output_limit_tokens: int
    reports_token_usage: bool
    reports_provider_duration: bool
    rate_limit_policy: dict[str, Any]
    pricing_snapshot_id: str
    limits_source: str
    pricing_source: str
    verified_at: str
    credential_probe: Literal["passed", "not_required"]


# ---------------------------------------------------------------------------
# A6: append-only ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalAuthorization(CanonicalRecord):
    RECORD_TYPE = "approval_authorization"
    IDENTITY_FIELDS = (
        "dataset_id",
        "dataset_version",
        "authority_actor_id",
        "authorization_basis",
        "evidence_sha256",
        "verified_by",
    )

    approval_authorization_id: str
    dataset_id: str
    dataset_version: str
    authority_actor_id: str
    authorization_basis: str
    evidence_uri: str
    evidence_sha256: str
    verified_by: str
    verified_at: str


@dataclass(frozen=True)
class GroundTruthAssertion(CanonicalRecord):
    RECORD_TYPE = "ground_truth_assertion"
    IDENTITY_FIELDS = (
        "evaluation_unit_id",
        "role",
        "target_state",
        "text",
        "guideline_id",
        "source",
        "author_actor_id",
        "source_artifact_sha256",
    )

    assertion_id: str
    evaluation_unit_id: str
    role: Literal["gold_full"]
    target_state: Literal["transcribable", "blank", "fully_illegible"]
    text: str | None
    guideline_id: str
    source: str
    author_actor_id: str
    created_at: str
    source_artifact_sha256: str | None


@dataclass(frozen=True)
class GroundTruthEvent(CanonicalRecord):
    RECORD_TYPE = "ground_truth_event"
    IDENTITY_FIELDS = (
        "sequence",
        "prior_event_hash",
        "assertion_id",
        "action",
        "actor_id",
        "reason",
        "superseded_by_assertion_id",
        "event_source",
        "approval_authorization_id",
    )

    event_id: str
    sequence: int
    prior_event_hash: str | None
    assertion_id: str
    action: Literal["submit", "approve", "reject", "supersede"]
    actor_id: str
    event_source: Literal["human_workflow", "dataset_import"]
    approval_authorization_id: str | None
    reason: str | None
    superseded_by_assertion_id: str | None
    created_at: str


@dataclass(frozen=True)
class EffectiveGroundTruthSnapshot(CanonicalRecord):
    RECORD_TYPE = "effective_gt_snapshot"
    IDENTITY_FIELDS = (
        "assertion_ids",
        "event_ids",
        "authorization_ids",
        "conflicted_evaluation_unit_ids",
        "unavailable_evaluation_unit_ids",
        "resolver_policy_version",
    )

    effective_gt_snapshot_id: str
    assertion_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    authorization_ids: tuple[str, ...]
    conflicted_evaluation_unit_ids: tuple[str, ...]
    unavailable_evaluation_unit_ids: tuple[str, ...]
    resolver_policy_version: str


@dataclass(frozen=True)
class ImportedGroundTruth:
    assertion: GroundTruthAssertion
    initial_events: tuple[GroundTruthEvent, ...]
    approval_authorization: ApprovalAuthorization | None


# ---------------------------------------------------------------------------
# A4-A7: results, alignment, fusion, and Benchmark Observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageModelResult(CanonicalRecord):
    """The single derived terminal result for a Dispatch Pair.

    Selection is deterministic: the first successful complete response by
    attempt number wins; otherwise the versioned terminal-precedence table
    applies. ``total_cost_usd`` includes every charged attempt, not only the
    ``selected_attempt_id`` one.
    """

    RECORD_TYPE = "page_model_result"
    # every declared semantic input/output field except own ID, record_sha256,
    # artifact URI, and rendering/creation timestamps; referenced artifact
    # SHA-256 values are included
    IDENTITY_FIELDS = (
        "dispatch_pair_id",
        "dispatch_intent_id",
        "run_manifest_id",
        "evaluation_unit_id",
        "page_input_variant_id",
        "roster_member_config_id",
        "selected_attempt_id",
        "attempt_ids",
        "terminal_outcome",
        "parsed_text",
        "eligibility",
        "ineligibility_reasons",
        "total_cost_usd",
        "total_measured_duration_ms",
    )

    page_model_result_id: str
    dispatch_pair_id: str
    dispatch_intent_id: str
    run_manifest_id: str
    evaluation_unit_id: str
    page_input_variant_id: str
    roster_member_config_id: str
    selected_attempt_id: str | None
    attempt_ids: tuple[str, ...]
    terminal_outcome: Literal[
        "success",
        "whole_unit_abstention",
        "truncated",
        "content_filtered",
        "permanent_failure",
        "retry_exhausted",
        "budget_refused",
        "unsupported_input",
        "cancelled",
        "indeterminate",
    ]
    parsed_text: str | None
    eligibility: Literal["eligible", "ineligible"]
    ineligibility_reasons: tuple[str, ...]
    total_cost_usd: Decimal
    total_measured_duration_ms: float
    created_at: str


@dataclass(frozen=True)
class ConsensusResult(CanonicalRecord):
    """This type appears in ``materialize_scores``'s signature
    and the required-stores list but its fields were never defined elsewhere
    (a genuine design gap, closed here in code -- ticket 08). One per (run,
    Evaluation Unit): the canonical eligible-hypothesis set, quorum bookkeeping, CE, and
    a pointer to the persisted alignment artifact that computed it. Distinct
    from ``FusedHypothesis``, which is the emitted merged text derived from
    the same alignment.
    """

    RECORD_TYPE = "consensus_result"
    IDENTITY_FIELDS = (
        "run_manifest_id",
        "evaluation_unit_id",
        "eligible_page_model_result_ids",
        "quorum_size",
        "eligible_hypothesis_count",
        "consensus_entropy",
        "alignment_artifact_sha256",
        "text_policy_id",
        "alignment_policy_id",
    )

    consensus_result_id: str
    run_manifest_id: str
    evaluation_unit_id: str
    # sorted for a deterministic preimage; membership, not order, is semantic
    eligible_page_model_result_ids: tuple[str, ...]
    quorum_size: int
    eligible_hypothesis_count: int
    consensus_entropy: float | None  # None means not_available (below quorum or L=0)
    alignment_artifact_sha256: str
    alignment_artifact_uri: str
    text_policy_id: str
    alignment_policy_id: str
    created_at: str


@dataclass(frozen=True)
class FusedHypothesis(CanonicalRecord):
    """This type was named elsewhere but its fields were never defined (a
    genuine design gap, closed here in code -- ticket 08). The canonical,
    persisted counterpart of ``align.FusedHypothesis`` (an in-memory
    computation result, not a ``CanonicalRecord``): the emitted merged text
    plus a pointer to the same alignment artifact ``ConsensusResult``
    references, so postprocess can reproduce both from one source.
    """

    RECORD_TYPE = "fused_hypothesis"
    IDENTITY_FIELDS = (
        "run_manifest_id",
        "evaluation_unit_id",
        "consensus_result_id",
        "text",
        "fusion_policy_id",
        "alignment_artifact_sha256",
    )

    fused_hypothesis_id: str
    run_manifest_id: str
    evaluation_unit_id: str
    consensus_result_id: str
    text: str
    fusion_policy_id: str
    alignment_artifact_sha256: str
    alignment_artifact_uri: str
    created_at: str


@dataclass(frozen=True)
class BenchmarkObservation(CanonicalRecord):
    """Confidence bands are deliberately not a stored field
    here: they are a display/triage convenience computed over ``consensus_entropy``
    at read time (``ConsensusEntropyBandsConfig.band_for`` in
    ``align.py``), not a separate canonical fact about the observation.
    """

    RECORD_TYPE = "benchmark_observation"
    # every declared field except own ID, record_sha256, and created_at --
    # same convention as RunManifest/PageModelResult.
    IDENTITY_FIELDS = (
        "run_manifest_id",
        "dataset_id",
        "dataset_version",
        "corpus_evidence_kind",
        "split",
        "evaluation_unit_id",
        "source_page_id",
        "source_sha256",
        "page_input_variant_id",
        "input_variant_sha256",
        "subject_kind",
        "subject_id",
        "roster_member_config_id",
        "exact_model_id",
        "provider",
        "router",
        "prompt_sha256",
        "hyperparameters_sha256",
        "adapter_version",
        "challenge_seed_family",
        "failure_condition_codes",
        "evaluation_scope_status",
        "layout_complexity",
        "language",
        "mixed_language",
        "gt_snapshot_id",
        "gt_assertion_id",
        "gt_availability",
        "metric_policy_id",
        "hypothesis_text_sha256",
        "reference_length_chars",
        "reference_length_words",
        "char_substitutions",
        "char_insertions",
        "char_deletions",
        "word_substitutions",
        "word_insertions",
        "word_deletions",
        "cer",
        "wer",
        "consensus_entropy",
        "quorum_size",
        "eligible_hypothesis_count",
        "outcome",
        "attributed_cost_usd",
        "attributed_latency_ms",
        "resource_attribution",
    )

    benchmark_observation_id: str
    run_manifest_id: str
    dataset_id: str
    dataset_version: str
    corpus_evidence_kind: str
    split: Literal["calibration", "locked_evaluation", "diagnostic"]
    evaluation_unit_id: str
    source_page_id: str
    source_sha256: str
    page_input_variant_id: str
    input_variant_sha256: str
    subject_kind: Literal["page_model_result", "fused_hypothesis", "baseline_transcription"]
    subject_id: str
    roster_member_config_id: str | None
    exact_model_id: str | None
    provider: str | None
    router: str | None
    prompt_sha256: str | None
    hyperparameters_sha256: str | None
    adapter_version: str | None
    challenge_seed_family: str | None
    failure_condition_codes: tuple[str, ...]
    evaluation_scope_status: str
    layout_complexity: str
    language: str | None
    mixed_language: bool
    gt_snapshot_id: str
    gt_assertion_id: str | None
    gt_availability: Literal["available", "unavailable", "conflicted", "excluded_fully_illegible"]
    metric_policy_id: str
    hypothesis_text_sha256: str | None
    reference_length_chars: int | None
    reference_length_words: int | None
    char_substitutions: int | None
    char_insertions: int | None
    char_deletions: int | None
    word_substitutions: int | None
    word_insertions: int | None
    word_deletions: int | None
    cer: float | None
    wer: float | None
    consensus_entropy: float | None
    quorum_size: int | None
    eligible_hypothesis_count: int | None
    outcome: str
    attributed_cost_usd: Decimal | None
    attributed_latency_ms: float | None
    resource_attribution: Literal[
        "member_all_attempts", "ensemble_acquisition", "not_applicable"
    ]
    created_at: str


__all__ = [
    "CanonicalRecord",
    "seal",
    "SourcePage",
    "EvaluationUnit",
    "TransformStep",
    "PerturbationProvenance",
    "DecodeProvenance",
    "PageInputVariant",
    "BaselineTranscription",
    "DetectorEvidence",
    "FailureAssessment",
    "RosterMemberConfiguration",
    "RunManifest",
    "DatasetSplit",
    "CalibrationSelection",
    "DispatchIntent",
    "dispatch_pair_id",
    "attempt_id",
    "attempt_event_id",
    "dispatch_refused_event_id",
    "RawProviderEnvelope",
    "FieldValue",
    "RawEnvelopeRef",
    "ParsedOcrOutput",
    "ApiCallAttemptStarted",
    "ApiCallAttemptFinished",
    "DispatchRefused",
    "VerifiedModelRegistryEntry",
    "ApprovalAuthorization",
    "GroundTruthAssertion",
    "GroundTruthEvent",
    "EffectiveGroundTruthSnapshot",
    "ImportedGroundTruth",
    "PageModelResult",
    "ConsensusResult",
    "FusedHypothesis",
    "BenchmarkObservation",
]
