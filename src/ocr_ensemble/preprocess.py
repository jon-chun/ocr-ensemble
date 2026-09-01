"""A0/A1 preprocess: dataset import and Source Page / Page Input Variant sealing.

Importable API: ``preprocess_dataset(request) -> PreprocessArtifacts``.

Ticket 02 scope: import one fixture through the HAVI adapter, seal its
``SourcePage``/``EvaluationUnit``/``DecodeProvenance``/``PageInputVariant``
(``natural_baseline`` only — no mitigation logic), and write the three sealed
JSONL stores via the atomic-write discipline. Preprocess is
confirmed the sole owner of these identities: nothing here recomputes an ID
handed to it by a prior stage, and nothing downstream recomputes an ID this
module seals.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ocr_ensemble.adapters.havi import HaviFailureModeAdapter, ImportedSource, source_sha256
from ocr_ensemble.identity import sha256_hex
from ocr_ensemble.records import DecodeProvenance, EvaluationUnit, PageInputVariant, SourcePage
from ocr_ensemble.records import seal as _seal
from ocr_ensemble.storage import write_jsonl_atomic

SCHEMA_VERSION = "ocr-ensemble-schema/v2"
READING_ORDER_POLICY_ID = "reading_order_v1"
PREPROCESS_POLICY_ID = "preprocess_v1"
DECODER_ID = "pillow"


@dataclass(frozen=True)
class PreprocessRequest:
    """Minimal ticket-02 request shape.

    The full ``PreprocessRequest`` (preprocess spec §3) additionally requires
    ``adapter_config_path``, ``preprocess_policy_path``, ``override_path``, and
    ``resume`` — deferred to the ticket that needs config-file-driven,
    whole-corpus preprocessing rather than one pinned fixture.
    """

    dataset_root: Path
    output_root: Path
    fixture_relative_paths: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessArtifacts:
    source_pages: tuple[SourcePage, ...]
    evaluation_units: tuple[EvaluationUnit, ...]
    decode_provenances: tuple[DecodeProvenance, ...]
    input_variants: tuple[PageInputVariant, ...]


def _decode_provenance(imported: ImportedSource, source_page_id: str) -> DecodeProvenance:
    with Image.open(imported.source_path) as im:
        original_media_type = Image.MIME.get(im.format or "", "application/octet-stream")
        original_bit_depth = {
            "1": 1,
            "L": 8,
            "LA": 8,
            "P": 8,
            "RGB": 8,
            "RGBA": 8,
            "I": 32,
            "F": 32,
        }.get(im.mode)
        original_color_mode = im.mode
    return DecodeProvenance(
        decode_provenance_id="placeholder",
        source_page_id=source_page_id,
        decoder_id=DECODER_ID,
        decoder_version=Image.__version__,
        original_media_type=original_media_type,
        original_bit_depth=original_bit_depth,
        original_color_mode=original_color_mode,
        original_orientation_metadata=None,
        output_encoding="png",
    )


def _drop_fully_opaque_alpha(im: Image.Image) -> Image.Image:
    """Lossless-normalize decode only (preprocess spec §5): if the image
    carries an alpha channel that is fully opaque everywhere (no transparency
    information present), drop it; otherwise return the image unchanged. This
    never discards visually meaningful data, so it stays within "declared
    lossless decode/format normalization only" rather than a real content
    transform.
    """
    if "A" not in im.mode:
        return im
    alpha = im.getchannel("A")
    lo, hi = alpha.getextrema()
    if lo == 255 and hi == 255:
        return im.convert("RGB" if im.mode != "LA" else "L")
    return im


def preprocess_dataset(request: PreprocessRequest) -> PreprocessArtifacts:
    if request.dataset_root.resolve() == request.output_root.resolve() or str(
        request.output_root.resolve()
    ).startswith(str(request.dataset_root.resolve()) + "/"):
        raise ValueError("output_root must never overlap the source dataset_root")

    adapter = HaviFailureModeAdapter()
    imported_sources = adapter.import_sources(
        request.dataset_root, only_relative_paths=request.fixture_relative_paths
    )

    source_pages: list[SourcePage] = []
    evaluation_units: list[EvaluationUnit] = []
    decode_provenances: list[DecodeProvenance] = []
    input_variants: list[PageInputVariant] = []

    for imported in imported_sources:
        checksum = source_sha256(imported.source_path)
        with Image.open(imported.source_path) as im:
            width_px, height_px = im.size
            byte_size = imported.source_path.stat().st_size
            media_type = Image.MIME.get(im.format or "", "application/octet-stream")

        source_page = _seal(
            SourcePage(
                source_page_id="placeholder",
                schema_version=SCHEMA_VERSION,
                dataset_id=imported.dataset_id,
                dataset_version=imported.dataset_version,
                dataset_item_id=imported.dataset_item_id,
                source_uri=imported.source_path.resolve().as_uri(),
                source_sha256=checksum,
                media_type=media_type,
                byte_size=byte_size,
                width_px=width_px,
                height_px=height_px,
                color_mode=im.mode,
                bit_depth=None,
                orientation_metadata=None,
                source_genre=imported.source_genre,
                license_id=imported.license_id,
            )
        )

        evaluation_unit = _seal(
            EvaluationUnit(
                evaluation_unit_id="placeholder",
                schema_version=SCHEMA_VERSION,
                source_page_id=source_page.source_page_id,
                selector=None,  # whole_image: complete source image
                layout_complexity=imported.layout_complexity,
                reading_order_policy_id=READING_ORDER_POLICY_ID,
                language=imported.language,
                mixed_language=False,
                evaluation_scope_status=imported.evaluation_scope_status,
            )
        )

        decode_provenance = _seal(_decode_provenance(imported, source_page.source_page_id))

        # natural_baseline: source-faithful, lossless-normalized decode only,
        # no deskew/dewarp/denoise/contrast/restoration (preprocess spec §5).
        variant_dir = request.output_root / "artifacts" / "input-variant"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_filename = evaluation_unit.evaluation_unit_id.replace(":", "_") + ".png"
        variant_path = variant_dir / variant_filename
        with Image.open(imported.source_path) as im:
            normalized = _drop_fully_opaque_alpha(im)
            normalized.save(variant_path, format="PNG")
        variant_bytes = variant_path.read_bytes()
        variant_sha256 = sha256_hex(variant_bytes)

        input_variant = _seal(
            PageInputVariant(
                page_input_variant_id="placeholder",
                schema_version=SCHEMA_VERSION,
                evaluation_unit_id=evaluation_unit.evaluation_unit_id,
                source_page_id=source_page.source_page_id,
                variant_kind="natural_baseline",
                mitigation_kind=None,
                artifact_uri=variant_path.resolve().as_uri(),
                artifact_sha256=variant_sha256,
                media_type="image/png",
                byte_size=len(variant_bytes),
                width_px=width_px,
                height_px=height_px,
                decode_provenance_id=decode_provenance.decode_provenance_id,
                transform_chain=(),
                preprocess_policy_id=PREPROCESS_POLICY_ID,
                perturbation_provenance=None,
            )
        )

        source_pages.append(source_page)
        evaluation_units.append(evaluation_unit)
        decode_provenances.append(decode_provenance)
        input_variants.append(input_variant)

    _write_sealed_stores(
        request.output_root, source_pages, evaluation_units, input_variants
    )

    return PreprocessArtifacts(
        source_pages=tuple(source_pages),
        evaluation_units=tuple(evaluation_units),
        decode_provenances=tuple(decode_provenances),
        input_variants=tuple(input_variants),
    )


def _write_sealed_stores(
    output_root: Path,
    source_pages: list[SourcePage],
    evaluation_units: list[EvaluationUnit],
    input_variants: list[PageInputVariant],
) -> None:
    write_jsonl_atomic(
        output_root / "source_pages.jsonl",
        (dataclasses.asdict(p) for p in source_pages),
    )
    write_jsonl_atomic(
        output_root / "evaluation_units.jsonl",
        (dataclasses.asdict(u) for u in evaluation_units),
    )
    write_jsonl_atomic(
        output_root / "input_variants.jsonl",
        (dataclasses.asdict(v) for v in input_variants),
    )


def cli_main(_argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "the full preprocess CLI (config-file-driven, whole-corpus) is "
        "implemented starting with the ticket that widens beyond ticket 02's "
        "single pinned fixture"
    )
