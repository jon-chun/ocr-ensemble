"""`local_deterministic` Mitigation Strategy plugins (preprocess spec §5).

Ticket 10 scope: one geometry strategy (deskew) using only library-local,
non-network transforms (OpenCV). A `local_deterministic` variant dispatches
within the same Run Manifest as the `natural_baseline` variant it derives
from -- no `PairedExperiment` concept is imported or referenced anywhere in
this module, matching the relaxed dispatch policy for this kind.

`MitigationOutcome` is the shared return shape a strategy reports through. It
is deliberately generic over "kind" (`mitigation_kind` on the produced
variant) so an `ai_model_enhancement` strategy (ticket 11, a paid vision-model
call requiring `PairedExperiment`) can return the same shape later without
this module changing; only its dispatch policy differs, and dispatch/pairing
lives outside this module entirely (there is no Run Manifest or dispatch
concept implemented yet -- tickets 03+).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image

from ocr_ensemble.identity import sha256_hex
from ocr_ensemble.preprocess import PREPROCESS_POLICY_ID, SCHEMA_VERSION, _seal
from ocr_ensemble.records import EvaluationUnit, PageInputVariant, TransformStep

DESKEW_ALGORITHM_ID = "deskew_projection_profile_cv2"
DESKEW_ALGORITHM_VERSION = "1.0.0"
DESKEW_STEP_ID = "deskew"

# Below this magnitude the detected rotation is not distinguishable from
# projection-profile measurement noise on a straight page (preprocess spec §5:
# "if a safe correction cannot be determined ... emit no mitigated variant").
MIN_CORRECTABLE_SKEW_DEGREES = 0.3

# Above this magnitude a single-pass affine rotation is not a "safe"
# correction for this strategy -- large apparent rotation is more likely
# perspective/warp or a detector failure than true in-plane skew, and
# guessing a large warp is exactly what preprocess spec §5 forbids.
MAX_CORRECTABLE_SKEW_DEGREES = 15.0

_COARSE_SEARCH_DEGREES = 10.0
_COARSE_STEP_DEGREES = 0.2
_FINE_WINDOW_DEGREES = 0.3
_FINE_STEP_DEGREES = 0.02


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected a file:// artifact_uri, got {uri!r}")
    return Path(unquote(parsed.path))


@dataclass(frozen=True)
class MitigationOutcome:
    """Either a sealed mitigated variant, or an explicit no-correction reason.

    Exactly one of ``variant``/``no_correction_reason`` is set. This is the
    "clear return type/mechanism" the ticket asks for so a caller never has
    to infer abstention from an absent list entry.
    """

    variant: PageInputVariant | None
    no_correction_reason: str | None
    detected_skew_degrees: float | None
    corrected_skew_degrees: float | None

    def __post_init__(self) -> None:
        if (self.variant is None) == (self.no_correction_reason is None):
            raise ValueError(
                "MitigationOutcome must set exactly one of variant / "
                "no_correction_reason"
            )


class MitigationStrategy(Protocol):
    """Shape a `local_deterministic` strategy implements.

    Takes the already-sealed `natural_baseline` variant as input rather than
    a raw path: a local transform operates on pixels that were already
    decoded once for that baseline, so the mitigated variant it produces
    reuses the baseline's `decode_provenance_id` instead of re-decoding
    (preprocess spec §5 -- decode provenance identity is about the original
    source decode, not about any transform layered on top of it).

    Generic enough that an `ai_model_enhancement` strategy could return the
    same `MitigationOutcome` shape from an analogous method later (ticket
    11); this module builds only the `local_deterministic` side.
    """

    mitigation_kind: str

    def apply(
        self,
        *,
        baseline_variant: PageInputVariant,
        evaluation_unit: EvaluationUnit,
        output_root: Path,
    ) -> MitigationOutcome: ...


def _detect_skew_degrees(binary: np.ndarray) -> float:
    """Projection-profile skew estimate: the rotation angle that maximizes
    horizontal-row-sum variance is the angle at which text baselines are
    most tightly aligned into sharp peaks (standard, well-documented
    deskew technique; robust here against the false near-zero angle a
    naive `minAreaRect` over the full page mask reports when the page's
    axis-aligned border dominates the bounding rectangle).
    """

    def score(angle: float) -> float:
        h, w = binary.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        row_sums = rotated.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(row_sums) ** 2))

    best_angle, best_score = 0.0, -1.0
    angle = -_COARSE_SEARCH_DEGREES
    while angle <= _COARSE_SEARCH_DEGREES:
        candidate_score = score(angle)
        if candidate_score > best_score:
            best_score, best_angle = candidate_score, angle
        angle += _COARSE_STEP_DEGREES

    refined_angle, refined_score = best_angle, -1.0
    angle = best_angle - _FINE_WINDOW_DEGREES
    while angle <= best_angle + _FINE_WINDOW_DEGREES:
        candidate_score = score(angle)
        if candidate_score > refined_score:
            refined_score, refined_angle = candidate_score, angle
        angle += _FINE_STEP_DEGREES

    return round(refined_angle, 3)


@dataclass(frozen=True)
class DeskewStrategy:
    """`local_deterministic` geometry Mitigation Strategy: detects in-plane
    rotational skew via an OpenCV projection-profile search and corrects it
    with a single affine warp (preprocess spec §5).
    """

    mitigation_kind = "local_deterministic"

    def apply(
        self,
        *,
        baseline_variant: PageInputVariant,
        evaluation_unit: EvaluationUnit,
        output_root: Path,
    ) -> MitigationOutcome:
        if baseline_variant.variant_kind != "natural_baseline":
            raise ValueError(
                "DeskewStrategy.apply requires the sealed natural_baseline "
                f"variant as input, got variant_kind={baseline_variant.variant_kind!r} "
                "(preprocess spec §5: mitigations never replace or derive "
                "from another mitigated variant implicitly)"
            )

        baseline_path = _path_from_file_uri(baseline_variant.artifact_uri)
        with Image.open(baseline_path) as im:
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            rgb = np.array(im)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        detected_skew = _detect_skew_degrees(binary)

        if abs(detected_skew) < MIN_CORRECTABLE_SKEW_DEGREES:
            return MitigationOutcome(
                variant=None,
                no_correction_reason=(
                    f"detected skew {detected_skew}deg is below the "
                    f"{MIN_CORRECTABLE_SKEW_DEGREES}deg minimum-correctable "
                    "threshold; page is already effectively straight"
                ),
                detected_skew_degrees=detected_skew,
                corrected_skew_degrees=None,
            )

        if abs(detected_skew) > MAX_CORRECTABLE_SKEW_DEGREES:
            return MitigationOutcome(
                variant=None,
                no_correction_reason=(
                    f"detected skew {detected_skew}deg exceeds the "
                    f"{MAX_CORRECTABLE_SKEW_DEGREES}deg maximum this strategy "
                    "treats as a safe in-plane rotation; a single affine warp "
                    "at this magnitude is more likely to be masking "
                    "perspective distortion or a detector failure than "
                    "correcting true page skew, so no transform is applied"
                ),
                detected_skew_degrees=detected_skew,
                corrected_skew_degrees=None,
            )

        height_px, width_px = rgb.shape[:2]
        center = (width_px / 2.0, height_px / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, detected_skew, 1.0)
        resampling_method = "bicubic"
        corrected = cv2.warpAffine(
            rgb,
            rotation_matrix,
            (width_px, height_px),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        _, corrected_binary = cv2.threshold(
            corrected_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        corrected_skew = _detect_skew_degrees(corrected_binary)

        variant_dir = output_root / "artifacts" / "input-variant"
        variant_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            evaluation_unit.evaluation_unit_id.replace(":", "_")
            + f"_{DESKEW_STEP_ID}.png"
        )
        variant_path = variant_dir / filename
        Image.fromarray(corrected).save(variant_path, format="PNG")
        variant_bytes = variant_path.read_bytes()
        variant_sha256 = sha256_hex(variant_bytes)

        transform_step = TransformStep(
            step_id=DESKEW_STEP_ID,
            algorithm_id=DESKEW_ALGORITHM_ID,
            algorithm_version=DESKEW_ALGORITHM_VERSION,
            parameters={
                "detected_skew_degrees": detected_skew,
                "corrected_skew_degrees": corrected_skew,
                "output_width_px": width_px,
                "output_height_px": height_px,
                "resampling_method": resampling_method,
                "border_mode": "replicate",
                "rotation_matrix": rotation_matrix.tolist(),
                "rotation_center": list(center),
                "detected_page_boundary": None,
            },
        )

        variant = _seal(
            PageInputVariant(
                page_input_variant_id="placeholder",
                schema_version=SCHEMA_VERSION,
                evaluation_unit_id=evaluation_unit.evaluation_unit_id,
                source_page_id=baseline_variant.source_page_id,
                variant_kind="mitigated",
                mitigation_kind="local_deterministic",
                artifact_uri=variant_path.resolve().as_uri(),
                artifact_sha256=variant_sha256,
                media_type="image/png",
                byte_size=len(variant_bytes),
                width_px=width_px,
                height_px=height_px,
                decode_provenance_id=baseline_variant.decode_provenance_id,
                transform_chain=(transform_step,),
                preprocess_policy_id=PREPROCESS_POLICY_ID,
                perturbation_provenance=None,
            )
        )

        return MitigationOutcome(
            variant=variant,
            no_correction_reason=None,
            detected_skew_degrees=detected_skew,
            corrected_skew_degrees=corrected_skew,
        )


__all__ = [
    "MitigationOutcome",
    "MitigationStrategy",
    "DeskewStrategy",
    "MIN_CORRECTABLE_SKEW_DEGREES",
    "MAX_CORRECTABLE_SKEW_DEGREES",
]
