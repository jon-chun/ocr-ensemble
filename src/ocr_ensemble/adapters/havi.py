"""HAVI Failure-Mode Corpus dataset adapter (preprocess spec §4.1).

Ticket 02 scope: ``import_sources`` only, for one trivial, clean fixture. Ticket
05 scope: ``import_ground_truth`` for that same fixture. Baseline import is not
required for this corpus (HAVI has no bundled OCR baseline).

``ImportedSource`` is not defined as a dataclass anywhere in the design docs — only
its required fields are described in prose (preprocess spec §4). This module
defines it here, matching that prose contract; the design docs should be patched
to include the concrete type once its shape is proven out by downstream tickets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ocr_ensemble.ground_truth import author_and_approve_gold_full
from ocr_ensemble.identity import sha256_hex
from ocr_ensemble.records import ImportedGroundTruth

ADAPTER_ID = "havi_failure_mode_v1"
ADAPTER_VERSION = "1.0.0"

# D-* family registry.
VALID_FAMILIES = frozenset(
    {
        "D-GEO",
        "D-PHOT",
        "D-OPT",
        "D-MAT",
        "D-INK",
        "D-INTER",
        "D-NOISE",
        "D-COMP",
        "D-OCC",
        "D-SURR",
        "D-LAYER",
        "D-NEWS-MULTICOL",
        "D-NEWS-CONT",
        "D-NEWS-AD",
        "D-NEWS-MASTHEAD",
        "D-NEWS-CLIP",
        "D-NEWS-HALFTONE",
        "D-NEWS-MICRO",
    }
)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})

HAVI_GOLD_GUIDELINE_ID = "havi_gold_full_transcription_v1"

_HAVI_GOLD_FIXTURE_RELATIVE_PATH = "D-COMP/ocr_d-comp_low-1_20260706.png"

# Full-unit gold transcription of the ticket-02/05 fixture (a scanned Jan 4,
# 1969 "The Black Panther" newspaper article, "Breakfast for School Children"),
# authored by reading the source image directly, without any model output
# exposure (the HAVI authoring rule). Line breaks from the printed
# column layout are not reproduced verbatim in the flowed source text; only
# paragraph breaks (one blank line) are preserved, matching the reading-order
# contract for a linear Evaluation Unit.
_HAVI_GOLD_FIXTURE_TEXT = """PAGE 16 THE BLACK PANTHER SATURDAY, JAN. 4, 1969

Breakfast for School Children

OAKLAND, California -- The National Advisory Cabinet to the Black Panther Party is working with and for St. Augustine Episcopal Church's program: breakfast in the morning for Oakland's school children in the black community.

All children in grammar schools and growing young adults in Junior High Schools can receive free, FULL BREAKFASTS in the mornings before they go to school. The first of these breakfasts will exist one hour before school hours at St. Augustine's Church, 27th and West, and the Black Community Center, at 42nd and Grove Streets, EVERY SCHOOL MORNING.

The National Advisory Cabinet and church members are calling on all mothers and others who want to work with this revolutionary program of making sure that our young have full stomachs before going to school. The schools and the Board of Education should have had this program instituted a long time ago. How can our children learn anything when most of their stomachs are empty? Black people is the Black Community-mothers, welfare recipients, grandmothers, guardians, and others who are trying to raise children in the black community where racists oppress us - are asked to come forth to work and support this needed program. Soul food: grits, eggs, bread, and meat for the stomachs is where it's at when it comes to properly preparing our children for education. LET'S DO IT NOW. Support this community program.

Those who want to volunteer their work every morning or every other morning can come to the BLACK PANTHER PARTY CENTRAL HEADQUARTERS at 3106 Shattuck Ave., Berkeley or contact Father Niel at these numbers: 534-6684, 893-1016. Interested persons may also contact Ruth Beckford Smith at 893-8211 or sign up with other community peoples and citizens for full stomachs and better education of black children.

We urge as many mothers and other black citizens as possible to unite with this COMMUNITY-BLACK PANTHER PROGRAM. We are also asking all businesses throughout the black community to donate the necessary food and utensils to prepare the foods for our children. Call the Black Panther Office at 845-0103 or 845-0104. Everything of value donated to BREAKFAST FOR CHILDREN is tax deductable. Items or funds may be sent c/o St. Augustine Episcopal Church. Just let us know, both black and white communities and citizens, what you can donate in money, time, etc.

Thank you"""


@dataclass(frozen=True)
class ImportedSource:
    """One corpus item mapped into shared vocabulary (preprocess spec §4).

    Not yet a canonical hashed record: ``preprocess_dataset`` (A1) turns this
    into a sealed ``SourcePage``/``EvaluationUnit`` pair.
    """

    dataset_id: str
    dataset_version: str
    adapter_id: str
    adapter_version: str
    dataset_item_id: str
    source_path: Path
    source_genre: str | None
    language: str | None
    license_id: str | None
    seed_family: str | None
    evidence_origin: Literal["natural", "synthetic"]
    layout_complexity: Literal["linear", "layout_dependent", "unknown"]
    evaluation_scope_status: Literal[
        "in_scope_scored", "diagnostic_non_gating", "unsupported_fixture"
    ]


class HaviFailureModeAdapter:
    """Adapter for the project-owned ``aiai-ocr-dataset/`` HAVI corpus."""

    adapter_id = ADAPTER_ID
    adapter_version = ADAPTER_VERSION

    def import_sources(
        self, source_root: Path, *, only_relative_paths: Iterable[str] | None = None
    ) -> list[ImportedSource]:
        """Import corpus items in sorted ``dataset_item_id`` order.

        ``only_relative_paths``, when given, restricts import to exactly the
        named files (relative to ``source_root``) — used by ticket 02 to import
        one fixture without scanning the whole corpus.
        """
        candidates = (
            [source_root / p for p in only_relative_paths]
            if only_relative_paths is not None
            else sorted(
                p
                for p in source_root.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )
        )

        imported: list[ImportedSource] = []
        for path in sorted(candidates, key=lambda p: p.relative_to(source_root).as_posix()):
            if not path.is_file():
                raise FileNotFoundError(f"HAVI source not found: {path}")
            seed_family = self._seed_family_from_path(path, source_root)
            dataset_item_id = path.relative_to(source_root).as_posix()
            imported.append(
                ImportedSource(
                    dataset_id="havi_failure_mode_v1",
                    dataset_version=self._dataset_version(source_root),
                    adapter_id=self.adapter_id,
                    adapter_version=self.adapter_version,
                    dataset_item_id=dataset_item_id,
                    source_path=path,
                    source_genre="newspaper",
                    language="en",
                    license_id=None,
                    seed_family=seed_family,
                    evidence_origin="natural",
                    layout_complexity="linear",
                    evaluation_scope_status="in_scope_scored",
                )
            )
        return imported

    def import_ground_truth(
        self,
        _source_root: Path,
        *,
        evaluation_unit_ids_by_dataset_item_id: dict[str, str],
        author_actor_id: str,
        approver_actor_id: str,
        author_created_at: str,
        approver_created_at: str,
        guideline_id: str = HAVI_GOLD_GUIDELINE_ID,
    ) -> list[ImportedGroundTruth]:
        """Author ``gold_full`` HAVI ground truth for the pinned ticket-05 fixture.

        Ticket 05 scope: exactly the ``D-COMP/ocr_d-comp_low-1_20260706.png``
        fixture. ``evaluation_unit_ids_by_dataset_item_id`` maps each imported
        item's ``dataset_item_id`` to the ``evaluation_unit_id`` preprocess (A1)
        already sealed for it -- preprocess is
        the sole owner of Evaluation Unit identity, and downstream stages
        never recompute or replace it, so this adapter never re-derives
        that ID from the raw source; it is handed the sealed identity by its
        caller. The bare ``DatasetAdapter`` protocol shows
        ``import_ground_truth(self, source_root) -> Iterable[ImportedGroundTruth]``
        without a way to obtain that mapping; this concrete signature adds it
        rather than have the adapter violate that invariant to work around the gap.
        """
        dataset_item_id = _HAVI_GOLD_FIXTURE_RELATIVE_PATH
        evaluation_unit_id = evaluation_unit_ids_by_dataset_item_id.get(dataset_item_id)
        if evaluation_unit_id is None:
            raise KeyError(
                f"no sealed evaluation_unit_id supplied for {dataset_item_id!r}; "
                "run preprocess for this fixture first"
            )

        assertion, (submit_event, approve_event) = author_and_approve_gold_full(
            evaluation_unit_id=evaluation_unit_id,
            text=_HAVI_GOLD_FIXTURE_TEXT,
            guideline_id=guideline_id,
            source="havi_human_authored",
            author_actor_id=author_actor_id,
            approver_actor_id=approver_actor_id,
            author_created_at=author_created_at,
            approver_created_at=approver_created_at,
            source_artifact_sha256=None,
        )
        return [
            ImportedGroundTruth(
                assertion=assertion,
                initial_events=(submit_event, approve_event),
                approval_authorization=None,
            )
        ]

    def import_baselines(self, _source_root: Path):  # noqa: ANN201
        raise NotImplementedError(
            "HAVI has no bundled OCR baseline; import_baselines is not required for this corpus"
        )

    @staticmethod
    def _seed_family_from_path(path: Path, source_root: Path) -> str | None:
        """A valid `D-*` parent folder normally supplies the one seed family;
        a unit outside such a folder has no seed family from
        this heuristic alone.
        """
        for parent in path.relative_to(source_root).parents:
            name = parent.as_posix()
            if name in VALID_FAMILIES:
                return name
        return None

    @staticmethod
    def _dataset_version(_source_root: Path) -> str:
        # No corpus manifest exists yet (preflight territory for
        # the full corpus); ticket 02 pins a fixed literal version so identity
        # is reproducible until a real manifest/release tag is introduced.
        return "aiai-ocr-dataset-2026-08-31"


def source_sha256(path: Path) -> str:
    return sha256_hex(path.read_bytes())


_DATASET_ITEM_ID_SAFE = re.compile(r"[^A-Za-z0-9._/-]")


def sanitize_dataset_item_id(raw: str) -> str:
    if _DATASET_ITEM_ID_SAFE.search(raw):
        raise ValueError(f"unsafe dataset_item_id: {raw!r}")
    return raw
