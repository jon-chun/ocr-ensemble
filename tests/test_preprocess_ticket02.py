"""Ticket 02: A0/A1 import one fixture page and seal its identity.

Exercises the real ``aiai-ocr-dataset/D-COMP`` fixture end to end: adapter ->
preprocess -> sealed source_pages/evaluation_units/input_variants stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_ensemble.adapters.havi import HaviFailureModeAdapter
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset
from ocr_ensemble.storage import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "aiai-ocr-dataset"
FIXTURE_RELATIVE_PATH = "D-COMP/ocr_d-comp_low-1_20260706.png"


@pytest.fixture()
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "preprocess-out"


@pytest.fixture()
def request_(output_root: Path) -> PreprocessRequest:
    return PreprocessRequest(
        dataset_root=DATASET_ROOT,
        output_root=output_root,
        fixture_relative_paths=(FIXTURE_RELATIVE_PATH,),
    )


def test_fixture_exists():
    assert (DATASET_ROOT / FIXTURE_RELATIVE_PATH).is_file()


def test_adapter_imports_exactly_one_source_with_expected_seed_family():
    adapter = HaviFailureModeAdapter()
    imported = adapter.import_sources(
        DATASET_ROOT, only_relative_paths=(FIXTURE_RELATIVE_PATH,)
    )
    assert len(imported) == 1
    source = imported[0]
    assert source.dataset_item_id == FIXTURE_RELATIVE_PATH
    assert source.seed_family == "D-COMP"
    assert source.evidence_origin == "natural"
    assert source.layout_complexity == "linear"
    assert source.evaluation_scope_status == "in_scope_scored"


def test_preprocess_seals_one_source_page_evaluation_unit_and_variant(
    request_: PreprocessRequest,
):
    artifacts = preprocess_dataset(request_)

    assert len(artifacts.source_pages) == 1
    assert len(artifacts.evaluation_units) == 1
    assert len(artifacts.input_variants) == 1

    source_page = artifacts.source_pages[0]
    assert source_page.source_page_id.startswith("source_page_sha256:")
    assert len(source_page.source_sha256) == 64
    assert source_page.dataset_id == "havi_failure_mode_v1"
    assert source_page.dataset_item_id == FIXTURE_RELATIVE_PATH

    evaluation_unit = artifacts.evaluation_units[0]
    assert evaluation_unit.source_page_id == source_page.source_page_id
    assert evaluation_unit.selector is None  # whole_image
    assert evaluation_unit.layout_complexity == "linear"

    variant = artifacts.input_variants[0]
    assert variant.evaluation_unit_id == evaluation_unit.evaluation_unit_id
    assert variant.source_page_id == source_page.source_page_id
    assert variant.variant_kind == "natural_baseline"
    assert variant.mitigation_kind is None
    assert variant.transform_chain == ()
    assert variant.perturbation_provenance is None


def test_preprocess_performs_no_geometry_or_restoration_transform(
    request_: PreprocessRequest,
):
    # natural_baseline must be source-faithful: same pixel dimensions as the
    # source (preprocess spec §5 — no deskew/dewarp/denoise/contrast/restoration).
    artifacts = preprocess_dataset(request_)
    source_page = artifacts.source_pages[0]
    variant = artifacts.input_variants[0]
    assert variant.width_px == source_page.width_px
    assert variant.height_px == source_page.height_px


def test_sealed_stores_are_written_to_disk(
    request_: PreprocessRequest, output_root: Path
):
    preprocess_dataset(request_)

    source_pages = read_jsonl(output_root / "source_pages.jsonl")
    evaluation_units = read_jsonl(output_root / "evaluation_units.jsonl")
    input_variants = read_jsonl(output_root / "input_variants.jsonl")

    assert len(source_pages) == 1
    assert len(evaluation_units) == 1
    assert len(input_variants) == 1

    # every JSONL line must be valid, independently parseable JSON (sealed
    # store discipline)
    for path in (
        output_root / "source_pages.jsonl",
        output_root / "evaluation_units.jsonl",
        output_root / "input_variants.jsonl",
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_rerunning_preprocess_is_idempotent(request_: PreprocessRequest):
    first = preprocess_dataset(request_)
    second = preprocess_dataset(request_)

    assert first.source_pages[0].source_page_id == second.source_pages[0].source_page_id
    assert (
        first.evaluation_units[0].evaluation_unit_id
        == second.evaluation_units[0].evaluation_unit_id
    )
    assert (
        first.input_variants[0].page_input_variant_id
        == second.input_variants[0].page_input_variant_id
    )


def test_rerunning_preprocess_creates_no_duplicate_records(
    request_: PreprocessRequest, output_root: Path
):
    preprocess_dataset(request_)
    preprocess_dataset(request_)

    source_pages = read_jsonl(output_root / "source_pages.jsonl")
    evaluation_units = read_jsonl(output_root / "evaluation_units.jsonl")
    input_variants = read_jsonl(output_root / "input_variants.jsonl")

    assert len(source_pages) == 1
    assert len(evaluation_units) == 1
    assert len(input_variants) == 1


def test_output_root_may_not_overlap_dataset_root():
    bad_request = PreprocessRequest(
        dataset_root=DATASET_ROOT,
        output_root=DATASET_ROOT / "nested-output",
        fixture_relative_paths=(FIXTURE_RELATIVE_PATH,),
    )
    with pytest.raises(ValueError):
        preprocess_dataset(bad_request)
