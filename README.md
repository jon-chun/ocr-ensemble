# OCR Ensemble

*STATUS: (as of Sep 1, 2026) - close to functional MVP 0.1 release, finalizing ensemble model selection, configuration, and budgets*

This repository is the minimal public mirror of an early-stage project for
probabilistically reconstructing degraded print with an ensemble of OCR and AI
models.

## Current status

The pipeline stages import/seal, manifest/budget/dispatch (against a stub
provider), Page-Model Result derivation, character alignment with Consensus
Entropy and equal-vote fusion, an append-only ground-truth journal, score
materialization, structural postprocess validation, and a minimal analyze
report are implemented under `src/ocr_ensemble/`. Real (non-stub) provider
adapters are not yet in this mirror.

Run it with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run python main.py
```

## Tests

```sh
uv run pytest
```

Most tests are self-contained. Eight test files load a pinned sample image
from `aiai-ocr-dataset/D-COMP/` that is not included in this public mirror
(see "Data" below) and will fail with `FileNotFoundError` until that asset
is published or you supply your own fixture at the same path.

## Data

The development corpus is not included in this public mirror. Its source,
provenance, and redistribution rights are being reviewed before any assets are
published. See [`aiai-ocr-dataset/README.md`](aiai-ocr-dataset/README.md).

Large benchmark exports such as Parquet files are intentionally ignored.

## Public mirror policy

Only public-facing code, configuration, and redistribution-safe data belong in
this repository. Internal planning documents, development notes, agent or chat
sessions, review artifacts, local paths, downloaded third-party datasets, and
generated outputs are excluded, including from Git history.

## License

No open-source license has been selected yet. Unless a file states otherwise,
all rights are reserved.

