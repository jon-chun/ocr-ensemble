# OCR Ensemble

This repository is the minimal public mirror of an early-stage project for
probabilistically reconstructing degraded print with an ensemble of OCR and AI
models.

## Current status

The public repository currently contains only the runnable Python scaffold. The
ensemble pipeline is not implemented here yet.

Run it with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv run python main.py
```

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

