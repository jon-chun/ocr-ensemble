"""Top-level ``ocr-ensemble`` console entry point.

A unified subcommand CLI (``run``, ``preprocess``, ``dispatch``,
``materialize-results``, ``consensus``, ``snapshot-gt``, ``score``,
``postprocess``, ``analyze``) is not implemented yet. Each pipeline stage
currently has its own standalone script under ``utils/``:
``ocr-ensemble-fix-run-pipeline.py``, ``ocr-ensemble-fix-postprocess.py``,
``ocr-ensemble-fix-analyze.py``. This stub only establishes the declared
console entry point so ``uv sync`` / ``uv run ocr-ensemble`` resolve to a
real, if unimplemented, target.
"""

from __future__ import annotations


def main(_argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "the unified ocr-ensemble CLI is not implemented yet; use the "
        "standalone scripts under utils/ for each pipeline stage"
    )


if __name__ == "__main__":
    raise SystemExit(main())
