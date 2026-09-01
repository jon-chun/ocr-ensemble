"""Public entry point for the OCR Ensemble project.

The full CLI is ``ocr_ensemble.cli:main`` (installed as the ``ocr-ensemble``
console script). This file exists so ``uv run python main.py`` also works
without installing the package.
"""

from ocr_ensemble.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

