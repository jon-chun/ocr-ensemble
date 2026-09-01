#!/usr/bin/env python3
"""Thin CLI wrapper. All logic lives in ``ocr_ensemble.postprocess``."""

from ocr_ensemble.postprocess import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
