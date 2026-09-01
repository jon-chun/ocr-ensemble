#!/usr/bin/env python3
"""Thin CLI wrapper. All logic lives in ``ocr_ensemble.preprocess``."""

from ocr_ensemble.preprocess import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
