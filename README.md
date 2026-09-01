# OCR Ensemble

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

### MVP status

**Vertical slice 1 (tickets 01–08): complete.** A single fixture page runs
end-to-end through the full A0–A9 pipeline against a stub model —
import/seal → manifest/budget/dispatch → result derivation →
alignment/Consensus Entropy/fusion → ground-truth journal → score
materialization → postprocess validation → analyze report. 150 tests pass in
the development repo.

| # | Ticket | Status | Notes |
|---|--------|--------|-------|
| 01 | Scaffold + canonical record types | ✅ Done | `records.py`, `identity.py`, `storage.py` |
| 02 | A0/A1 import + seal one fixture | ✅ Done | HAVI adapter (`adapters/havi.py`) |
| 03 | A2/A3 manifest, budget, stub dispatch | ✅ Done | `manifest.py`, `budget.py`, `dispatch.py` |
| 04 | A4 derive Page-Model Result | ✅ Done | `results.py` — closed a real spec gap (terminal-outcome precedence table) |
| 05 | A6 ground-truth journal | ✅ Done | `ground_truth.py` |
| 06 | A5 alignment / CE / fusion | ✅ Done | `align.py` — caught and fixed a real alignment bug pre-merge |
| 07 | A7 score materialization | ✅ Done | `scoring.py` — full 44-field `BenchmarkObservation` per spec |
| 08 | A8/A9 postprocess + analyze (vertical slice 1) | ✅ Done | `run_pipeline.py`, `postprocess.py`, `analyze.py` — first real clean-checkout demo |
| 10 | Local deterministic mitigation (deskew) | ✅ Done | `mitigate.py` — implemented alongside 05, out of strict order |
| 09 | Real provider adapters | 🟡 Preflight only | Live verification done and committed (`docs/registries/verified-model-registry_20260901.md`) — confirmed model IDs/pricing for OpenAI, Anthropic, Google, xAI, OpenRouter. No adapter code written yet. |
| 14 | BLN600 adapter + dataset split | ⛔ Blocked | Needs its own live preflight (dataset acquisition/layout facts) before it can move to ready |
| 11 | AI-model enhancement (paired experiment) | ⬜ Not started | Blocked on 09 |
| 12 | Full N-member alignment/CE-per-pair | ⬜ Not started | Blocked on 06 (done) + 09 |
| 13 | Postprocess anomaly detection + queues | ⬜ Not started | Blocked on 08 (done) + 09 |
| 15 | Scientific Acceptance Scorecard | ⬜ Not started | Blocked on 09, 12, 13, 14 |

**Frontier (unblocked, awaiting work):** ticket 09 (adapters ready to
implement against verified facts) and ticket 14 (blocked pending its own
preflight).

This public mirror includes code/tests/utils through ticket 10 only (no
provider adapters beyond the HAVI import stub, no dataset).

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

## Code base analysis

Snapshot of the development repo (`ocr-ensemble-dev`), not this mirror.

**Size**
- Source: 5,041 LOC across 19 files (`src/ocr_ensemble/`)
- Tests: 3,112 LOC across 13 files — a 0.62:1 test-to-source ratio
- Largest module: `records.py` (1,153 LOC — canonical record types and sealing logic)

**Test coverage:** 97% (1,898 statements, 55 missed), 150/150 tests passing.
Only `cli.py` (0%) is untested — it's a `NotImplementedError` stub, not real
logic. Everything else is 90–100%; the misses are mostly defensive/edge-case
branches (e.g. malformed-input guards in `postprocess.py`, `results.py`).

**Complexity** (via `radon`): average A (3.15) across 175 functions/methods,
no file below "A" maintainability. Two isolated C-complexity hotspots are
worth naming — `align_hypotheses`/`align_pair` (Levenshtein DP with a
multi-way merge) and `_derive_outcome` (D, the terminal-outcome precedence
DAG) — both inherently branchy domain logic rather than disorganized code,
and both are 100%-covered. The maintainability floor is `records.py` at
31.5 ("A", but the weakest), a function of raw size rather than disorder.

**Static analysis:**
- `ruff`: 22 findings, all low-severity style (unsorted imports, deprecated
  typing imports, unused imports) — 21 auto-fixable, zero correctness issues.
- `mypy --ignore-missing-imports`: 27 errors in 3 files (`scoring.py`,
  `preprocess.py`, `mitigate.py`), all the same root cause — functions typed
  to return specific record subclasses (`BenchmarkObservation`, `SourcePage`,
  etc.) but returning the `CanonicalRecord` base type. A real type-safety
  gap, not a runtime bug (tests pass), but worth tightening.
- 0 TODO/FIXME/XXX markers in source.

**Documentation:** ~49% of functions/classes have docstrings, concentrated
on public APIs and non-obvious logic.

**Dependencies:** 7 direct in the development repo (`datasets`, `gradio`,
`huggingface-hub`, `opencv-python-headless`, `pillow`, `requests`, `tqdm`).
This public mirror trims that to 3 (`numpy`, `opencv-python-headless`,
`pillow`), since `datasets`/`gradio`/`huggingface-hub` are dev-only tooling
not imported by any synced module.

**Bottom line:** high test coverage and low complexity for a project at MVP
vertical-slice-1 stage. The one real actionable gap is the 27 mypy
return-type mismatches noted above, worth resolving before ticket 09 adds
more code depending on those return types.

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
