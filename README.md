# EPC Smart Search

Windows-first local EPC contract assistant built around:

- A bundled prebuilt contract database for the customer runtime
- SQLite + FTS5 for read-heavy clause retrieval
- Query planning and enriched chunk features for context-aware search
- Local semantic-style reranking with stored hashing embeddings
- A Kiewey desktop chat surface (PySide6)
- Optional local Gemma generation with graceful fallback to citation mode
- Preflight checks that warn when contract-bearing artifacts are left under the workspace

## Sensitive files and Git

This repository can be **public** with respect to **how** search works: the SQLite **schema**, indexing pipeline, and Python in `epc_smart_search/` (including `storage.py`) describe the setup only—they do **not** contain contract text.

What must **stay private** (never in the public remote) is **contract information**: full text, chunks, embeddings, and any derived index **inside** database files.

| What | On GitHub (public)? |
|------|---------------------|
| SQLite design, FTS wiring, `ContractStore`, tests using synthetic/minimal data | Yes — OK to show |
| Any **populated** `.db` (working copy, prebuilt bundle, rebuild output) | **No** — contains contract content |
| Contract PDF (`/*.pdf` at repo root) | **No** |
| Logs / OCR cache under app data | **No** |

`.gitignore` blocks all **`*.db`** files (plus journal/WAL/SHM) and **`/*.pdf`**, so indexed data and source PDFs are not pushed by mistake.

Customer builds seed the runtime from a bundled **`contract_store.prebuilt.db`** that you distribute **with the installer or an internal release artifact**. Keep contract-bearing artifacts outside the repo workspace whenever possible; the launcher and packager now warn if PDFs or populated DBs are detected under the workspace.

## Gemma runtime

The app prefers a derived **text-only** Gemma checkpoint when one is available. EPC Smart Search sends text prompts only, so a text-only checkpoint avoids loading image/audio towers.

- Default multimodal checkpoint: `~/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e2b-it/1`
- Default derived text-only checkpoint: `~/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e2b-it-text-only/1`

To export the text-only checkpoint from existing Gemma weights, run from the CUDA-enabled **Gemma Test** environment:

```powershell
& "..\Gemma Test\.venv\Scripts\python.exe" .\export_gemma_text_only.py
```

Optional environment variables:

- `GEMMA_TEXT_ONLY_MODEL_PATH` — explicit text-only checkpoint path
- `GEMMA_MODEL_PATH` — explicit multimodal checkpoint path
- `GEMMA_PREFER_TEXT_ONLY=0` — force multimodal checkpoint
- `EPC_SMART_SEARCH_MODEL_DIR` — explicit bundled AI model directory override
- `EPC_SMART_SEARCH_DISABLE_AI=1` — force Lite behavior for troubleshooting

## Run

1. Install the local app/runtime dependencies:

```powershell
py -3.12 -m pip install -r requirements-dev.txt
```

2. If you want local AI during development, make a Gemma runtime available either through the external **Gemma Test** environment or a direct `EPC_SMART_SEARCH_MODEL_DIR` override.
3. If you need a seeded contract DB for local runs, keep the source artifact outside the workspace and only copy it into the packaged app flow when needed.
4. From this folder:

```powershell
.\launch_epc_smart_search.ps1
```

The launcher now runs a preflight check before starting the UI. The app seeds the bundled prebuilt database into the local app data directory on first run. If the bundled contract data is missing or invalid, the UI fails closed and asks the user to contact support. If AI is unavailable, the app still launches in citation mode and disables generation-only controls.

## Rebuild contract data

Customer builds are read-only. Run a full clean reindex whenever the retrieval architecture changes in a way that can invalidate derived data, including schema updates, chunking changes, fact extraction changes, normalization changes, retrieval routing changes, or embedding-input changes.

Use the internal CLI and keep the source PDF outside the workspace. For a repeatable clean rebuild that also refreshes the live runtime database, run:

```powershell
python -m epc_smart_search.rebuild_contract --pdf C:\secure\contracts\Clean Contract.pdf --out C:\secure\builds\contract_store.prebuilt.new.db --clean-target --install-live-db
```

The clean rebuild flow:

- Deletes the target SQLite file plus its `-journal`, `-wal`, and `-shm` sidecars before rebuilding when `--clean-target` is used
- Replaces the live runtime DB at `%LOCALAPPDATA%\EPC Smart Search\contract_store.db` or the active fallback app-data location when `--install-live-db` is used
- Recreates schema, pages, chunks, structured facts, search features, and embeddings from the current code
- Validates that the rebuilt DB has non-zero pages, chunks, facts, and embeddings, and that the expected fact check is present
- Runs live-pipeline smoke queries for:
  - `What is the configuration of the dew point heaters?`
  - `Summarize the closed cooling water system`

Expected validation output includes:

- `Validation [output]` with row counts for pages, chunks, features, facts, and embeddings
- `Validation [live]` when `--install-live-db` is used
- `Smoke Query [exact]` showing `Retrieval Mode: fact_lookup`
- `Smoke Query [summary]` showing `Retrieval Mode: topic_summary`
- `Run Next Time` with the exact command to repeat

Release flow (contract-bearing `.db` files are never committed to the public repo):

1. Rebuild to a fresh `.db` path outside the workspace.
2. Repackage the customer bundles with either:

```powershell
.\package_epc_smart_search.ps1 -Profile Lite -PrebuiltDbPath C:\secure\builds\contract_store.prebuilt.new.db
.\package_epc_smart_search.ps1 -Profile AI -PrebuiltDbPath C:\secure\builds\contract_store.prebuilt.new.db -ModelDir C:\secure\models\gemma-4-e2b-it-text-only\1
```

3. The packager stages that external DB into the installer payload without requiring a contract-bearing `.db` to live in the repo workspace.

## Develop and test

```powershell
python -m pytest
python -m ruff check .
```

The regression benchmark lives in `tests/test_regression_benchmark.py` and seeds one shared synthetic EPC corpus from `assets/regression_benchmark_corpus.json`. Questions and expectations live in `assets/regression_benchmark_cases.json`, grouped into exact value, system summary, section lookup, page lookup, and no-answer coverage.

To extend the benchmark, add or update sanitized chunks in the corpus file, then add a case with the category, question, and only the expectations needed for that prompt. Prefer reusing existing chunks, keep values and page numbers explicit, and avoid prompts that require Gemma so the suite stays deterministic in CI.

## Package

For Windows portable bundles:

```powershell
.\package_epc_smart_search.ps1 -Profile Lite -PrebuiltDbPath C:\secure\builds\contract_store.prebuilt.new.db
.\package_epc_smart_search.ps1 -Profile AI -PrebuiltDbPath C:\secure\builds\contract_store.prebuilt.new.db -ModelDir C:\secure\models\gemma-4-e2b-it-text-only\1
```

The target customer packaging direction for Lite and AI portable bundles is documented in `PACKAGING_PLAN.md`.

## Release profiles

The current packaging flow supports two portable `--onedir` zip profiles:

- **Lite** — packaged app, assets, and bundled contract DB only. Retrieval and extractive answers always work. No local AI runtime is bundled.
- **AI** — Lite contents plus a bundled Gemma helper executable and bundled text-only model assets. If AI cannot start on the machine, the app degrades to Lite behavior automatically.

For now, the bundled contract database flow stays the same across both releases. The practical difference is whether the bundle includes the local AI runtime and text-only Gemma assets.

## Notes

- **Git:** Public repo shows code and schema only; **no** contract PDFs and **no** populated SQLite files (`*.db`); see `.gitignore`. Local `.epc_smart_search/`, logs, and test caches stay private too.
- **Data directory:** Typically `%LOCALAPPDATA%\EPC Smart Search\`; if that is unavailable, the app may use a `.epc_smart_search` folder under the repo (also ignored).
- **OCR:** Fallback uses Windows WinRT OCR for weak or empty pages.
- **Answers:** If retrieved evidence is weak, the assistant refuses instead of guessing.
- **Dependencies:** `requirements-runtime.txt` covers the desktop/runtime surface, `requirements-dev.txt` adds test/lint/package tooling, and `requirements-gemma.txt` documents the Gemma service extras.
