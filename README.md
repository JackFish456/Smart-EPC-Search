# EPC Smart Search

Windows-first local EPC contract assistant built around:

- A bundled prebuilt contract database for the customer runtime
- SQLite + FTS5 for read-heavy clause retrieval
- Query planning and enriched chunk features for context-aware search
- Local semantic-style reranking with stored hashing embeddings
- A Kiewey desktop chat surface (PySide6)
- Local Gemma generation through the **Gemma Test** environment
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

## Run

1. Ensure the desktop **Gemma Test** project has its `.venv` and model files.
2. Install the local app/runtime dependencies:

```powershell
py -3.12 -m pip install -r requirements-dev.txt
```

3. If you need a seeded contract DB for local runs, keep the source artifact outside the workspace and only copy it into the packaged app flow when needed.
4. From this folder:

```powershell
.\launch_epc_smart_search.ps1
```

The launcher now runs a preflight check before starting the UI. The app seeds the bundled prebuilt database into the local app data directory on first run. If the bundled contract data is missing or invalid, the UI fails closed and asks the user to contact support.

## Rebuild contract data

Customer builds are read-only. To rebuild the contract data, use the internal CLI and write to a fresh output file. The source PDF must be supplied explicitly via `--pdf` or `EPC_CONTRACT_PDF`, and it must live outside the repo workspace:

```powershell
python -m epc_smart_search.rebuild_contract --pdf C:\secure\contracts\Clean Contract.pdf --out C:\secure\builds\contract_store.prebuilt.new.db
```

The rebuild utility:

- Uses the same chunking, feature, and embedding pipeline as the app
- Validates the rebuilt SQLite database before reporting success
- Refuses to overwrite the live app database or an existing output file

Release flow (contract-bearing `.db` files are never committed to the public repo):

1. Rebuild to a fresh `.db` path outside the workspace.
2. Repackage the EXE with `.\package_epc_smart_search.ps1 -PrebuiltDbPath C:\secure\builds\contract_store.prebuilt.new.db`.
3. The packager stages that external DB into the installer payload without requiring a contract-bearing `.db` to live in the repo workspace.

## Develop and test

```powershell
python -m pytest
python -m ruff check .
```

## Package

For a Windows bundle (requires the dev requirements and an external prebuilt DB path):

```powershell
.\package_epc_smart_search.ps1 -PrebuiltDbPath C:\secure\builds\contract_store.prebuilt.new.db
```

The target customer packaging direction for Lite and AI portable bundles is documented in `PACKAGING_PLAN.md`.

## Release profiles

The current packaging flow supports one Windows app bundle plus two **Gemma runtime/model profiles**:

- **GPU-capable profile** — ship the packaged app with the CUDA-enabled **Gemma Test** environment, a preferred **text-only** checkpoint, and the CUDA PyTorch + `bitsandbytes` stack so Gemma can run in 4-bit NF4 on a dedicated GPU.
- **CPU / no dedicated GPU profile** — ship the same packaged app with a **text-only** checkpoint and a CPU-compatible Gemma environment. This works without a dedicated GPU, but responses will be slower.

Recommended internal release naming:

- `EPC Smart Search - GPU`
- `EPC Smart Search - Standard`

For now, the EXE and bundled contract database can stay the same across both releases. The practical difference is the accompanying Gemma environment and model payload, not the PyInstaller app itself.

## Notes

- **Git:** Public repo shows code and schema only; **no** contract PDFs and **no** populated SQLite files (`*.db`); see `.gitignore`. Local `.epc_smart_search/`, logs, and test caches stay private too.
- **Data directory:** Typically `%LOCALAPPDATA%\EPC Smart Search\`; if that is unavailable, the app may use a `.epc_smart_search` folder under the repo (also ignored).
- **OCR:** Fallback uses Windows WinRT OCR for weak or empty pages.
- **Answers:** If retrieved evidence is weak, the assistant refuses instead of guessing.
- **Dependencies:** `requirements-runtime.txt` covers the desktop/runtime surface, `requirements-dev.txt` adds test/lint/package tooling, and `requirements-gemma.txt` documents the Gemma service extras.
