# EPC Smart Search

Windows-first local EPC contract assistant built around:

- A bundled prebuilt contract database for the customer runtime
- SQLite + FTS5 for read-heavy clause retrieval
- Query planning and enriched chunk features for context-aware search
- Local semantic-style reranking with stored hashing embeddings
- A Kiewey desktop chat surface (PySide6)
- Local Gemma generation through the **Gemma Test** environment

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

Customer builds seed the runtime from a bundled **`contract_store.prebuilt.db`** that you distribute **with the installer or an internal release artifact**—not by cloning from GitHub.

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
2. For a full local run, place `assets/contract_store.prebuilt.db` on disk next to the project (it is **not** in Git; obtain it from your internal build or rebuild flow below).
3. From this folder:

```powershell
.\launch_epc_smart_search.ps1
```

The app seeds the bundled prebuilt database into the local app data directory on first run. If the bundled contract data is missing or invalid, the UI fails closed and asks the user to contact support.

## Rebuild contract data

Customer builds are read-only. To rebuild the contract data, use the internal CLI and write to a fresh output file:

```powershell
python -m epc_smart_search.rebuild_contract --pdf .\Clean Contract.pdf --out .\assets\contract_store.prebuilt.new.db
```

The rebuild utility:

- Uses the same chunking, feature, and embedding pipeline as the app
- Validates the rebuilt SQLite database before reporting success
- Refuses to overwrite the live app database or an existing output file

Release flow (contract-bearing `.db` files are never committed to the public repo):

1. Rebuild to a fresh `.db` path.
2. Copy the validated output to `assets\contract_store.prebuilt.db` on your machine for packaging only.
3. Repackage the EXE with `.\package_epc_smart_search.ps1` (the installer carries the DB; GitHub does not).

## Develop and test

```powershell
python -m pytest
```

## Package

For a Windows bundle (requires `pyinstaller`):

```powershell
.\package_epc_smart_search.ps1
```

## Notes

- **Git:** Public repo shows code and schema only; **no** contract PDFs and **no** populated SQLite files (`*.db`); see `.gitignore`. Local `.epc_smart_search/`, logs, and test caches stay private too.
- **Data directory:** Typically `%LOCALAPPDATA%\EPC Smart Search\`; if that is unavailable, the app may use a `.epc_smart_search` folder under the repo (also ignored).
- **OCR:** Fallback uses Windows WinRT OCR for weak or empty pages.
- **Answers:** If retrieved evidence is weak, the assistant refuses instead of guessing.
