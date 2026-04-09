# EPC Smart Search

Windows-first local EPC contract assistant built around:

- `Clean Contract.pdf` as the single source of truth
- SQLite + FTS5 for read-heavy clause retrieval
- local semantic-style reranking with stored embeddings
- a Kiewey desktop chat surface
- local Gemma generation through the `Gemma Test` environment

The contract PDF is **not** stored in this repository (it is listed in `.gitignore`). After you clone or copy the project, place your own `Clean Contract.pdf` in the **project root** (same folder as `launch_epc_smart_search.ps1`) before the first index build.

## Gemma Runtime

The app now prefers a derived **text-only** Gemma checkpoint when one is available. This is useful because EPC Smart Search sends text prompts only, so a text-only checkpoint avoids carrying the image/audio towers in memory.

- Default multimodal checkpoint: `~/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e2b-it/1`
- Default derived text-only checkpoint: `~/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e2b-it-text-only/1`

To export the text-only checkpoint locally from the existing Gemma weights, run this from the CUDA-enabled `Gemma Test` environment:

```powershell
& "..\Gemma Test\.venv\Scripts\python.exe" .\export_gemma_text_only.py
```

Optional environment variables:

- `GEMMA_TEXT_ONLY_MODEL_PATH` to point at a specific text-only checkpoint
- `GEMMA_MODEL_PATH` to point at a specific multimodal checkpoint
- `GEMMA_PREFER_TEXT_ONLY=0` to force the runtime back to the multimodal checkpoint

## Run

1. Make sure the desktop-level `Gemma Test` project still has its `.venv` and model files.
2. Confirm `Clean Contract.pdf` is in this folder (see above).
3. In this folder, run:

```powershell
.\launch_epc_smart_search.ps1
```

The first run builds the contract index. After that, clicking Kiewey opens the chat and answers from cited contract text only.

## Package

If you want a Windows bundle and already have `pyinstaller` available:

```powershell
.\package_epc_smart_search.ps1
```

## Notes

- Git ignores `Clean Contract.pdf`, local SQLite files under `.epc_smart_search/`, and Python/virtualenv cruft—see `.gitignore` for the full list.
- The SQLite database is stored under `%LOCALAPPDATA%\EPC Smart Search\`.
- OCR fallback uses Windows WinRT OCR for weak or empty pages.
- If the contract evidence is weak, the assistant refuses instead of guessing.
