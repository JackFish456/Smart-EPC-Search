# Packaging Plan: Lite + AI Portable Windows Bundles

## Summary

Build two offline `--onedir` Windows distributions and ship them as zip files, not a single-file EXE.

- `EPC Smart Search Lite`: PySide app + bundled contract DB + current local retrieval/extractive answer path only.
- `EPC Smart Search AI`: everything in Lite, plus the bundled fine-tuned Gemma runtime and model assets.
- Core requirement: the app must always remain usable on no-GPU laptops. AI features are optional capabilities, not startup requirements.
- Target hardware policy:
  - No dedicated GPU: Lite behavior only.
  - 8 GB VRAM NVIDIA GPU: preferred AI path using the text-only Gemma checkpoint and lowest-VRAM load settings.
  - Unsupported or weak hardware inside the AI build: degrade to Lite behavior automatically.

## Key Changes

### Runtime and UX

- Remove the packaged app's dependency on the external `Gemma Test` project and its separate Python path.
- Make Gemma capability-based:
  - app starts without Gemma
  - retrieval and citations always work
  - generation is enabled only after local capability detection succeeds
- Default AI runtime behavior:
  - use the derived text-only fine-tuned Gemma checkpoint only
  - keep multimodal and image towers out of customer builds
  - disable or hide `Deep Think` when AI is unavailable
  - on fallback, show a short status message such as `AI mode unavailable on this machine; citation mode is active.`

### Build and packaging

- Replace the current single packaging script with one build entrypoint that supports `Lite` and `AI` profiles.
- Standardize on PyInstaller `--onedir` for both profiles.
- Lite bundle contents:
  - app executable
  - `assets/`
  - bundled prebuilt contract DB
  - no `torch`, no `transformers`, no model weights
- AI bundle contents:
  - Lite contents
  - packaged Gemma service and runtime dependencies
  - bundled fine-tuned text-only Gemma weights in a known relative folder inside the bundle
- Distribute both as portable zips:
  - `EPC-Smart-Search-Lite-win64.zip`
  - `EPC-Smart-Search-AI-win64.zip`

### Interfaces and config

- Add a build profile interface, either `package_epc_smart_search.ps1 -Profile Lite|AI` or two thin wrappers over one shared script.
- Replace external-path assumptions with bundled relative-path resolution for model and runtime assets.
- Add only these runtime overrides:
  - `EPC_SMART_SEARCH_MODEL_DIR` for explicit model override
  - `EPC_SMART_SEARCH_DISABLE_AI=1` to force Lite behavior for troubleshooting
- Keep the contract DB seeding flow as-is.

### Model loading policy

- AI bundle uses the fine-tuned Gemma text-only checkpoint by default.
- First choice on supported NVIDIA GPUs: quantized low-VRAM load path already used by the repo.
- If GPU load fails or no supported GPU is present:
  - do not block app launch
  - skip generation
  - continue with Lite retrieval and extractive answers
- Keep summary and deep-answer generation opt-in and conservative to avoid extra VRAM pressure.

## Test Plan

- Lite build launches on a machine with no Python, no Gemma assets, and no GPU.
- Lite build answers retrieval questions with citations and extractive text and never attempts to launch external Gemma paths.
- AI build launches offline with bundled weights and no Python installed system-wide.
- AI build on an 8 GB VRAM NVIDIA machine enables generation and uses the text-only model path.
- AI build on CPU-only hardware degrades cleanly to Lite behavior without crashing or disabling search.
- Missing or invalid bundled DB still fails closed exactly as current tests expect.
- Packaging smoke tests verify both zips contain the correct asset sets and start from a clean machine profile.

## Assumptions and defaults

- Windows-only distribution for v1.
- Portable zip delivery is the primary rollout path; code signing and installer work are deferred and should not change runtime layout.
- The fine-tuned Gemma model is available in a bundleable local format compatible with the current runtime.
- We are not targeting a single-file EXE for v1 because model assets and support files make `--onedir` the safer offline packaging format.
- "As many users as possible" means universal Lite support plus optional local AI for the subset of users with supported 8 GB VRAM hardware.

## Current status

This file captures the target packaging direction. The repository is not fully at this target yet.

Current repo state as of 2026-04-09:

- Tests pass and lint is clean on the current branch.
- The existing packager builds a single Windows app bundle and still expects an external prebuilt contract DB path.
- Launch and Gemma service flows still assume the external `Gemma Test` environment for AI support.
- The runtime already prefers a derived text-only checkpoint when one is present.
- The runtime already has a low-VRAM CUDA path plus CPU and offload fallback warnings.
- The repo does not yet expose `Lite` and `AI` packaging profiles, portable zip assembly, or bundled relative AI asset resolution.

## Implementation gaps

To reach the target plan, the next implementation pass should cover:

- Packaging scripts:
  - add `Lite` and `AI` profile support
  - switch customer bundles to explicit `--onedir` output
  - zip the final profile folders
- Runtime resolution:
  - stop depending on `Gemma Test` for packaged builds
  - resolve bundled model and runtime assets relative to the packaged app
  - honor `EPC_SMART_SEARCH_MODEL_DIR` and `EPC_SMART_SEARCH_DISABLE_AI`
- Capability handling:
  - make AI startup non-blocking
  - gate generation-only UI based on runtime capability detection
  - keep retrieval available when AI is absent or unsupported
- Validation:
  - add smoke tests for Lite and AI artifact contents
  - add packaged-startup checks that do not require a system Python install
