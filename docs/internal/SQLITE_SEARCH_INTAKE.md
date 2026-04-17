# SQLite Search Intake

Status as of 2026-04-17: `feature/sqlite-search` is not available in this checkout or on `origin`.

When that branch becomes available, review it as an intake source only. Do not merge it wholesale into `Major-rehaul`.

## Acceptance Rubric

- Preserve retrieval-first behavior.
- Preserve or improve exact system and equipment matching.
- Preserve refusal behavior when the contract only contains related text.
- Avoid broadening answers past the user's question.
- Avoid parallel implementations for the same runtime behavior.

## Adopt

- SQLite retrieval, FTS, indexing, or query-planning changes that measurably improve precision or latency.
- Preflight or startup changes that make packaged/runtime behavior more reliable.
- Small operational improvements that reduce rebuild friction without changing the product contract.

## Reject

- UI churn that does not support retrieval precision or runtime stability.
- Prompt-only experiments that shift fact finding away from retrieval.
- Broad-answer logic that weakens grounding or refusal behavior.
- Duplicate code paths that bypass the current retrieval and answer policy flow.

## Superseded

- Any idea already implemented on `Major-rehaul` with better precision, tighter refusal behavior, or stronger tests.

## Intake Procedure

1. Fetch the branch and confirm the exact ref.
2. Compare commits against `Major-rehaul`.
3. Compare touched files and classify each change as `adopt`, `reject`, or `superseded`.
4. Port accepted changes manually in small commits.
5. Add focused regressions for every adopted behavior change.
6. Run the full test and lint suite before and after each port set when practical.

## Useful Commands

```powershell
git fetch origin feature/sqlite-search
git log --oneline --left-right Major-rehaul...origin/feature/sqlite-search
git diff --stat Major-rehaul...origin/feature/sqlite-search
git diff --name-only Major-rehaul...origin/feature/sqlite-search
git diff Major-rehaul...origin/feature/sqlite-search -- epc_smart_search tests
python -m pytest
python -m ruff check .
```
