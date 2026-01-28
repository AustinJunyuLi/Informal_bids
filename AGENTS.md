# Repository Guidelines

## Project structure & ownership
- `src/informal_bids/`: core package (data generators, samplers, sensitivity, plots).
- `reports/` and `meeting/`: documentation and dated notes; source-of-truth for assumptions.
- `contrib/`: scratch and external references (non-canonical).
- `outputs/`: generated artifacts; keep untracked.
- `data/`: placeholder for future datasets.

## How to run and reproduce
Use `README.md` for the full setup, CLI entry points, and OS-specific steps. Keep any new run instructions there so collaborators have a single source of truth. When adding new entry points or outputs, update `README.md` accordingly.

## Coding style & naming conventions
- Python only; 4-space indentation; follow PEP 8.
- `snake_case` for functions/variables, `PascalCase` for classes.
- Prefer dataclasses for parameter bundles (see `data.py`).
- Keep numerical logic explicit; vectorize only when it improves clarity.

## Testing and verification
- No formal test suite yet. Validate changes by re-running the relevant baseline/sensitivity workflows and inspecting outputs under `outputs/`.
- If you add tests, document how to run them in `README.md` and keep them fast by default.

## Commit and PR guidelines
- Use short, imperative commit messages (e.g., “Add cutoff diagnostics”).
- Keep commits scoped to one logical change; avoid committing `outputs/`.
- For PRs: summarize what changed and why, include repro steps, and link any updated figures or reports.

## Operational notes
- `pyproject.toml` defines CLI scripts and dependencies.
- Reports should reference plots from `outputs/` (centralized). Do not create per-report figure folders (`reports/**/figures` is ignored).
- If JIT compilation is an issue during debugging, set `NUMBA_DISABLE_JIT=1`.
