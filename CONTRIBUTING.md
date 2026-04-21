# Contributing

Thanks for contributing to `MatMul Tiling Analyzer`.

## Development Setup

```powershell
cd .\matmul_tiling_analyzer
python -m pip install -e .
python -m pytest
```

## Recommended Workflow

1. Add or update a source-to-analyzer mapping in `matmul_tiling_analyzer/source_mapping.py`.
2. Implement or refine the branch logic in `matmul_tiling_analyzer/strategies.py` and the related scheduler code.
3. Add at least one representative case under `cases/` when the change affects user-visible behavior.
4. Run `python -m pytest`.
5. Re-run the relevant batch command with `python .\cli.py --input=... --output-dir=...`.
6. Update documentation if the strategy coverage, output format, or source mapping changed.

## Scope Rules

- Prefer source-grounded behavior over heuristic shortcuts.
- Keep Ascend950 behavior traceable to specific host and kernel source files.
- When behavior is inferred rather than directly mirrored, document that clearly.
- Do not silently merge unrelated cleanup with branch-coverage work.

## Documentation Rules

- Keep `README.md` focused on project overview and usage.
- Keep `docs/source_branch_mapping.md` aligned with runtime mapping in `source_mapping.py`.
- Keep `docs/example_outputs.md` in sync with the current text-report format.
- Record user-visible changes in `CHANGELOG.md`.

## Testing Expectations

- Parser changes should include or update tests under `tests/`.
- Strategy or scheduler changes should be validated with one or more real CSV cases.
- If a branch cannot yet be fully modeled, keep the limitation explicit in documentation.
