# Contributing

Thanks for contributing to `MatMulV3 Tiling Replay`.

## Development Setup

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Recommended Workflow

1. Add or update source trace notes in `docs/source-map.md`.
2. Implement or refine branch logic in `src/matmul_tiling_replay/replay.py`.
3. Add at least one representative case under `cases/` when the change affects user-visible behavior.
4. Run `python -m unittest discover -s tests -v`.
5. Re-run the relevant batch command with `python .\cli.py --input=... --output-dir=...`.
6. Update documentation if the strategy coverage, output format, or source mapping changed.

## Scope Rules

- Prefer source-grounded behavior over heuristic shortcuts.
- Keep MatMulV3 behavior traceable to specific host and kernel source files.
- When behavior is inferred rather than directly mirrored, document that clearly.
- Do not silently merge unrelated cleanup with branch-coverage work.

## Documentation Rules

- Keep `README.md` focused on project overview and usage.
- Keep `docs/source-map.md` aligned with runtime mapping in `src/matmul_tiling_replay/replay.py`.
- Keep `docs/core-task-schema.md` in sync with `cores.json`.
- Record major workflow decisions in `docs/work-log.md`.

## Testing Expectations

- Parser changes should include or update tests under `tests/`.
- Strategy or scheduler changes should be validated with one or more real CSV cases.
- If a branch cannot yet be fully modeled, keep the limitation explicit in documentation.
