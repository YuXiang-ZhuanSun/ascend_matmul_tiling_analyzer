# Local Panel

`matmul-tiling-panel` launches a local browser UI for MatMulV3 tiling diagnosis.

It is designed to work the same way on Windows, macOS, and Linux. The panel uses only the Python standard library for serving the UI, so the package stays lightweight and easy to install.

## Install

For development:

```powershell
python -m pip install -e .
```

From a built wheel:

```powershell
python -m pip install dist/matmul_tiling_analyzer-*.whl
```

## Launch

```powershell
matmul-tiling-panel
```

If the script directory is not on `PATH`, use the module form:

```powershell
python -m matmul_tiling_analyzer.panel
```

Custom host or port:

```powershell
matmul-tiling-panel --host 127.0.0.1 --port 8765
```

Run without opening a browser automatically:

```powershell
matmul-tiling-panel --no-browser
```

## What The Panel Shows

- manual `M/K/N` shape input
- CSV testcase upload
- selected strategy and selected branch
- decoded `tiling_key`
- tiling fields, source mapping, and kernel dispatch
- 32-core workload heatmap
- per-core task detail drill-down
- top imbalance cores by estimated work units

## Build Release Artifacts

The project can be packaged as normal Python distribution artifacts:

```powershell
python -m pip install build
python -m build
```

This creates:

```text
dist/
  matmul_tiling_analyzer-<version>.tar.gz
  matmul_tiling_analyzer-<version>-py3-none-any.whl
```

The repository also includes a GitHub Actions release workflow. Push a tag like `v0.3.0` to build wheel and source distribution artifacts:

```powershell
git tag v0.3.0
git push origin v0.3.0
```
