# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

pyCircuit is a Python-based hardware description framework that compiles Python functions to synthesizable RTL through MLIR. It has two main components:

1. **Python frontend** (`compiler/frontend/pycircuit/`) — the main development interface. Installed editably via `pip install -e ".[dev]"`.
2. **MLIR compiler backend** (`compiler/mlir/`) — C++ project that builds `pycc` and `pyc-opt` binaries via CMake+Ninja. Requires LLVM/MLIR development libraries.

### Running the Python frontend

- Emit MLIR from a design: `PYTHONPATH=compiler/frontend:designs python3 -m pycircuit.cli emit designs/examples/counter/counter.py -o /tmp/counter.pyc`
- Run a design directly: `PYTHONPATH=compiler/frontend:designs python3 designs/examples/counter/counter.py`
- Standard commands documented in `README.md` and `docs/getting-started/quickstart.md`.

### Lint and test

- Lint: `ruff check .`, `black --check .`, `mypy compiler/frontend/pycircuit` (mypy has known existing errors, CI runs it with `|| true`)
- Tests: `pytest` (the `tests/` directory does not exist yet; exit code 5 is expected)
- API hygiene gate: `python3 flows/tools/check_api_hygiene.py compiler/frontend/pycircuit designs/examples docs README.md`

### C++ backend build (pycc) — known limitation

The C++ backend code currently uses MLIR APIs (`GreedyRewriteConfig::enableFolding()`, missing `ModuleOp` include) that are not available in any Ubuntu-packaged LLVM version (tested 18, 19, 20). It appears to target a bleeding-edge LLVM from source (post-20 / main branch). If you need `pycc`, you must either:

- Build LLVM/MLIR from source (main branch), or
- Obtain a pre-built `pycc` binary and place it at `build/bin/pycc`.

The Python frontend (emit, design compilation to MLIR IR) works independently of `pycc`.

### System dependencies for C++ backend

When LLVM compatibility is fixed, the build requires: `sudo apt-get install -y ninja-build llvm-XX-dev libmlir-XX-dev mlir-XX-tools libzstd-dev libstdc++-14-dev` (replace XX with appropriate version), then `bash flows/scripts/pyc build`.

### Environment notes

- `~/.local/bin` must be on PATH for pip-installed tools (`pycircuit`, `pytest`, `ruff`, `black`, `mypy`). Already added to `~/.bashrc`.
