# Pitaya: environment setup with uv

Use this guide to recreate the project's Python environment. The recorded setup uses NumPy and pandas, with ipykernel for notebooks. Dependency versions are recorded in `uv.lock`.

## 1. Install the prerequisites

Install Git and ensure you have access to [the repository](https://github.com/arfarfq/pitaya). Install uv once:

**macOS / Linux**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal if needed, then check `uv --version`. Anaconda is not required.

## 2. Clone and install

```sh
git clone https://github.com/arfarfq/pitaya.git
cd pitaya
uv sync --locked
```

Run project commands from this folder. uv creates `.venv` and installs the locked dependencies, including the default `dev` group containing ipykernel. It uses the Python version requested by `.python-version`, downloading a compatible interpreter if needed.

`--locked` prevents changes to `uv.lock` and reports an error if it is missing or inconsistent with `pyproject.toml`. Coworkers do not need to run `uv init` or add the existing dependencies again.

## 3. Check the environment and run code

```sh
uv run --locked python -c "import sys, numpy, pandas, ipykernel; print(sys.executable); print(sys.version); print('NumPy:', numpy.__version__); print('pandas:', pandas.__version__)"
```

The executable should be inside the project's `.venv`. To run a script, replace `path/to/script.py` with its actual path:

```sh
uv run --locked python path/to/script.py
```

`uv run` uses the project environment; manual activation is unnecessary.

## 4. Use notebooks in VS Code

Install VS Code's Python and Jupyter extensions, open the project folder, then open your notebook. In **Select Kernel → Python Environments**, select:

- macOS / Linux: `.venv/bin/python`
- Windows: `.venv\Scripts\python.exe`

ipykernel is already included. In a notebook, `import sys; print(sys.executable)` confirms which interpreter is running.

## 5. Get project updates

With your local work committed or safely saved:

```sh
git pull --ff-only
uv sync --locked
```

When intentionally adding a dependency, use `uv add PACKAGE` (or `uv add --dev PACKAGE` for development tools), then commit both `pyproject.toml` and `uv.lock`. Keep `.python-version` under version control; exclude `.venv/`.

For exact replication of a shared result, use the same Git commit and Python patch version. The supplied log used CPython 3.12.7; check `.python-version` to see how precisely it is pinned. Different operating systems can require different platform-specific packages.

## Before sharing — project owner

The final recorded commit deletes `src/pitaya/__init__.py`. Because the earlier setup built Pitaya as a package, this may break a fresh `uv sync`. Restore the package file if packaging is intended, or explicitly configure a non-package project with `[tool.uv]` and `package = false` in `pyproject.toml`, then regenerate and commit the lockfile. Verify the setup in a fresh clone before distributing this guide.

## References

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv project workflow](https://docs.astral.sh/uv/guides/projects/)
- [Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Jupyter and VS Code](https://docs.astral.sh/uv/guides/integration/jupyter/)
- [Project packaging configuration](https://docs.astral.sh/uv/concepts/projects/config/)
