# Releasing gama to PyPI

The PyPI distribution is **`gama-llm`** (the import name and CLI stay `gama`; the bare
`gama` name is taken on PyPI by an AutoML project). Publishing is automated via **PyPI
Trusted Publishing** (OIDC) — no API token is stored in the repo or CI.

## One-time setup (maintainer)
On PyPI → the `gama-llm` project → **Publishing** → add a **Trusted Publisher**:
- Owner / org: `akihidem`
- Repository: `gama`
- Workflow filename: `publish.yml`
- Environment: `pypi`

(For a brand-new project name, PyPI lets you add a "pending" trusted publisher before the
first release, or do one manual upload to create the project.)

## Cut a release
1. Bump `version` in `pyproject.toml` **and** `__version__` in `gama/__init__.py`.
2. Commit, then create a GitHub **Release** (tag e.g. `v0.1.0`).
3. `.github/workflows/publish.yml` builds (`python -m build`) and publishes automatically.

## Manual alternative (if you prefer a token)
```bash
pip install build twine
python -m build
twine upload dist/*        # needs a PyPI token in ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD
```
