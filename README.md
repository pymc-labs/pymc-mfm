# PyMC MFM

PyMC MFM is being developed as a standalone library for training and using
**graph-conditional Marketing Foundation Models**.

**Current status:** repository structure and development environments only.
The model, training, inference, and diagnostics APIs will be implemented in the
[following issues](https://github.com/pymc-labs/pymc-mfm/issues/6).

## Structure

```text
src/pymc_mfm/
  model/         Canonical architecture and its neural-network components
  data/          Observation preprocessing, corpus generation and loading
  training/      Training objectives, optimization and checkpoint writing
  inference/     Checkpoint loading and observation-based inference APIs
  diagnostics/   Numerical diagnostics, calibration and evaluation metrics
  plotting/      Figures and reports, separate from numerical diagnostics
tests/           Focused behavioral and equivalence tests
examples/        Small examples using the public API
pyproject.toml   Package metadata, dependencies and development-tool settings
uv.lock          Locked dependencies for the uv workflow
environment.yml  Conda Python environment
```

The package folders currently contain only package markers; `tests/` and
`examples/` are intentionally empty. Follow-up work will fill these boundaries,
without runtime imports back into PFN. Scientific and optional dependencies will
be added when their implementations are introduced; the initial package has no
runtime dependencies.

Model weights, corpora, and generated experiment outputs do **not** belong in the
library. They remain external, on Hugging Face or
[Google Drive](https://drive.google.com/drive/folders/1G3jcruRo3PVq_mE_6Lvoiobfz2smc8Mg).

## Development environments

Python **3.13 or newer** is the support baseline. Both setup paths below default
to Python 3.13 and keep their environments inside the checkout. Run commands from
the repository root and choose one environment workflow.

### uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked
uv run python -c "import pymc_mfm"
uv run ruff check .
uv run ruff format --check .
uv build
```

`uv sync` installs the package in editable mode and includes the `dev` dependency
group in `.venv`. When deliberately changing dependencies, edit `pyproject.toml`,
run `uv lock`, and commit the updated lockfile.

### Conda / Mamba

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --group dev -e .
python -c "import pymc_mfm"
ruff check .
ruff format --check .
```

Mamba can replace Conda in the environment-creation command. Pip 25.1+ supports
`--group dev`, so package and development dependencies remain declared only in
`pyproject.toml`. This workflow resolves those declarations with pip; it does not
consume `uv.lock`. Do not install project dependencies into Conda's `base`.

Pytest is included for future tests. There is no test suite yet; installation,
import, lint, and package-building checks verify this initial repository setup.
