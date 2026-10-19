# Release preparation

PyMC MFM remains at **`0.0.1.dev0`**. These notes describe the standalone
library's development capabilities and distribution prerequisites, not an
approved or published release.

## Changes

- Observation-only inference checks numeric domains, padding, masks and tensor
  structure before precision narrowing. Training targets never enter inference;
  all padded control feature blocks are zero.
- Model bundles use strict configuration and safetensors loading. Checkpoint,
  training-run and data identities remain attached to exports and results;
  caller annotations cannot replace a loaded checkpoint's identity.
- Saved decomposition evidence rejects contradictory graph mode, marginal
  window and sample-count metadata. Reports can be rendered again from saved
  evidence without inference, training or reference fitting.
- Public entrypoints cover corpus generation, training, checkpoint/export
  loading, conditional inference and diagnostics. The small workflow includes
  actual local MLflow metric and artifact readback.
- Tracking readback defaults to the producing run's saved backend, not an
  unrelated ambient URI. Custom diagnostic-observer exceptions fail the run
  after local evidence is saved; optional report-build failures remain explicit.
- The optional generator dependency follows
  `git+https://github.com/pymc-labs/pymc-generator.git@main`; `uv.lock` records the
  resolved commit for locked uv installations.

## Public API and evidence limits

The offline observation example requires `--model` to name a known local MFM
bundle. No pretrained weights are bundled or selected automatically. Base local
inference does not require generation, plotting, MLflow or Hub clients.
`[generation]` supplies the generator, `[diagnostics]` adds numerical diagnostics
and plotting, `[training]` adds diagnostics and MLflow, and `[hub]` enables remote
artifact transport. Installing dependencies can require network access; local
inference and saved-evidence replay do not. Remote loading resolves the selected
revision, or the default branch when omitted, to one recorded immutable commit.
Use a full commit for reproducibility; offline Hub loading needs a populated cache.

Context and query inference calls are independent marginal evaluations, not
paired full-horizon draws. `model_average` samples router graphs, `map` uses
the empirical complete-graph mode, and `fixed` conditions on a supplied graph
hypothesis. A supplied graph is not posterior graph evidence or a lift-test
likelihood; auxiliary structure probabilities do not route these calls.

Truth-based diagnostics require aligned truth, training curves require recorded
histories, and reference diagnostics require usable, qualified reference
evidence. Independent-world calibration requires independent-world draws and
explicit prior/data identities. Missing inputs are unavailable or not applicable,
not zero scores or successful calibration checks. One-world rank screening and
finite graph-mixture variance are not calibration guarantees.

## Training resources and small workflow

The public canonical configuration specifies 62,500 AdamW updates of 64 worlds,
seed 31, a 128-dimensional encoder with four layers and eight heads, and flow
width 256. Profile A specifies 200,000 synthetic worlds with five direct-effect
channels; profile B specifies 400,000 worlds with four to ten. These are
training-prior settings, not causal activity inferred from observation columns.
They specify the workload, not a GPU-memory requirement or a wall-clock
performance guarantee.

```bash
uv sync --locked --extra generation --extra training
uv run --no-sync python examples/workflow.py --out runs/workflow
```

This example generates twelve synthetic worlds with 16 rows and an eight-row
context, then trains the full architecture for two optimizer updates. It
exercises checkpoint and safetensors loading, observation-only inference,
observation-only and synthetic-truth reports, and inference-free report replay.
It is an integration exercise, **not usable pretrained weights** or evidence of
full-budget training quality.

The workflow explicitly selects a local SQLite MLflow backend. It reads actual
backend metric histories, downloads checkpoint/report artifacts and checks their
hashes against the producing run's records. The saved metric-name table resolves
backend keys. Outputs require a new directory and are not published. See the
[README](README.md) for the inference, diagnostics and installed-distribution
entrypoints.

Fresh-install verification passed the full workflow from the built sdist,
offline inference from a base-only wheel environment, and saved-result
render/replay from a diagnostics-only wheel environment. Each ran with Python
`-I`, verified noneditable installed imports, and checked its dependency set.

One base-wheel CPU smoke on an Apple M3 Max (macOS arm64, Python 3.13.9,
PyTorch 2.14.0, two CPU threads) measured 3.83 seconds to load the full model,
0.22 seconds for four context draws and 0.17 seconds for four query draws.
Each window had eight rows and 26 components; peak **whole-process** RSS was
379.75 MiB. These are single-run observations using the two-update example,
not a benchmark distribution, hardware minimum or trained-model quality claim.

None of these workflows establishes business-domain calibration, causal
identification, out-of-distribution validity, lift-test evidence or safe
unattended budget optimization.

## Before distribution

- Maintainers must choose and approve licenses for the MFM code and any model
  weights before distributing them; those licenses are not yet declared.
  Approve any accompanying data or diagnostic evidence for redistribution too.
- Any distributed pretrained model needs an immutable bundle reference,
  checksums, a model card and a clearly stated evaluation scope. The two-update
  example is not a pretrained model release.
- Hub publication needs an authorized destination and an approved artifact
  license. A real upload, pinned download and offline reload must be verified
  for the intended artifacts before advertising that distribution path.
- Version/tag selection, repository visibility, package-registry publication
  and model publication each require explicit maintainer authorization.
- The generator intentionally follows a Git branch. Source and directly
  installed wheel/sdist distributions support that dependency; PyPI
  [does not accept direct-URL dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies).
  Any future PyPI release needs a separate distribution-policy decision.
