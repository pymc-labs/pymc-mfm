# Agent and contributor instructions

PyMC MFM is a library that will be published. Everything Git tracks here, including
history, pull request heads, CI logs and built archives, must be safe to publish.

## Set up

Use Python 3.13 or newer.

```bash
uv sync --locked
uv run --no-sync pytest -q -n 2 -m "not slow"
uv run ruff check . && uv run ruff format --check .
python3 scripts/check_public_boundary.py
```

For the full contributor suite, install the corpus-generation and notebook
dependency groups before running the slow tests serially:

```bash
uv sync --locked --group generation --group examples
uv run --no-sync pytest -q -m slow
```

After that sync, `uv run --no-sync pytest -q` runs the full suite serially. Plain
`pytest` applies no marker filter, so explicit file and node selections retain slow cases.

The package installs, tests and builds without `pymc-generator` or
access to a private model, a cloud account or a token. The repository's
`generation` group fetches `pymc-generator` from Git; neither it nor the notebook
dependencies are part of the standard installation.

## What belongs here

| Belongs here | Never commit here |
|---|---|
| Model, inference, training, evaluation and data-generation code | Trained weights or checkpoints from real runs (`.safetensors`, `.pt`, `.ckpt` and similar) |
| Tests, with small synthetic fixtures built or stored in `tests/` | Credentials of any kind, including tokens in notebooks, logs or fixtures |
| Examples and notebooks on synthetic data; weights loaded from the Hub at run time | Customer, user or learner data, or results derived from it |
| Benchmarks that take workspace, environment and bucket names as required arguments | Internal deployment identifiers: cloud project ids, Modal workspace, environment or app names, service hostnames, bucket names |
| Public docs and release notes | Production, app or service configuration; private experiment results and findings |

When unsure, keep it out and ask a maintainer. Generate fixtures at test time when
you can. Large or binary files need an allowlist entry with a reason (below).

## The boundary check

`scripts/check_public_boundary.py` checks what Git tracks, not the working tree. It
flags tracked symlinks, gitlinks (nested repositories, submodule pointers) and
`.gitmodules` files, weight-like files, binary files it cannot read, files over
1 MiB, large embedded base64 payloads, credential patterns, service hostnames (any
`pymc-labs.com` host except `www`, and hosts on `fly.dev`, `run.app`, `workers.dev` and
`modal.run`), and `.private` anywhere
in a path. Notebooks (`*.ipynb`) may hold base64 only as code-cell image outputs
(PNG, JPEG or GIF, at most 2 MiB each, basic structure checks only, not a full decoder); all other
notebook strings are scanned as text. A parser cannot rule out hidden data in a valid
image, so each notebook with image outputs fails until an allowlist entry with `sha256`
pins its reviewed bytes. A rerun changes the hash: update `sha256` in the same pull
request. Range checks also read commit messages and fail on a root commit that the
base branch does not contain (`--allowed-root SHA` accepts a listed root). After
admin authorization, CI checks the whole tree, each PR commit and the built
distributions. Exceptions are read from the committed allowlist, not your working copy.

```bash
python3 scripts/check_public_boundary.py             # tracked files at HEAD
python3 scripts/check_public_boundary.py --staged    # what the next commit holds
python3 scripts/check_public_boundary.py --range origin/main..HEAD
uv build && python3 scripts/check_public_boundary.py --archive dist/*
```

A legitimate exception goes in `.github/public-boundary-allow.toml` with the path,
the rules it may break and a reason. Pin binary files with `sha256` so a changed
file needs a new review. An entry with `pending` passes routine CI but fails
`--release`, the mode used before publishing. Do not weaken a rule to make a change
pass.

## GitHub Actions policy

Configure these repository-level [Actions execution policies](https://docs.github.com/en/actions/how-tos/administer/control-workflow-execution)
together:

- Allow only the repository **Admin** role to start workflows. Exclude only
  `.github/workflows/approved-ci.yml`, `tests.yml`, `docs-build.yml`,
  `public-boundary.yml` and `release-build.yml` (all under `.github/workflows/`).
- For those exact five excluded paths, allow only `repository_dispatch`,
  `pull_request_target` and `workflow_call`. Do not allow events that load workflow
  YAML from a PR, an arbitrary branch or a tag on those paths.

Keep `ci-ok`, `build` and `boundary` required from GitHub Actions, strict
up-to-date checks enabled, and the MFM reviewer requirement in place. The first
rollout requires the trusted workflow to be on `main`; it cannot issue its own
pre-merge statuses. An admin must independently verify that rollout revision and
coordinate the policy activation with the protected merge. Afterward, verify a
non-admin dispatch runs no checks and that a valid admin dispatch satisfies the
three required contexts on the reviewed head.

## Maintainers with the companion repo

Maintainers may have a private companion checkout linked at `.private` in this
checkout. Git ignores that link. If it exists, also read the `AGENTS.md` inside it.
Write private material only through that link, never into tracked paths here. Do
not name the companion repo, its paths or its contents in tracked files, commit
messages or pull request text.

To package or share the public tree, use `git archive` or `uv build`. Do not archive
or copy the checkout root with tools that follow links (`tar -h`, `zip`, `7z`,
`rsync -L`, `cp -rL`, `grep -R`): they follow the private link and copy its contents.

The link is a convenience. GitHub permissions are the access control. Hooks are
local and can be bypassed, so CI repeats the generic checks.
