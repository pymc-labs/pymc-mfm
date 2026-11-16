# Agent and contributor instructions

PyMC MFM is a library that will be published. Everything Git tracks here, including
history, pull request heads, CI logs and built archives, must be safe to publish.

## Set up

```bash
uv sync --locked --extra local --extra client
uv run --no-sync pytest -q -n 2 -m "not slow"
uv run ruff check . && uv run ruff format --check .
python3 scripts/check_public_boundary.py
```

Install all extras before running the slow integration tests, and run them serially:

```bash
uv sync --locked --all-extras
uv run --no-sync pytest -q -m slow
```

After that sync, `uv run --no-sync pytest -q` runs the full suite serially. Plain
`pytest` applies no marker filter, so explicit file and node selections retain slow cases.

The library installs, tests and builds from this repository alone. Nothing here may
require access to another repository, a private model, a cloud account or a token.

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
1 MiB, large embedded base64 payloads, credential patterns, and `.private` anywhere
in a path. Notebooks (`*.ipynb`) may hold base64 only as code-cell image outputs
(PNG, JPEG or GIF, at most 2 MiB each, basic structure checks only, not a full decoder); all other
notebook strings are scanned as text. A parser cannot rule out hidden data in a valid
image, so each notebook with image outputs fails until an allowlist entry with `sha256`
pins its reviewed bytes. A rerun changes the hash: update `sha256` in the same pull
request. Range checks also read commit messages and fail on a root commit that the
base branch does not contain (`--allowed-root SHA` accepts a listed root). CI runs it on every pull request
over the whole tree, over every commit in the pull request and over the built sdist
and wheels. Exceptions are read from the committed allowlist, not your working copy.

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
