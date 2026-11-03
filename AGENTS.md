# Agent and contributor instructions

PyMC MFM is a library that will be published. Everything Git tracks here, including
history, pull request heads, CI logs and built archives, must be safe to publish.

## Set up

```bash
uv sync --locked
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
python3 scripts/check_public_boundary.py
```

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
flags tracked symlinks, weight-like files, files over 1 MiB, large embedded base64
payloads, credential patterns and paths into the `.private` link. CI runs it on
every pull request over the whole tree, over every commit in the pull request and
over the built sdist and wheels.

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

The link is a convenience. GitHub permissions are the access control. Hooks are
local and can be bypassed, so CI repeats the generic checks.
