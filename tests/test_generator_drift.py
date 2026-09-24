"""Generator drift: small shard-plan probes must reproduce their recorded arrays.

The ``pymc-generator`` dependency tracks ``main``. Each probe is one shard of a
plan in ``SHARD_PLANS`` generated with two cells and the plan's seed, draws per
cell and overrides, so it covers the start of that shard. A change on ``main``
that alters what the A, B or evaluation priors generate changes a recorded
array hash here. CI installs the locked commit; the scheduled generator-drift
workflow resolves ``main`` itself. The profile-A probe is generated through the
CLI. After an intended generator change, re-record with

    python -m tests.test_generator_drift --write [--force]
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import pymc_mfm.data.__main__ as cli
from pymc_mfm.data.recipes import SHARD_PLANS, load_corpora, prior_sidecar_path, provenance

RECORD = Path(__file__).resolve().parent / "data" / "generator_probes.json"
PROBE_CELLS = 2
PROBES = (("A", 0), ("B", 0), ("evaluation", 0), ("evaluation", 9))
CLI_PROBE = ("A", 0)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)


def generator_commit() -> str | None:
    """The git commit the installed ``pymc-generator`` was built from, if recorded."""
    try:
        direct = importlib.metadata.distribution("pymc-generator").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")


def fingerprints(corpus: dict) -> dict[str, dict]:
    """Every array, ``identifiability.*`` included, keyed by name.

    The digest covers the name, ``dtype.str``, shape and C-order bytes; the
    float64 sum only helps judge the size of a change.
    """
    arrays = {
        key: value for key, value in corpus.items() if key not in ("diagnostics", "identifiability")
    }
    for key, value in corpus.get("identifiability", {}).items():
        arrays[f"identifiability.{key}"] = value
    records = {}
    for name, value in sorted(arrays.items()):
        array = np.asarray(value)
        digest = hashlib.sha256(json.dumps([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
        records[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": digest.hexdigest(),
            "sum": float(array.sum(dtype=np.float64)),
        }
    return records


def probe_settings(plan_name: str, shard: int) -> dict:
    plan = SHARD_PLANS[plan_name]
    return {
        "seed": plan.seed(shard),
        "n_cells": PROBE_CELLS,
        "draws_per_cell": plan.draws_per_cell,
        "overrides": json.loads(json.dumps(plan.shard_overrides(shard))),
    }


def write_cli_probe(directory: Path) -> Path:
    """Generate the CLI probe into ``directory`` and return the corpus path.

    ``--shard`` fixes a plan's full cell count, so the probe passes the plan's
    seed, draws and overrides explicitly.
    """
    plan_name, shard = CLI_PROBE
    plan = SHARD_PLANS[plan_name]
    arguments = [
        "generate",
        "--profile",
        plan.profile.name,
        "--n-cells",
        str(PROBE_CELLS),
        "--draws-per-cell",
        str(plan.draws_per_cell),
        "--seed",
        str(plan.seed(shard)),
        "--out",
        str(directory),
    ]
    for key, value in plan.shard_overrides(shard).items():
        arguments += ["--override", f"{key}={json.dumps(value)}"]
    assert cli.main(arguments) == 0
    return directory / f"shard_{plan.seed(shard)}.npz"


def probe_corpus(plan_name: str, shard: int, cli_probe: Path) -> dict:
    from pymc_generator import DataGenerator, load_corpus

    if (plan_name, shard) == CLI_PROBE:
        return load_corpus(cli_probe)
    return DataGenerator(SHARD_PLANS[plan_name].prior(shard, n_cells=PROBE_CELLS)).generate()


@pytest.fixture(scope="module")
def cli_probe(tmp_path_factory):
    return write_cli_probe(tmp_path_factory.mktemp("cli_probe"))


def test_cli_writes_a_loadable_probe_with_a_bound_sidecar(cli_probe):
    assert sorted(cli_probe.parent.iterdir()) == [cli_probe, prior_sidecar_path(cli_probe)]
    (loaded,) = load_corpora([cli_probe])
    assert loaded.n_worlds == PROBE_CELLS * SHARD_PLANS["A"].draws_per_cell
    record = provenance(loaded)
    assert record["data_origin"] == "synthetic"
    prior = record["prior_sidecar"]["prior"]
    assert (prior["seed"], prior["n_cells"], prior["draws_per_cell"]) == (530200, 2, 10)


@pytest.mark.parametrize(("plan_name", "shard"), PROBES, ids=[f"{p}-{s}" for p, s in PROBES])
def test_probe_arrays_match_the_recorded_generator_output(plan_name, shard, request):
    record = json.loads(RECORD.read_text())
    recorded = record["probes"][f"{plan_name}/{shard}"]
    assert recorded["settings"] == probe_settings(plan_name, shard), (
        "the shard plan changed since the probes were recorded; re-record them"
    )
    cli_probe = request.getfixturevalue("cli_probe") if (plan_name, shard) == CLI_PROBE else None
    actual = fingerprints(probe_corpus(plan_name, shard, cli_probe))
    expected = recorded["arrays"]
    assert sorted(set(expected) - set(actual)) == [], "recorded arrays are missing"
    assert sorted(set(actual) - set(expected)) == [], "the generator writes unrecorded arrays"
    changed = [
        f"{name}: recorded {expected[name]}, generated {actual[name]}"
        for name in expected
        if any(actual[name][key] != expected[name][key] for key in ("dtype", "shape", "sha256"))
    ]
    assert not changed, (
        f"generator output drifted (recorded with pymc-generator commit "
        f"{record.get('pymc_generator_commit')}, installed {generator_commit()}):\n"
        + "\n".join(changed)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.test_generator_drift",
        description="Record the shard-plan probe arrays of the installed pymc-generator.",
    )
    parser.add_argument("--write", action="store_true", required=True, help=f"write {RECORD}")
    parser.add_argument("--force", action="store_true", help="replace an existing record")
    args = parser.parse_args(argv)
    if RECORD.exists() and not args.force:
        parser.error(f"{RECORD} exists; pass --force to replace the recorded probes")
    import pymc_generator

    with tempfile.TemporaryDirectory() as directory:
        cli_probe = write_cli_probe(Path(directory))
        probes = {
            f"{plan_name}/{shard}": {
                "settings": probe_settings(plan_name, shard),
                "arrays": fingerprints(probe_corpus(plan_name, shard, cli_probe)),
            }
            for plan_name, shard in PROBES
        }
    payload = {
        "format_version": 1,
        "pymc_generator_version": pymc_generator.__version__,
        "pymc_generator_commit": generator_commit(),
        "probes": probes,
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(RECORD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
