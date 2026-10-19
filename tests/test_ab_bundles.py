"""The converted A/B bundles keep their identities and reproduce recorded outputs.

Each card can name only the training plan whose shards hold the bundle's
recorded evaluation worlds: A's plan for A, B's plan for B.

Bundles come from ``MFM_AB_BUNDLES``, a local collection directory holding
``profiles/A`` and ``profiles/B``, or, once published, from the local Hugging
Face cache at ``PUBLISHED``. The tests skip only while that variable is unset
and no published copy is cached; a set but unusable variable fails.

Each profile's ``recorded_outputs.json`` holds the raw observations of row
``ROW`` of shard ``SHARD`` of MFM's ``evaluation`` shard plan, that world's
generating graph (the supplied hypothesis of mode ``fixed``) and the expected
``decompose`` outputs for every mode and window, each from ``N_SAMPLES`` draws
under its own seed: posterior-mean components ``[T, D]`` and graph draws. It
names the weights it was recorded with by their ``state_sha256``, the digest
``from_pretrained`` records, so it cannot vouch for a cast or edited copy.
Posterior means are compared at ``ATOL`` context-ruler units (natural units
divided by the world's ``sales_scale``). On the recording machine (arm64 CPU,
float32) 1 and 12 threads differ by at most 6.0e-8 ruler units, so ``ATOL``
exceeds 100 times the measured thread-count spread and leaves room for other
CPU kernels. Graph draws must match exactly: every sampled window-mean graph
coordinate lies at least 0.97 from the sign threshold. Because floating-point
reductions depend on the CPU kernels and the thread count, the sidecar's
``producer`` records the platform system and machine and the torch thread count
next to the package versions; it holds no host, user or node name and no path.

Record the sidecars again only for newly converted bundles; this needs the
``generation`` extra::

    python -m tests.test_ab_bundles --write <collection> [--force]
"""

import argparse
import itertools
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch

from pymc_mfm.inference._artifact_schema import (
    normalized_version,
    python_version,
    recorded_outputs_bytes,
    recorded_outputs_metadata,
)
from pymc_mfm.inference.api import decompose
from pymc_mfm.inference.hub import from_pretrained, write_model_cards

PROFILES = ("A", "B")
#: SHA-256 of the original training checkpoint of each profile.
SOURCE_CHECKPOINTS = {
    "A": "89a57f7ca17bb25173ddc0f85f061123217dca1ded3abd43079afd3014dd63e3",
    "B": "4cc082d9c24f26534e0f18c70d86e883b3f5da4957efa83f31359fc76a609792",
}
#: Held-out joint decomposition energy score recorded by each training run.
DECOMPOSITION_ENERGY_SCORES = {"A": 0.008838079095303743, "B": 0.009995010646948025}
#: ``(repo_id, full commit)`` of the published collection; ``None`` until publication.
PUBLISHED: tuple[str, str] | None = None
SIDECAR = "recorded_outputs.json"
PLAN = "evaluation"
SHARD = 0
ROW = 0
N_SAMPLES = 4
MODES = ("model_average", "map", "fixed")
WINDOWS = ("query", "context")
CASE_SEED = 20260923
ATOL = 1e-5


def collection() -> Path:
    """The bundle collection to test; skips only when none is configured or cached."""
    value = os.environ.get("MFM_AB_BUNDLES")
    if value is not None:
        if not value or not Path(value).is_dir():
            pytest.fail(f"MFM_AB_BUNDLES={value!r} is not a bundle collection directory")
        return Path(value)
    if PUBLISHED is None:
        pytest.skip("not published")
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo_id, revision = PUBLISHED
    patterns = [f"profiles/{profile}/*" for profile in PROFILES]
    try:
        root = Path(
            hub.snapshot_download(
                repo_id, revision=revision, allow_patterns=patterns, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        pytest.skip("not cached")
    names = ("config.json", "model.safetensors", "manifest.json", SIDECAR)
    if not all((root / "profiles" / p / name).is_file() for p in PROFILES for name in names):
        pytest.skip("not cached")
    return root


def recorded_cases(model, observations, graph) -> dict[str, dict]:
    """Posterior-mean components and graph draws for every mode and window."""
    cases = {}
    for index, (mode, window) in enumerate(itertools.product(MODES, WINDOWS)):
        seed = CASE_SEED + index
        torch.manual_seed(seed)
        result = decompose(
            model,
            observations,
            n_samples=N_SAMPLES,
            mode=mode,
            window=window,
            graph=graph if mode == "fixed" else None,
        )
        cases[f"{mode}/{window}"] = {
            "seed": seed,
            "sales_scale": result.provenance["sales_scale"],
            "components_mean": result.components.mean(axis=0).tolist(),
            "graphs": result.graphs.astype(int).tolist(),
        }
    return cases


@pytest.fixture(scope="module", params=PROFILES)
def bundle(request):
    root = collection()
    profile = request.param
    model = from_pretrained(root, profile=profile)
    sidecar = json.loads((root / "profiles" / profile / SIDECAR).read_text())
    return profile, model, sidecar


def test_bundle_and_sidecar_identify_the_source_checkpoint_and_weights(bundle):
    profile, model, sidecar = bundle
    assert model.artifact_metadata["profile"] == profile
    assert model.provenance["source_checkpoint_sha256"] == SOURCE_CHECKPOINTS[profile]
    assert model.provenance["checkpoint_sha256"] == SOURCE_CHECKPOINTS[profile]
    assert (sidecar["profile"], sidecar["source_checkpoint_sha256"]) == (
        profile,
        SOURCE_CHECKPOINTS[profile],
    )
    assert sidecar["state_sha256"] == model.provenance["state_sha256"]
    # The sidecar passes the strict schema that publication applies.
    recorded_outputs_metadata(
        sidecar,
        profile=profile,
        provenance=model.artifact_metadata["provenance"],
        state_sha256=model.provenance["state_sha256"],
        model_config=model.config,
    )


def test_training_evaluation_is_the_recorded_joint_holdout_evaluation(bundle):
    profile, model, _ = bundle
    record = model.artifact_metadata["provenance"]["training_evaluation"]
    assert (record["leg"], record["head"], record["eval_seed"]) == ("holdout", "joint", 616161)
    score = record["metrics"]["joint/query/decomposition_energy_score"]
    assert score == DECOMPOSITION_ENERGY_SCORES[profile]


def test_cards_name_only_the_training_plan_that_holds_the_recorded_worlds(bundle, tmp_path):
    profile, _, _ = bundle
    source = (collection() / "profiles" / profile).resolve()
    copy = tmp_path / "profiles" / profile
    copy.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "manifest.json"):
        (copy / name).symlink_to(source / name)
    other = {"A": "B", "B": "A"}[profile]
    with pytest.raises(ValueError):
        write_model_cards(copy, license="not selected", corpus_plan=other)
    assert write_model_cards(copy, license="not selected", corpus_plan=profile) == [
        copy / "README.md"
    ]


def test_decompose_reproduces_the_recorded_outputs(bundle):
    _, model, sidecar = bundle
    observations = {key: np.asarray(value) for key, value in sidecar["observations"].items()}
    assert sidecar["n_samples"] == N_SAMPLES
    actual = recorded_cases(model, observations, np.asarray(sidecar["graph"]))
    assert actual.keys() == sidecar["cases"].keys()
    failures = []
    for name, expected in sidecar["cases"].items():
        scale = expected["sales_scale"]
        try:
            assert actual[name]["seed"] == expected["seed"]
            assert actual[name]["sales_scale"] == scale
            np.testing.assert_allclose(
                np.asarray(actual[name]["components_mean"]) / scale,
                np.asarray(expected["components_mean"]) / scale,
                rtol=0,
                atol=ATOL,
                err_msg=name,
            )
            np.testing.assert_array_equal(actual[name]["graphs"], expected["graphs"], err_msg=name)
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    assert not failures, "\n".join(failures)


def evaluation_world() -> tuple[dict[str, object], dict[str, list], list]:
    """Identity, raw observations and generating graph of the recorded world, as JSON."""
    import pymc_generator

    from pymc_mfm.data.recipes import SHARD_PLANS

    plan = SHARD_PLANS[PLAN]
    corpus = pymc_generator.DataGenerator(plan.prior(SHARD)).generate()
    context = int(corpus["support_mask"][ROW].sum())
    observations = {
        "spend": corpus["treatment_raw"][ROW],
        "controls": corpus["covariates"][ROW],
        "context_outcome": corpus["outcome_raw"][ROW, :context],
        "channel_mask": corpus["treatment_active_mask"][ROW],
        "control_mask": corpus["covariate_active_mask"][ROW],
        "latent_mask": corpus["latent_active_mask"][ROW],
    }
    world = {"plan": PLAN, "shard": SHARD, "shard_seed": plan.seed(SHARD), "row": ROW}
    return (
        world,
        {key: value.tolist() for key, value in observations.items()},
        corpus["g"][ROW].tolist(),
    )


def producer() -> dict[str, object]:
    """The recording environment, versions as the release-version rule requires them."""
    return {
        "python": python_version(),
        "pymc-mfm": normalized_version(version("pymc-mfm")),
        "pymc-generator": normalized_version(version("pymc-generator")),
        "torch": normalized_version(torch.__version__),
        "numpy": normalized_version(np.__version__),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.test_ab_bundles",
        description="Record the recorded-output sidecar of every A/B bundle in a collection.",
    )
    parser.add_argument("--write", type=Path, required=True, metavar="COLLECTION")
    parser.add_argument("--force", action="store_true", help="replace existing sidecars")
    args = parser.parse_args(argv)
    targets = {profile: args.write / "profiles" / profile / SIDECAR for profile in PROFILES}
    if not args.force and any(target.exists() for target in targets.values()):
        parser.error("a sidecar exists; pass --force to replace the recorded outputs")
    models = {profile: from_pretrained(args.write, profile=profile) for profile in PROFILES}
    for profile, model in models.items():
        if model.provenance.get("source_checkpoint_sha256") != SOURCE_CHECKPOINTS[profile]:
            parser.error(f"profile {profile} is not the converted source checkpoint")
    world, observations, graph = evaluation_world()
    arrays = {key: np.asarray(value) for key, value in observations.items()}
    for profile, model in models.items():
        payload = {
            "format_version": 1,
            "profile": profile,
            "source_checkpoint_sha256": SOURCE_CHECKPOINTS[profile],
            "state_sha256": model.provenance["state_sha256"],
            "world": world,
            "observations": observations,
            "graph": graph,
            "n_samples": N_SAMPLES,
            "cases": recorded_cases(model, arrays, np.asarray(graph)),
            "producer": producer(),
        }
        targets[profile].write_bytes(recorded_outputs_bytes(payload))
        print(targets[profile])
    return 0


if __name__ == "__main__":
    sys.exit(main())
