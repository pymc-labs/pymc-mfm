"""Mini development plans and training presets: geometry, seeds, CLI defaults, bundle rule."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from pymc_mfm.data import __main__ as data_cli
from pymc_mfm.data.recipes import SHARD_PLANS
from pymc_mfm.inference._artifact_schema import _recorded_world
from pymc_mfm.inference.hub import _training_plan
from pymc_mfm.training import __main__ as training_cli
from pymc_mfm.training.config import PRESETS, TrainingConfig

needs_generator = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)
MINI_PLANS = ("mini", "mini_evaluation")


def test_mini_plan_geometry():
    mini, evaluation = SHARD_PLANS["mini"], SHARD_PLANS["mini_evaluation"]
    for plan in (mini, evaluation):
        assert plan.profile.name == "A"
        assert (plan.n_shards, plan.draws_per_cell, plan.development) == (1, 4, True)
        overrides = plan.shard_overrides(0)
        assert overrides["n_treatments"] == 8
        assert overrides["n_covariates"] == 4
        assert overrides["n_treatments_active_range"] == (6, 8)
        assert overrides["n_covariates_active_range"] == (3, 4)
    assert mini.worlds == 20_000
    assert evaluation.worlds == 128
    assert mini.seed(0) == 550000


def test_profile_and_evaluation_plans_are_not_development_plans():
    for name in ("A", "B", "evaluation"):
        assert not SHARD_PLANS[name].development


def test_plan_seeds_are_unique_across_plans():
    seeds = [plan.seed(shard) for plan in SHARD_PLANS.values() for shard in range(plan.n_shards)]
    assert len(seeds) == len(set(seeds))


@needs_generator
@pytest.mark.parametrize("name", MINI_PLANS)
def test_mini_priors_pass_generator_validation(name):
    prior = SHARD_PLANS[name].prior(0, n_cells=2)
    assert (prior.n_time_steps, prior.n_treatments, prior.n_covariates) == (104, 8, 4)
    assert prior.n_latent == 1


@pytest.mark.parametrize("name", MINI_PLANS)
def test_recorded_world_refuses_development_plans(name):
    plan = SHARD_PLANS[name]
    world = {"plan": name, "shard": 0, "shard_seed": plan.seed(0), "row": 0}
    with pytest.raises(ValueError, match="development plan"):
        _recorded_world(world)


def test_recorded_world_still_accepts_the_evaluation_plan():
    plan = SHARD_PLANS["evaluation"]
    _recorded_world({"plan": "evaluation", "shard": 1, "shard_seed": plan.seed(1), "row": 0})


@pytest.mark.parametrize("name", MINI_PLANS)
def test_cards_do_not_accept_development_plans_as_training_plans(name):
    with pytest.raises(ValueError, match="profile training plan"):
        _training_plan(name)


def test_data_cli_selects_mini_plans_and_allows_the_cell_probe(monkeypatch, tmp_path):
    seen = []

    def fake_generate(prior, path):
        seen.append((prior, path))
        return {"is_val": np.zeros(1, dtype="uint8")}

    class _Prior:
        def __init__(self, n_cells):
            self.n_cells = n_cells

    monkeypatch.setattr(data_cli, "generate_corpus", fake_generate)
    monkeypatch.setattr(
        type(SHARD_PLANS["mini"]), "prior", lambda self, shard, n_cells=None: _Prior(n_cells)
    )
    for flag, seed in (("--mini-shard", 550000), ("--mini-evaluation-shard", 551000)):
        assert data_cli.main(["generate", flag, "0", "--n-cells", "8", "--out", str(tmp_path)]) == 0
        assert seen[-1][0].n_cells == 8
        assert seen[-1][1] == tmp_path / f"shard_{seed}.npz"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--mini-shard", "1"],
        ["--mini-evaluation-shard", "-1"],
        ["--mini-shard", "0", "--mini-evaluation-shard", "0"],
        ["--mini-shard", "0", "--evaluation-shard", "0"],
        ["--mini-shard", "0", "--profile", "A"],
        ["--mini-shard", "0", "--seed", "550000"],
        ["--mini-shard", "0", "--draws-per-cell", "4"],
    ],
)
def test_data_cli_rejects_invalid_mini_arguments(arguments, monkeypatch, tmp_path):
    def refuse(prior, path):
        raise AssertionError("generated for invalid arguments")

    monkeypatch.setattr(data_cli, "generate_corpus", refuse)
    with pytest.raises(SystemExit) as raised:
        data_cli.main(["generate", *arguments, "--out", str(tmp_path)])
    assert raised.value.code == 2


def test_canonical_preset_is_the_training_config_default():
    assert PRESETS["canonical"] == TrainingConfig()


def test_mini_and_smoke_preset_values():
    mini, smoke = PRESETS["mini"], PRESETS["smoke"]
    assert (mini.steps, mini.warmup_steps, mini.eval_every, mini.batch_size) == (
        8000,
        400,
        1000,
        64,
    )
    assert mini.milestone_steps == ()
    assert smoke.warmup_steps > 0
    assert smoke.warmup_steps < smoke.steps
    assert smoke.eval_every <= smoke.steps
    assert smoke.milestone_steps == ()


@pytest.fixture
def trained_config(monkeypatch, tmp_path):
    """Run the training CLI with corpus loading and ``train`` replaced; return a runner.

    The runner returns the ``TrainingConfig`` that the CLI passed to ``train``.
    """
    import pymc_mfm.data.recipes as recipes

    passed = {}

    class _Result:
        run_dir = tmp_path
        status = "finished"
        final_step = 0
        run = {"training_config": {"budget": None}, "best": None, "tracking": None}

    def fake_train(corpora, *, training, **kwargs):
        passed["training"] = training
        return _Result()

    monkeypatch.setattr(recipes, "load_corpora", lambda paths: [SimpleNamespace(corpus={})])
    monkeypatch.setattr(recipes, "provenance", lambda item: {})
    monkeypatch.setattr(training_cli, "train", fake_train)

    def run(*flags):
        argv = ["--corpus", "c.npz", "--out-dir", str(tmp_path / "run"), *flags]
        assert training_cli.main(argv) == 0
        return passed["training"]

    return run


def test_cli_without_a_preset_keeps_the_canonical_defaults(trained_config):
    assert trained_config() == TrainingConfig()


def test_cli_preset_supplies_defaults_and_explicit_flags_override(trained_config):
    mini = trained_config("--preset", "mini")
    assert (mini.steps, mini.warmup_steps, mini.eval_every) == (8000, 400, 1000)
    overridden = trained_config("--preset", "mini", "--steps", "100", "--eval-every", "50")
    assert (overridden.steps, overridden.warmup_steps, overridden.eval_every) == (100, 400, 50)


def test_cli_rejects_an_unknown_preset(trained_config):
    with pytest.raises(SystemExit):
        trained_config("--preset", "huge")
