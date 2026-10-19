"""Loss weights are training-recipe coefficients applied exactly as documented."""

import json

import pytest
import torch
from conftest import make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mfm.model import LossWeights
from pymc_mfm.training import TrackingConfig, TrainingConfig, train

RECIPE = {
    "struct_weight": 0.7,
    "struct_cy_weight": 1.2,
    "graph_conditional_weight": 0.6,
    "context_decomposition_weight": 0.3,
    "context_graph_weight": 0.2,
}


def _documented_combination(terms, weights):
    """``(task_loss, loss)`` from the weighted sum documented on LossWeights."""
    cdw = weights["context_decomposition_weight"]
    gcw = weights["graph_conditional_weight"]
    task = (
        terms["flow_loss"]
        + cdw * terms["context_flow_loss"]
        + gcw * terms["graph_conditional_flow_loss"]
        + gcw * cdw * terms["graph_conditional_context_flow_loss"]
    )
    return task, task + terms["struct_loss"]


def test_canonical_recipe_trains_with_the_ab_loss_weights():
    assert TrainingConfig().loss_weights == LossWeights()
    assert LossWeights().to_dict() == {
        "struct_weight": 1.0,
        "struct_cy_weight": 1.0,
        "graph_conditional_weight": 1.0,
        "context_decomposition_weight": 0.5,
        "context_graph_weight": 0.1,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("struct_weight", 0.0),
        ("struct_cy_weight", float("nan")),
        ("graph_conditional_weight", True),
        ("context_decomposition_weight", "0.5"),
        ("context_graph_weight", -0.1),
    ],
)
def test_every_objective_stays_active_with_a_finite_positive_weight(name, value):
    with pytest.raises(ValueError):
        LossWeights(**{name: value})


def test_serialized_weights_round_trip_and_never_fill_defaults():
    weights = LossWeights(**RECIPE)
    assert LossWeights.from_dict(json.loads(json.dumps(weights.to_dict()))) == weights
    incomplete = weights.to_dict()
    del incomplete["context_graph_weight"]
    with pytest.raises(ValueError):
        LossWeights.from_dict(incomplete)
    with pytest.raises(ValueError):
        LossWeights.from_dict({**RECIPE, "flow_weight": 1.0})


def test_task_loss_is_the_documented_weighted_combination(model, batch):
    model = model.double()
    batch = {key: value.double() for key, value in batch.items()}
    torch.manual_seed(41)
    terms = model.loss_terms(batch, weights=LossWeights(**RECIPE))
    values = {name: value.item() for name, value in terms.items()}
    task, total = _documented_combination(values, RECIPE)
    assert values["task_loss"] == pytest.approx(task, rel=1e-12)
    assert values["loss"] == pytest.approx(total, rel=1e-12)


def test_training_records_satisfy_the_combination_for_the_configured_weights(tmp_path):
    train(
        [make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=3)],
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(steps=2, eval_every=2, milestone_steps=(), **RECIPE),
        device="cpu",
    )
    records = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    for record in records:
        task, total = _documented_combination(record, RECIPE)
        assert record["task_loss"] == pytest.approx(task, rel=1e-6)
        assert record["loss"] == pytest.approx(total, rel=1e-6)
        assert record["struct_loss_raw"] == pytest.approx(record["struct_loss"] / 0.7)
        assert record["objective_weights"] == pytest.approx(
            {**RECIPE, "graph_conditional_context_weight": 0.6 * 0.3}
        )
    # The executed objective records the internal coefficients it applied.
    trace = json.loads((tmp_path / "run" / "objective_trace.json").read_text())
    assert trace["structure"]["weight"] == 0.7
    cy = trace["structure"]["bce"]["families"]["cy"]
    assert cy["raw_family_weight"] == pytest.approx(1.2)
    assert trace["context_graph_weight"] == 0.2
