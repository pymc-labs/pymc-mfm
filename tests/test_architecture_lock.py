"""Recorded outputs of one small fixed-seed model lock the fixed architecture.

``data/architecture_lock.json`` records a float64 model with two encoder
layers, so layer order, residual chaining and per-layer weights all shape the
model outputs. Its parameters are overwritten in sorted-name order with
``0.1 * RandomState(20260923)`` normal draws. Losses (default and non-default
weights), the default loss's gradient for every parameter, structure
probabilities, joint router draws and conditional draws in every mode and
window, plus one draw count that crosses the sampling chunk, are compared at
``rtol=1e-7, atol=1e-9``: float64 kernels differ by at most a few ULPs between
CPUs and thread counts (1 and 12 threads agree exactly on the recording
machine), far below any architectural or numerical change. The gradients catch
changes that leave every forward value alone, such as a detached input, which
would still train a different network. Graphs are compared exactly; every
recorded window-mean graph coordinate lies at least 1e-6 from the sign
threshold. The raw-observation encoding runs the production float32 casts,
whose transcendental kernels differ by ULPs between CPU dispatch targets, so it
is compared at ``rtol=atol=1e-5``.

Record again only after an intentional architecture change::

    python -m tests.test_architecture_lock --write [--force]
"""

import argparse
import itertools
import json
import platform
import sys
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch

from pymc_mfm.inference.api import observation_batch
from pymc_mfm.model import MFM, LossWeights, ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout, binary_graphs_from_flow

GOLDEN = Path(__file__).resolve().parent / "data" / "architecture_lock.json"
MODEL = {
    "d_model": 8,
    "n_layers": 2,
    "n_heads": 2,
    "d_ff": 16,
    "flow_hidden": 16,
    "n_controls": 2,
    "n_demand": 2,
}
LOSS_WEIGHTS = {
    "struct_weight": 0.7,
    "struct_cy_weight": 1.2,
    "graph_conditional_weight": 0.6,
    "context_decomposition_weight": 0.3,
    "context_graph_weight": 0.2,
}
SEEDS = {
    "parameters": 20260923,
    "batch": 20260924,
    "observations": 20260925,
    "loss_terms": 101,
    "predict": 201,
    "sample_context": 202,
    "sample_disentangled": 301,
    "chunk_crossing": 317,
}
N_SAMPLES = 3
CHUNK_CROSSING_N_SAMPLES = 17
MODES = ("model_average", "map", "fixed")
WINDOWS = ("query", "context")
MIN_GRAPH_MARGIN = 1e-6
RECIPE = {
    "model": MODEL,
    "loss_weights": LOSS_WEIGHTS,
    "seeds": SEEDS,
    "n_samples": N_SAMPLES,
    "chunk_crossing_n_samples": CHUNK_CROSSING_N_SAMPLES,
    "dtype": "float64",
}
ENCODING_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
MODEL_TOLERANCE = {"rtol": 1e-7, "atol": 1e-9}


def lock_model() -> MFM:
    """The tiny architecture with float64 parameters from one seeded stream."""
    model = MFM(ModelConfig(**MODEL)).double().eval()
    rng = np.random.RandomState(SEEDS["parameters"])
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in sorted(parameters):
            values = 0.1 * rng.standard_normal(tuple(parameters[name].shape))
            parameters[name].copy_(torch.from_numpy(values))
    return model


def lock_batch() -> dict[str, torch.Tensor]:
    """Encoded float64 training batch with padding in every node family."""
    rng = np.random.RandomState(SEEDS["batch"])
    b, k, m, j, n_context, n_query = 2, 3, 2, 2, 5, 3
    width = 1 + k + m + 4
    shapes = {
        "ctx_spend_z": (b, n_context, k),
        "qry_spend_z": (b, n_query, k),
        "ctx_controls_z": (b, n_context, m),
        "qry_controls_z": (b, n_query, m),
        "ctx_revenue_z": (b, n_context, 1),
        "spend_mean": (b, k),
        "response_moments": (b, k, 4),
        "adstock_moments": (b, k, 9),
        "control_moments": (b, m, 3),
        "control_location_scale": (b, m, 2),
        "control_effect_moments": (b, m, 6),
        "targets_z": (b, n_query, width),
        "context_targets_z": (b, n_context, width),
    }
    arrays = {name: rng.standard_normal(shape) for name, shape in shapes.items()}
    arrays["spend_std"] = 1.0 + np.abs(rng.standard_normal((b, k)))
    week = np.arange(n_context + n_query, dtype=np.float64) / (n_context + n_query)
    arrays["ctx_week"] = np.broadcast_to(week[None, :n_context, None], (b, n_context, 1))
    arrays["qry_week"] = np.broadcast_to(week[None, n_context:, None], (b, n_query, 1))
    arrays["active_c_mask"] = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    arrays["active_m_mask"] = np.array([[1.0, 1.0], [1.0, 0.0]])
    arrays["active_j_mask"] = np.array([[1.0, 0.0], [1.0, 1.0]])
    layout = EdgeSlotLayout(k, m, j)
    allowed = layout.active_slot_mask(
        arrays["active_c_mask"], arrays["active_m_mask"], arrays["active_j_mask"]
    )
    graph = ((rng.random_sample(allowed.shape) < 0.5) & (allowed > 0)).astype(np.float64)
    arrays.update({f"g_{family}": block for family, block in layout.unpack(graph).items()})
    return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}


def lock_observations() -> dict[str, np.ndarray]:
    """Raw native-unit observations of one world with padded columns."""
    rng = np.random.RandomState(SEEDS["observations"])
    n_weeks, n_context = 8, 5
    channel_mask = np.array([1, 1, 0])
    control_mask = np.array([1, 0])
    return {
        "spend": np.exp(rng.normal(8.0, 0.5, (n_weeks, 3))) * channel_mask,
        "controls": rng.normal(0.0, 1.0, (n_weeks, 2)) * control_mask,
        "context_outcome": np.exp(rng.normal(6.0, 0.2, n_context)),
        "channel_mask": channel_mask,
        "control_mask": control_mask,
        "latent_mask": np.array([1, 0]),
    }


def lock_outputs() -> tuple[dict[str, object], float]:
    """Every locked output, and the smallest active window-mean graph coordinate."""
    model = lock_model()
    batch = lock_batch()
    outputs: dict[str, object] = {}
    encoded = observation_batch(**lock_observations(), model_config=model.config)
    for key, value in encoded.items():
        outputs[f"observation_encoding/{key}"] = value.tolist()

    # The gradients the trainer's backward pass takes from the default-weight loss.
    torch.manual_seed(SEEDS["loss_terms"])
    model.loss_terms(batch, weights=LossWeights())["loss"].backward()
    for name, parameter in model.named_parameters():
        # A parameter the loss stops reaching has no gradient; it records as zeros.
        gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        outputs[f"loss_gradients/default/{name}"] = gradient.tolist()

    router_flows = []
    sample = model.sample

    def recording_sample(*args, **kwargs):
        result = sample(*args, **kwargs)
        if kwargs.get("return_structure"):
            router_flows.append(result[1])
        return result

    # Record every router draw, including those made inside sample_disentangled.
    model.sample = recording_sample
    layout = EdgeSlotLayout(3, 2, 2)
    with torch.no_grad():
        weekly, channels, controls = model.encoder.forward_with_controls(batch)
        probabilities = model.struct_head.predictive_probs(
            model.structure_logits(weekly, channels, controls)
        )
        for family, value in probabilities.items():
            outputs[f"structure_probabilities/{family}"] = value.tolist()
        for label, weights in (
            ("default", LossWeights()),
            ("weighted", LossWeights(**LOSS_WEIGHTS)),
        ):
            torch.manual_seed(SEEDS["loss_terms"])
            terms = model.loss_terms(batch, weights=weights)
            for name, value in terms.items():
                outputs[f"loss_terms/{label}/{name}"] = value.item()
        for name, draw in (("predict", model.predict), ("sample_context", model.sample_context)):
            torch.manual_seed(SEEDS[name])
            components, flow = draw(batch, N_SAMPLES, return_structure=True)
            outputs[f"{name}/components"] = components.tolist()
            outputs[f"{name}/graph_flow"] = flow.tolist()
            outputs[f"{name}/graphs"] = binary_graphs_from_flow(flow).tolist()
        truth = layout.pack({family: batch[f"g_{family}"] for family in layout.slices})
        cases = [
            (f"{mode}/{window}", mode, window, N_SAMPLES, SEEDS["sample_disentangled"] + index)
            for index, (mode, window) in enumerate(itertools.product(MODES, WINDOWS))
        ]
        cases.append(
            (
                f"model_average/query/n{CHUNK_CROSSING_N_SAMPLES}",
                "model_average",
                "query",
                CHUNK_CROSSING_N_SAMPLES,
                SEEDS["chunk_crossing"],
            )
        )
        for label, mode, window, n_samples, seed in cases:
            torch.manual_seed(seed)
            components, graphs = model.sample_disentangled(
                batch,
                n_samples,
                mode=mode,
                window=window,
                graph=truth if mode == "fixed" else None,
            )
            outputs[f"sample_disentangled/{label}/components"] = components.tolist()
            outputs[f"sample_disentangled/{label}/graphs"] = graphs.tolist()
    active = (
        layout.active_slot_mask(
            batch["active_c_mask"], batch["active_m_mask"], batch["active_j_mask"]
        )
        > 0
    )
    margin = min(flow.mean(dim=-2)[:, active].abs().min().item() for flow in router_flows)
    return outputs, margin


def producer() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "pymc-mfm": version("pymc-mfm"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }


def render(payload: dict[str, object]) -> str:
    """Valid JSON with one recorded output per line."""

    def dumps(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    header = [f"{dumps(key)}:{dumps(payload[key])}," for key in sorted(payload) if key != "outputs"]
    outputs = sorted(payload["outputs"].items())
    rows = [f"{dumps(name)}:{dumps(value)}" for name, value in outputs]
    return "{\n" + "\n".join(header) + '\n"outputs":{\n' + ",\n".join(rows) + "\n}\n}\n"


@pytest.fixture(scope="module")
def recorded():
    golden = json.loads(GOLDEN.read_text())
    assert golden["recipe"] == RECIPE, "the recorded lock was made from another recipe"
    with torch.random.fork_rng(devices=[]):
        outputs, _ = lock_outputs()
    assert outputs.keys() == golden["outputs"].keys()
    return golden["outputs"], outputs


def _mismatches(recorded, selected, compare) -> list[str]:
    expected, actual = recorded
    failures = []
    for name in sorted(expected):
        if selected(name):
            try:
                compare(np.asarray(actual[name]), np.asarray(expected[name]), err_msg=name)
            except AssertionError as error:
                failures.append(str(error))
    return failures


def _is_encoding(name: str) -> bool:
    return name.startswith("observation_encoding/")


def _is_graph(name: str) -> bool:
    return name.endswith("/graphs")


def _is_gradient(name: str) -> bool:
    return name.startswith("loss_gradients/")


def test_raw_observation_encoding_matches_the_lock(recorded):
    failures = _mismatches(
        recorded, _is_encoding, partial(np.testing.assert_allclose, **ENCODING_TOLERANCE)
    )
    assert not failures, "\n".join(failures)


def test_losses_probabilities_and_draws_match_the_lock(recorded):
    failures = _mismatches(
        recorded,
        lambda name: not (_is_encoding(name) or _is_gradient(name) or _is_graph(name)),
        partial(np.testing.assert_allclose, **MODEL_TOLERANCE),
    )
    assert not failures, "\n".join(failures)


def test_default_loss_gradients_match_the_lock(recorded):
    failures = _mismatches(
        recorded, _is_gradient, partial(np.testing.assert_allclose, **MODEL_TOLERANCE)
    )
    assert not failures, "\n".join(failures)


def test_graph_draws_match_the_lock_exactly(recorded):
    failures = _mismatches(recorded, _is_graph, np.testing.assert_array_equal)
    assert not failures, "\n".join(failures)


def test_default_configuration_is_the_ab_size_and_geometry():
    assert ModelConfig().to_dict() == {
        "format_version": 2,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "flow_hidden": 256,
        "n_controls": 6,
        "n_demand": 1,
    }


def test_ab_geometry_has_the_recorded_tensors_and_module_sizes():
    with torch.device("meta"):
        model = MFM(ModelConfig(n_controls=6))
    assert len(model.state_dict()) == 156
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in model.named_children()
    } == {
        "encoder": 1_316_480,
        "head": 544_880,
        "struct_head": 50_568,
        "graph_conditioner": 182_784,
        "conditional_head": 348_871,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.test_architecture_lock",
        description="Record the architecture lock outputs of the installed package.",
    )
    parser.add_argument("--write", action="store_true", required=True, help=f"write {GOLDEN}")
    parser.add_argument("--force", action="store_true", help="replace an existing recording")
    args = parser.parse_args(argv)
    if GOLDEN.exists() and not args.force:
        parser.error(f"{GOLDEN} exists; pass --force to replace the recorded lock")
    outputs, margin = lock_outputs()
    if margin < MIN_GRAPH_MARGIN:
        parser.error(f"a graph coordinate lies {margin:.3g} from the sign threshold; reseed")
    payload = {
        "format_version": 1,
        "producer": producer(),
        "recipe": RECIPE,
        "checks": {"min_abs_window_mean_graph_coordinate": margin},
        "outputs": outputs,
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(render(payload))
    print(GOLDEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
