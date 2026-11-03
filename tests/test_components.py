"""Component registry: defaults, contract refusals, research flag and release-path refusals."""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from pymc_mfm.inference import from_pretrained
from pymc_mfm.inference._artifact_schema import BUNDLE_FILES, digest_bytes, file_record, json_bytes
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.model.components import (
    REQUIRED_MEMBERS,
    ConditionalHead,
    Encoder,
    ResearchComponentError,
    RouterHead,
)
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.encoder import BiAttentionEncoder
from pymc_mfm.model.flow import JointFlowHead
from pymc_mfm.model.graph import GraphConditioner, StructureHead
from pymc_mfm.model.model import MFM
from pymc_mfm.model.registry import (
    DEFAULT_COMPONENTS,
    ENCODERS,
    HEADS,
    is_research,
    register_encoder,
    register_head,
    resolve_components,
)
from pymc_mfm.training.checkpoint import load_checkpoint, save_checkpoint

RESEARCH_COMPONENTS = {
    "encoder": "bi_attention",
    "head": "joint_flow",
    "conditional_head": "toy_small_flow",
}


def tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=4, n_layers=1, n_heads=1, d_ff=8, flow_hidden=8, n_controls=1, n_demand=1
    )


def mark_as_research(bundle: Path, components: dict[str, str] | None) -> None:
    """Set (or with ``None`` remove) the schemas' ``components`` mark; recompute every hash."""
    config = _read(bundle / "config.json")
    if components is None:
        config["schemas"].pop("components", None)
    else:
        config["schemas"]["components"] = components
    config_bytes = json_bytes(config)
    (bundle / "config.json").write_bytes(config_bytes)

    with safe_open(str(bundle / "model.safetensors"), framework="pt") as stream:
        header = {**stream.metadata(), "config_sha256": digest_bytes(config_bytes)}
        tensors = {name: stream.get_tensor(name).clone() for name in stream.keys()}
    save_file(tensors, str(bundle / "model.safetensors"), metadata=header)

    manifest = _read(bundle / "manifest.json")
    manifest["files"] = {name: file_record(bundle / name) for name in BUNDLE_FILES[:2]}
    manifest.pop("bundle_sha256")
    manifest["bundle_sha256"] = digest_bytes(json_bytes(manifest))
    (bundle / "manifest.json").write_bytes(json_bytes(manifest))


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def build_research_bundle(destination: Path) -> Path:
    """Write the research bundle to a new ``destination`` directory."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MFM(tiny_config()).eval()
    bundle = save_pretrained(model, destination)
    mark_as_research(bundle, RESEARCH_COMPONENTS)
    return bundle


@pytest.fixture
def research_bundle(tmp_path):
    """A valid tiny bundle whose only defect is the research mark."""
    return build_research_bundle(tmp_path / "research_bundle")


@pytest.fixture
def toy_head():
    """JointFlowHead with a smaller flow, registered for one test."""

    class ToySmallFlow(JointFlowHead):
        def __init__(self, config: ModelConfig, *, graph_router: bool):
            small = ModelConfig.from_dict({**config.to_dict(), "flow_hidden": 8})
            super().__init__(small, graph_router=graph_router)

    register_head("toy_small_flow", ToySmallFlow)
    yield ToySmallFlow
    del HEADS["toy_small_flow"]


def _stub(base: type, missing: str, members: tuple[str, ...]) -> type:
    """A class with every required member of ``base`` except ``missing``."""
    body: dict[str, object] = {"__annotations__": {}}
    for name in members:
        if name == missing:
            continue
        if name in ("n_controls", "n_demand"):
            body["__annotations__"][name] = int
        elif name == "n_fixed":
            body[name] = property(lambda self: 4)
        elif name == "_draw_noise":
            body[name] = staticmethod(lambda shape, *, device, dtype: None)
        else:
            body[name] = lambda self, *args, **kwargs: None
    return type("Stub", (base,), body)


def test_defaults_resolve_through_the_registry(model):
    assert resolve_components() == DEFAULT_COMPONENTS
    assert not is_research(None)
    assert model.components == DEFAULT_COMPONENTS
    assert model.is_research is False
    assert type(model.encoder) is BiAttentionEncoder
    assert type(model.head) is JointFlowHead
    assert type(model.conditional_head) is JointFlowHead
    assert issubclass(JointFlowHead, ConditionalHead)
    assert issubclass(BiAttentionEncoder, Encoder)


def test_components_stay_out_of_the_config(model):
    assert "components" not in model.config.to_dict()


def test_default_construction_keeps_the_original_rng_sequence():
    config = ModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16)
    torch.manual_seed(3)
    model = MFM(config)
    rng_after_model = torch.random.get_rng_state()

    torch.manual_seed(3)
    modules = {
        "encoder": BiAttentionEncoder(config),
        "head": JointFlowHead(config, graph_router=True),
        "struct_head": StructureHead(
            d_channel=config.d_model,
            d_global=config.d_model,
            n_demand=config.n_demand,
            n_controls=config.n_controls,
        ),
        "graph_conditioner": GraphConditioner(config.d_model, config.d_model),
        "conditional_head": JointFlowHead(config, graph_router=False),
    }
    expected = {
        f"{name}.{key}": value
        for name, module in modules.items()
        for key, value in module.state_dict().items()
    }
    assert model.state_dict().keys() == expected.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name]), name
    assert torch.equal(rng_after_model, torch.random.get_rng_state())


def test_unknown_slot_and_name_are_refused():
    with pytest.raises(ValueError, match="unknown component slots"):
        resolve_components({"decoder": "x"})
    with pytest.raises(ValueError, match="not registered"):
        resolve_components({"head": "nope"})


def test_default_classes_pass_every_contract():
    for registry, contract in ((ENCODERS, Encoder), (HEADS, RouterHead)):
        for cls in registry.values():
            assert issubclass(cls, contract)


@pytest.mark.parametrize("missing", REQUIRED_MEMBERS[Encoder])
def test_encoder_missing_one_member_is_refused(missing):
    cls = _stub(Encoder, missing, REQUIRED_MEMBERS[Encoder])
    with pytest.raises(TypeError, match=missing):
        register_encoder("broken_encoder", cls)
    assert "broken_encoder" not in ENCODERS


@pytest.mark.parametrize("missing", REQUIRED_MEMBERS[ConditionalHead])
def test_head_missing_one_member_is_refused(missing):
    cls = _stub(ConditionalHead, missing, REQUIRED_MEMBERS[ConditionalHead])
    with pytest.raises(TypeError, match=missing):
        register_head("broken_head", cls)
    assert "broken_head" not in HEADS


def test_a_class_outside_the_contract_is_refused():
    class Unrelated:
        pass

    with pytest.raises(TypeError, match="must subclass"):
        register_head("unrelated", Unrelated)
    with pytest.raises(TypeError, match="must subclass"):
        register_encoder("unrelated", Unrelated)


def test_a_complete_stub_registers_and_duplicates_are_refused():
    cls = _stub(Encoder, "", REQUIRED_MEMBERS[Encoder])
    register_encoder("complete_stub", cls)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_encoder("complete_stub", cls)
        assert is_research({"encoder": "complete_stub"})
    finally:
        del ENCODERS["complete_stub"]


def test_a_router_only_head_cannot_fill_the_conditional_slot():
    cls = _stub(RouterHead, "", REQUIRED_MEMBERS[RouterHead])
    register_head("router_only", cls)
    try:
        assert is_research({"head": "router_only"})
        with pytest.raises(TypeError, match="ConditionalHead"):
            resolve_components({"conditional_head": "router_only"})
    finally:
        del HEADS["router_only"]


def test_toy_head_builds_a_research_model_that_trains_and_samples(toy_head, batch):
    torch.manual_seed(10)
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=2
    )
    research = MFM(
        config, components={"head": "toy_small_flow", "conditional_head": "toy_small_flow"}
    ).eval()
    assert research.is_research is True
    assert isinstance(research.head, toy_head)
    assert sum(p.numel() for p in research.head.parameters()) < sum(
        p.numel() for p in MFM(config).head.parameters()
    )
    terms = research.loss_terms(batch)
    assert torch.isfinite(terms["task_loss"])
    samples, graphs = research.sample_disentangled(batch, 3)
    assert samples.shape[:2] == (3, 2)
    assert torch.isfinite(samples).all()
    assert graphs.shape[:2] == (3, 2)


def test_save_pretrained_refuses_a_research_model(toy_head, tmp_path):
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=2
    )
    research = MFM(config, components={"conditional_head": "toy_small_flow"})
    with pytest.raises(ResearchComponentError, match="research components"):
        save_pretrained(research, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_a_research_checkpoint_whose_tensors_match_the_default_never_loads_as_it(tmp_path):
    class ReversedSampling(JointFlowHead):
        """Same parameters as the default head; different samples."""

        def sample(self, *args, **kwargs):
            return -super().sample(*args, **kwargs)

    register_head("reversed_sampling", ReversedSampling)
    try:
        research = MFM(tiny_config(), components={"conditional_head": "reversed_sampling"})
        assert research.state_dict().keys() == MFM(tiny_config()).state_dict().keys()
        _save_checkpoint(research, tmp_path / "research.pt")
        with pytest.raises(ResearchComponentError):
            load_checkpoint(tmp_path / "research.pt")
        loaded, _ = load_checkpoint(tmp_path / "research.pt", allow_research=True)
        assert loaded.components == research.components
        assert isinstance(loaded.conditional_head, ReversedSampling)
    finally:
        del HEADS["reversed_sampling"]


def _save_checkpoint(model, path):
    save_checkpoint(
        path,
        model=model,
        kind="last",
        step=1,
        training_config={},
        run={},
        consumed={},
        evaluation=None,
        selection={},
    )


def test_from_pretrained_refuses_a_research_bundle(research_bundle):
    with pytest.raises(ResearchComponentError, match="research components"):
        from_pretrained(research_bundle)


def test_the_mark_is_the_only_reason_the_bundle_is_refused(research_bundle):
    # Removing the mark and recomputing the hashes leaves a bundle that loads.
    mark_as_research(research_bundle, None)
    loaded = from_pretrained(research_bundle)
    assert loaded.is_research is False
    assert loaded.components == DEFAULT_COMPONENTS


def test_a_default_bundle_still_loads(model, tmp_path):
    path = save_pretrained(model, tmp_path / "bundle", provenance={})
    loaded = from_pretrained(path)
    assert loaded.components == DEFAULT_COMPONENTS
    assert loaded.is_research is False
