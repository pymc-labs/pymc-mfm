"""Research heads for the training tests; importing this module registers them.

``--plugin research_plugin`` registers the same heads in a CLI run.
"""

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.flow import JointFlowHead
from pymc_mfm.model.registry import register_head

SMALL_FLOW = {"head": "p6b_small_flow", "conditional_head": "p6b_small_flow"}
SAME_SHAPE = {"head": "p6b_same_shape", "conditional_head": "p6b_same_shape"}


class SmallFlow(JointFlowHead):
    """The release head with a narrower flow, so its tensor shapes differ."""

    def __init__(self, config: ModelConfig, *, graph_router: bool):
        small = ModelConfig.from_dict({**config.to_dict(), "flow_hidden": 8})
        super().__init__(small, graph_router=graph_router)


class SameShape(JointFlowHead):
    """The release head under another name: same tensor names and shapes."""


register_head("p6b_small_flow", SmallFlow)
register_head("p6b_same_shape", SameShape)
