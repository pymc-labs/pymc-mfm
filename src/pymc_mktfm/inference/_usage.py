"""Private input-volume accounting from validated observation geometry."""

from __future__ import annotations


def _input_usage(
    *,
    context_steps: int,
    channel_slots: int,
    control_slots: int,
    represented_channels: int,
    represented_controls: int,
    query_steps: int | None = None,
) -> dict[str, object]:
    """Count time/entity cells, including one revenue entity per step."""
    allocated_entities = int(channel_slots) + int(control_slots) + 1
    represented_entities = int(represented_channels) + int(represented_controls) + 1
    context_steps = int(context_steps)
    usage = {
        "context": {
            "steps": context_steps,
            "allocated_tokens": context_steps * allocated_entities,
            "represented_tokens": context_steps * represented_entities,
        }
    }
    if query_steps is not None:
        query_steps = int(query_steps)
        usage["query"] = {
            "steps": query_steps,
            "allocated_tokens": query_steps * allocated_entities,
            "represented_tokens": query_steps * represented_entities,
        }
        usage["total"] = {
            "allocated_tokens": (context_steps + query_steps) * allocated_entities,
            "represented_tokens": (context_steps + query_steps) * represented_entities,
        }
    return usage
