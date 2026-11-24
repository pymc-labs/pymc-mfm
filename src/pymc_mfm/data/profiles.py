"""A and B describe synthetic training priors, never different model architectures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactProfile:
    """Size and active-channel settings for a named synthetic corpus profile.

    A profile configures data generation; it neither identifies checkpoint
    weights nor changes ``ModelConfig``. Trained artifacts record their own
    data and checkpoint identities.
    """

    name: str
    worlds: int
    live_channels: tuple[int, int]


PROFILE_A = ArtifactProfile(
    name="A",
    worlds=200_000,
    live_channels=(5, 5),
)
PROFILE_B = ArtifactProfile(
    name="B",
    worlds=400_000,
    live_channels=(4, 10),
)
