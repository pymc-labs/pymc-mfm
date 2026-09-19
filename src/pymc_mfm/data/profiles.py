"""A and B identify external data/weights, never different model architectures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactProfile:
    """Identity and data differences recorded in the original A/B references.

    Locations are deliberately absent: callers supply a local path to loading,
    and a future artifact transport can resolve these same identities. The
    checksum identifies the original reference checkpoint, not a converted MFM
    artifact. Neither this record nor choosing A/B changes ``ModelConfig``.
    """

    name: str
    corpus: str
    checkpoint_sha256: str
    worlds: int
    live_channels: tuple[int, int]


PROFILE_A = ArtifactProfile(
    name="A",
    corpus="fw53_d100_r1",
    checkpoint_sha256="89a57f7ca17bb25173ddc0f85f061123217dca1ded3abd43079afd3014dd63e3",
    worlds=200_000,
    live_channels=(5, 5),
)
PROFILE_B = ArtifactProfile(
    name="B",
    corpus="fw53cy410_d200_r0",
    checkpoint_sha256="4cc082d9c24f26534e0f18c70d86e883b3f5da4957efa83f31359fc76a609792",
    worlds=400_000,
    live_channels=(4, 10),
)
