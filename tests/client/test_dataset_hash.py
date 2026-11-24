"""The content hash agrees with the shared vectors that the service tests also read."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pymc_mfm.client import dataset_content_hash, frame_content_hash

VECTORS = json.loads((Path(__file__).parents[1] / "data" / "dataset-hash-vectors.json").read_text())
BY_CASE = {vector["case"]: vector for vector in VECTORS["vectors"]}


def test_vector_file_names_its_format():
    assert VECTORS["format"] == "mfm-dataset-v1"


@pytest.mark.parametrize("case", list(BY_CASE))
def test_every_shared_vector(case):
    vector = BY_CASE[case]
    assert dataset_content_hash(vector["body"]) == vector["content_hash"]


def test_same_hash_cases_match_and_different_cases_differ():
    base = BY_CASE["base"]["content_hash"]
    same = [c for c in BY_CASE if c.endswith("_same_hash")]
    assert same and all(BY_CASE[c]["content_hash"] == base for c in same)
    for case in ("one_value_differs", "role_swapped", "no_controls", "unicode_name_column"):
        assert BY_CASE[case]["content_hash"] != base


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_are_an_error(bad):
    body = {**BY_CASE["base"]["body"], "outcome": [1, bad, 3, 4]}
    with pytest.raises(ValueError):
        dataset_content_hash(body)


def test_frame_hash_equals_the_body_hash():
    body = BY_CASE["base"]["body"]
    frame = pd.DataFrame(
        {
            "week": pd.to_datetime(body["dates"]),
            "sales": body["outcome"],
            **body["spend"],
            **body["controls"],
        }
    ).sample(frac=1, random_state=0)  # row order does not matter
    got = frame_content_hash(
        frame, date="week", outcome="sales", spend=["tv", "search"], controls="holiday"
    )
    assert got == BY_CASE["base"]["content_hash"]
