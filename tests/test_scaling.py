"""Context-fitted unit invariance at the shared observation boundary."""

import numpy as np
import pytest

from pymc_mfm.data.observations import encode_observations


@pytest.fixture
def observations():
    rng = np.random.default_rng(78123)
    spend = rng.uniform(0.25, 2.0, size=(2, 40, 5))
    spend[:, :, 2] = 3.0
    spend[:, :, 3:] = 0.0
    controls = rng.normal(size=(2, 40, 5))
    controls[:, :, 1] += 1.0
    controls[:, :, 2] = -3.0
    controls[:, :, 3:] = 0.0
    outcome = (
        2.0
        + 0.2 * spend[:, :32, 0]
        - 0.1 * controls[:, :32, 0]
        + rng.normal(0.0, 0.03, size=(2, 32))
    )
    return {
        "spend": spend,
        "controls": controls,
        "context_outcome": outcome,
        "active_c": np.tile([1, 1, 1, 1, 0], (2, 1)),
        "active_m": np.tile([1, 1, 1, 1, 0], (2, 1)),
        "active_j": np.tile([1, 0], (2, 1)),
    }


def test_independent_units_preserve_every_feature_and_restore_outcome_scale(observations):
    reference = encode_observations(**observations)
    spend_units = np.array(
        [[1e307, 1e-307, 17.0, 1e-120, 1e120], [1e-250, 1e250, 0.125, 1e240, 1e-240]]
    )
    control_units = np.array(
        [[1e-307, 1e307, 8e-70, 1e80, 1e-80], [1e250, 1e-250, 4e90, 1e-90, 1e90]]
    )
    outcome_units = np.array([1e307, 1e-307])
    converted = encode_observations(
        **{
            **observations,
            "spend": observations["spend"] * spend_units[:, None],
            "controls": observations["controls"] * control_units[:, None],
            "context_outcome": observations["context_outcome"] * outcome_units[:, None],
        }
    )
    for key, value in converted.items():
        assert np.isfinite(value).all(), key
        if key != "sales_scale":
            np.testing.assert_allclose(value, reference[key], rtol=2e-6, atol=1e-6, err_msg=key)
    assert converted["sales_scale"].dtype == np.float64
    np.testing.assert_allclose(
        converted["sales_scale"], reference["sales_scale"] * outcome_units, rtol=5e-15, atol=0
    )
    np.testing.assert_allclose(
        reference["sales_scale"], np.quantile(observations["context_outcome"], 0.95, axis=1)
    )


def test_log_spend_statistics_and_query_use_the_context_mean_ruler():
    encoded = encode_observations(
        spend=np.array([[[2.0], [6.0], [100.0], [0.0]]]),
        controls=np.zeros((1, 4, 1)),
        context_outcome=np.array([[4.0, 8.0]]),
        active_c=np.ones((1, 1)),
        active_m=np.ones((1, 1)),
        active_j=np.ones((1, 1)),
    )
    # The context mean is four, regardless of the much larger query spend.
    log_spend = np.log([1.5, 2.5, 26.0, 1.0])
    mean = (log_spend[0] + log_spend[1]) / 2
    std = (log_spend[1] - log_spend[0]) / 2
    np.testing.assert_allclose(encoded["spend_mean"], [[mean]])
    np.testing.assert_allclose(encoded["spend_std"], [[std]])
    np.testing.assert_allclose(encoded["ctx_spend_z"][0, :, 0], (log_spend[:2] - mean) / std)
    np.testing.assert_allclose(encoded["qry_spend_z"][0, :, 0], (log_spend[2:] - mean) / std)


def test_signed_constant_zero_and_padded_columns_keep_their_meaning():
    encoded = encode_observations(
        spend=np.array([[[2.0, 0, 0]] * 4 + [[4.0, 0, 0], [8.0, 0, 0]]]),
        controls=np.array(
            [
                [
                    [-3.0, -4, 0, 0],
                    [-1, -4, 0, 0],
                    [1, -4, 0, 0],
                    [3, -4, 0, 0],
                    [5, -8, 0, 0],
                    [100, 8, 0, 0],
                ]
            ]
        ),
        context_outcome=np.array([[2.0, 4, 8, 16]]),
        active_c=np.array([[1, 1, 0]]),
        active_m=np.array([[1, 1, 1, 0]]),
        active_j=np.array([[0, 1]]),
    )
    np.testing.assert_allclose(
        encoded["control_location_scale"], [[[0, 1], [-1, 1], [0, 1], [0, 0]]]
    )
    np.testing.assert_allclose(
        encoded["ctx_controls_z"][0, :, 0], np.array([-3, -1, 1, 3]) / np.sqrt(5)
    )
    np.testing.assert_array_equal(encoded["ctx_controls_z"][0, :, 1:], 0)
    np.testing.assert_allclose(
        encoded["qry_controls_z"], [[[5 / np.sqrt(5), -1, 0, 0], [10, 3, 0, 0]]]
    )
    np.testing.assert_allclose(encoded["spend_mean"], [[np.log(2), 0, 0]])
    np.testing.assert_array_equal(encoded["spend_std"], 0)
    np.testing.assert_array_equal(encoded["ctx_spend_z"], 0)
    np.testing.assert_array_equal(encoded["qry_spend_z"][0, :, 1:], 0)
    np.testing.assert_allclose(encoded["qry_spend_z"][0, :, 0], (np.log([3, 5]) - np.log(2)) / 1e-6)
    for key in ("response_moments", "adstock_moments"):
        np.testing.assert_array_equal(encoded[key], 0)
    for key in ("control_moments", "control_effect_moments"):
        np.testing.assert_array_equal(encoded[key][0, 1:], 0)


def test_query_changes_cannot_change_any_context_feature_or_scale(observations):
    reference = encode_observations(**observations)
    changed = {key: value.copy() for key, value in observations.items()}
    n_context = observations["context_outcome"].shape[1]
    changed["spend"][:, n_context:, :3] *= 300
    changed["controls"][:, n_context:, :3] = -100 * changed["controls"][:, n_context:, :3] + 12
    encoded = encode_observations(**changed)
    for key in reference:
        if not key.startswith("qry_"):
            np.testing.assert_array_equal(encoded[key], reference[key], err_msg=key)


@pytest.mark.parametrize("name", ["spend", "controls"])
def test_zero_context_refuses_even_the_smallest_nonzero_query(observations, name):
    n_context = observations["context_outcome"].shape[1]
    observations[name][0, n_context, 3] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match=rf"{name}.*zero context.*nonzero query"):
        encode_observations(**observations)


@pytest.mark.parametrize(
    ("name", "context_value", "outlier"),
    [
        ("spend", 1e-300, 1e300),
        ("controls", 1e300, 1e-300),
        ("context_outcome", 1e-300, 1e300),
        ("context_outcome", 1e300, 1e-300),
    ],
)
def test_unrepresentable_relative_dynamics_are_rejected(observations, name, context_value, outlier):
    if name == "context_outcome":
        observations[name][0] = context_value
        observations[name][0, 0] = outlier
    else:
        observations[name][0, :, 0] = context_value
        observations[name][0, -1, 0] = outlier
    with pytest.raises(ValueError, match=rf"{name}.*float64 context units"):
        encode_observations(**observations)
