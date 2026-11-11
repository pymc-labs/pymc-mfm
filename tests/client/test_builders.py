import pytest

from pymc_mfm.client import amount, curve, grid, scenario


def test_amount_sets_a_channel_total():
    assert amount(24000) == {"amount": 24000}


def test_scenario_has_an_id_and_multipliers():
    assert scenario("tv_to_social", {"tv": 0.8, "social": amount(24000)}) == {
        "id": "tv_to_social",
        "multipliers": {"tv": 0.8, "social": {"amount": 24000}},
    }


def test_scenario_copies_its_amounts():
    shared = amount(5)
    built = scenario("mine", {"tv": shared})
    shared["amount"] = 99
    assert built["multipliers"]["tv"] == {"amount": 5}


def test_scenario_refuses_what_is_not_a_dict_of_multipliers():
    with pytest.raises(TypeError, match="dict"):
        scenario("bad", [0.8])


def test_curve_defaults_to_the_baseline():
    assert curve("tv", [0, 1.5]) == {"channel": "tv", "levels": [0, 1.5], "base": "baseline"}
    assert curve("tv", [2], base="half")["base"] == "half"


def test_grid_expands_to_multiplier_scenarios_with_stable_ids():
    scenarios = grid({"tv": [0.5, 1, 1.5], "search": [1, 2]})
    assert [item["id"] for item in scenarios] == [
        "tv_0.5",
        "tv_0.5__search_2",
        "search_2",
        "tv_1.5",
        "tv_1.5__search_2",
    ]
    by_id = {item["id"]: item for item in scenarios}
    assert by_id["tv_0.5__search_2"]["multipliers"] == {"tv": 0.5, "search": 2}
    assert by_id["search_2"]["multipliers"] == {"search": 2}


def test_grid_ids_follow_the_axes_order():
    scenarios = grid({"tv": [0.5, 1.5], "search": [0.5, 1.5]})
    assert [item["id"] for item in scenarios] == [
        "tv_0.5__search_0.5",
        "tv_0.5__search_1.5",
        "tv_1.5__search_0.5",
        "tv_1.5__search_1.5",
    ]


def test_grid_leaves_out_the_baseline_combination_and_is_repeatable():
    first = grid({"tv": [1, 2]})
    assert [item["id"] for item in first] == ["tv_2"]
    assert grid({"tv": [1, 2]}) == first


def test_grid_cleans_channel_names_and_checks_ids():
    assert grid({"paid social": [2]})[0]["id"] == "paid_social_2"
    with pytest.raises(ValueError, match="duplicate"):
        grid({"tv": [2, 2]})
    with pytest.raises(ValueError, match="64"):
        grid({"c" * 70: [2]})
    with pytest.raises(ValueError, match="at least one channel"):
        grid({})
