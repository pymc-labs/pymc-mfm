import pytest

from pymc_mfm.client import curve, grid, move, scale, scenario, set_controls, shift


def test_scale_with_and_without_channels():
    assert scale(0.5) == {"scale": 0.5}
    assert scale(1.5, ["tv"]) == {"scale": 1.5, "channels": ["tv"]}
    assert scale(1.5, channels=("tv", "search"))["channels"] == ["tv", "search"]


def test_move_by_share_or_amount():
    assert move("tv", "search", share=0.2) == {"move": {"from": "tv", "to": "search", "share": 0.2}}
    assert move("tv", "search", amount=50) == {"move": {"from": "tv", "to": "search", "amount": 50}}


@pytest.mark.parametrize("arguments", [{}, {"share": 0.1, "amount": 5}])
def test_move_needs_exactly_one_of_share_and_amount(arguments):
    with pytest.raises(ValueError, match="share or amount"):
        move("tv", "search", **arguments)


def test_shift_by_weeks_or_pattern():
    assert shift("tv", into_weeks=(1, 4)) == {"shift": {"channel": "tv", "into_weeks": [1, 4]}}
    assert shift("tv", pattern=[1, 0, 2]) == {"shift": {"channel": "tv", "pattern": [1, 0, 2]}}
    with pytest.raises(ValueError, match="into_weeks or pattern"):
        shift("tv")
    with pytest.raises(ValueError, match="into_weeks or pattern"):
        shift("tv", into_weeks=(1, 2), pattern=[1])


def test_set_controls_takes_one_series_per_control():
    assert set_controls(price=(9, 9)) == {"set_controls": {"price": [9, 9]}}
    with pytest.raises(ValueError):
        set_controls()


def test_curve_defaults_to_the_baseline():
    assert curve("tv", [0, 1.5]) == {"channel": "tv", "levels": [0, 1.5], "base": "baseline"}
    assert curve("tv", [2], base="half")["base"] == "half"


def test_scenario_has_steps_or_plan():
    steps = [scale(0.5), shift("tv", into_weeks=(1, 4))]
    assert scenario("half_frontload", *steps) == {"id": "half_frontload", "steps": steps}
    plan = {"spend": {"tv": [1.0, 2.0]}}
    assert scenario("mine", plan=plan) == {"id": "mine", "plan": plan}
    with pytest.raises(ValueError):
        scenario("empty")
    with pytest.raises(ValueError):
        scenario("both", scale(2), plan=plan)


def test_grid_expands_to_explicit_scenarios_with_stable_ids():
    scenarios = grid(tv=[0.5, 1, 1.5], search=[1, 2])
    assert [item["id"] for item in scenarios] == [
        "tv_0.5",
        "tv_0.5__search_2",
        "search_2",
        "tv_1.5",
        "tv_1.5__search_2",
    ]
    by_id = {item["id"]: item for item in scenarios}
    assert by_id["tv_0.5__search_2"]["steps"] == [
        {"scale": 0.5, "channels": ["tv"]},
        {"scale": 2, "channels": ["search"]},
    ]
    assert by_id["search_2"]["steps"] == [{"scale": 2, "channels": ["search"]}]


def test_grid_leaves_out_the_baseline_combination_and_is_repeatable():
    first = grid(tv=[1, 2])
    assert [item["id"] for item in first] == ["tv_2"]
    assert grid(tv=[1, 2]) == first


def test_grid_cleans_channel_names_and_checks_ids():
    assert grid(**{"paid social": [2]})[0]["id"] == "paid_social_2"
    with pytest.raises(ValueError, match="duplicate"):
        grid(tv=[2, 2])
    with pytest.raises(ValueError, match="64"):
        grid(**{"c" * 70: [2]})
    with pytest.raises(ValueError):
        grid()
