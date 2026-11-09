"""Local weekly-date checks: messages name the first bad row and its dates."""

from datetime import date

import pytest

from pymc_mfm.client._dates import check_plan_follows_history, check_weekly_dates

WEEKS = ["2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22"]


def test_weekly_dates_pass():
    check_weekly_dates(WEEKS, "history")
    check_weekly_dates([], "history")
    check_weekly_dates([date(2024, 7, 1)], "history")


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (
            [WEEKS[0], WEEKS[1], WEEKS[3]],
            "Row 3 of your history has the date 2024-07-22, but row 2 has 2024-07-08. "
            "The next required date is 2024-07-15. "
            "Check whether a week is missing from your history.",
        ),
        (
            [WEEKS[0], WEEKS[1], WEEKS[1]],
            "Row 3 of your history repeats the date 2024-07-08, which is also in row 2.",
        ),
        (
            [WEEKS[0], WEEKS[2], WEEKS[1], WEEKS[3]],
            "Row 3 of your history has the date 2024-07-08, but row 2 has 2024-07-15.",
        ),
        (
            [WEEKS[0], "2024-07-04"],
            "Row 2 of your history has the date 2024-07-04, but row 1 has 2024-07-01. "
            "Dates must be 7 days apart. The next required date is 2024-07-08.",
        ),
    ],
)
def test_first_bad_row_is_named(dates, message):
    with pytest.raises(ValueError) as error:
        check_weekly_dates(dates, "history")
    assert message in str(error.value)


def test_a_repeat_that_is_not_next_to_its_first_row_is_a_duplicate():
    with pytest.raises(ValueError, match="Row 3 .* repeats .* also in row 1"):
        check_weekly_dates([WEEKS[0], WEEKS[1], WEEKS[0]], "history")


def test_the_plan_must_start_the_week_after_the_history():
    required = "The next required date is 2024-07-29."
    for start in ("2024-08-05", "2024-07-22", "2024-07-15"):
        with pytest.raises(ValueError) as error:
            check_plan_follows_history({"dates": [start]}, date(2024, 7, 22))
        assert str(error.value) == (
            f"Your history ends on 2024-07-22. Your plan starts on {start}. {required} "
            "Check whether a week is missing from your history or plan."
        )


def test_plan_rows_are_checked_after_the_start():
    plan = {"dates": ["2024-07-29", "2024-08-05", "2024-08-19"]}
    with pytest.raises(
        ValueError, match="Row 3 of your plan .* 2024-08-19, but row 2 has 2024-08-05"
    ):
        check_plan_follows_history(plan, date(2024, 7, 22))


def test_a_plan_without_dates_is_left_to_the_service():
    check_plan_follows_history({"spend": {}}, date(2024, 7, 22))
    check_plan_follows_history({"dates": []}, date(2024, 7, 22))
