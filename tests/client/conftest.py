"""Fixtures for the client tests."""

import httpx
import pandas as pd
import pytest

from pymc_mktfm.client import Client

from .client_fixtures import BASE_URL, FakeService


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def client(service: FakeService, sleeps: list[float]) -> Client:
    """A client wired to the fake service. Waits are recorded, not slept."""
    instance = Client(BASE_URL, "test-key", max_retries=3, transport=httpx.MockTransport(service))
    instance._http._sleep = sleeps.append
    return instance


@pytest.fixture
def history() -> pd.DataFrame:
    weeks = pd.date_range("2024-07-01", periods=4, freq="W-MON")
    return pd.DataFrame(
        {
            "week": weeks,
            "sales": [1000.0, 1100.0, 1050.0, 1200.0],
            "tv": [100.0, 0.0, 120.0, 110.0],
            "search": [50.0, 55.0, 60.0, 45.0],
            "price": [9.9, 9.9, 10.5, 10.5],
        }
    )


@pytest.fixture
def plan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": pd.date_range("2024-07-29", periods=2, freq="W-MON"),
            "tv": [100.0, 100.0],
            "search": [50.0, 50.0],
            "price": [10.5, 10.5],
        }
    )
