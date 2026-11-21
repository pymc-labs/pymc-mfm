"""The client reads MKTFM_* variables first and falls back to the old MFM_* names."""

import httpx
import pytest

from pymc_mktfm.client import Client

NAMES = ("MKTFM_API_URL", "MKTFM_API_KEY", "MFM_API_URL", "MFM_API_KEY")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def make_client():
    return Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))


def test_new_names_win_over_old_names(monkeypatch):
    monkeypatch.setenv("MKTFM_API_URL", "https://new.example.com/")
    monkeypatch.setenv("MKTFM_API_KEY", "new-key")
    monkeypatch.setenv("MFM_API_URL", "https://old.example.com/")
    monkeypatch.setenv("MFM_API_KEY", "old-key")
    client = make_client()
    assert client._http.base_url == "https://new.example.com"
    assert client._http._api_key == "new-key"


def test_old_names_still_work(monkeypatch):
    monkeypatch.setenv("MFM_API_URL", "https://old.example.com/")
    monkeypatch.setenv("MFM_API_KEY", "old-key")
    client = make_client()
    assert client._http.base_url == "https://old.example.com"
    assert client._http._api_key == "old-key"


def test_each_variable_falls_back_on_its_own(monkeypatch):
    monkeypatch.setenv("MKTFM_API_URL", "https://new.example.com")
    monkeypatch.setenv("MFM_API_KEY", "old-key")
    client = make_client()
    assert client._http.base_url == "https://new.example.com"
    assert client._http._api_key == "old-key"


def test_arguments_win_over_both(monkeypatch):
    monkeypatch.setenv("MKTFM_API_URL", "https://new.example.com")
    monkeypatch.setenv("MKTFM_API_KEY", "new-key")
    client = Client(
        "https://arg.example.com",
        api_key="arg-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    assert client._http.base_url == "https://arg.example.com"
    assert client._http._api_key == "arg-key"


def test_missing_variables_name_the_new_variables():
    with pytest.raises(ValueError, match="MKTFM_API_URL"):
        Client(api_key="k")
    with pytest.raises(ValueError, match="MKTFM_API_KEY"):
        Client("https://api.example.com")
