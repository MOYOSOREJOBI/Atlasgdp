from __future__ import annotations

import pytest

from atlas_gdp.data.oecd_sdmx import load_oecd_quarterly
from atlas_gdp.data.sdmx_connector import ConnectorUnavailableError


def test_offline_mode_never_calls_network(monkeypatch, tmp_path) -> None:
    called = {"fetch": False}

    def fail_fetch():
        called["fetch"] = True
        raise AssertionError("network fetch should not be called in offline mode")

    monkeypatch.setattr("atlas_gdp.data.oecd_sdmx._fetch_oecd_payload", fail_fetch)

    with pytest.raises(ConnectorUnavailableError, match="Run once online to populate cache"):
        load_oecd_quarterly(tmp_path, offline_mode=True)

    assert called["fetch"] is False
