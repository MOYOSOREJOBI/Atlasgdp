from __future__ import annotations

from pathlib import Path

from atlas_gdp.settings import load_settings


def test_settings_env_overrides_yaml(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "data:",
                "  countries: [BRA, USA]",
                "  release_lag_days: 60",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATLAS_GDP_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "custom_artifacts"))
    monkeypatch.setenv("ATLAS_GDP_CONFIG", str(config_path))
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("DEFAULT_COUNTRIES", "CAN,MEX")
    monkeypatch.setenv("DEFAULT_RELEASE_LAG_DAYS", "30")

    settings = load_settings()

    assert settings.offline_mode is True
    assert settings.default_countries == ["CAN", "MEX"]
    assert settings.default_release_lag_days == 30
    assert settings.raw_config["data"]["countries"] == ["CAN", "MEX"]
    assert settings.raw_config["data"]["release_lag_days"] == 30
    assert settings.paths.artifacts == (tmp_path / "custom_artifacts").resolve()
    assert settings.settings_snapshot["config_path"] == str(config_path.resolve())
