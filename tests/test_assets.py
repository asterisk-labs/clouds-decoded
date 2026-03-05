"""Unit tests for the clouds_decoded.assets module."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# get_assets_dir
# ---------------------------------------------------------------------------

class TestGetAssetsDir:
    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "env_assets"
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(env_dir))

        from clouds_decoded.assets import get_assets_dir
        result = get_assets_dir()

        assert result == env_dir
        assert result.exists()

    def test_creates_directory(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "does_not_exist_yet"
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(new_dir))
        assert not new_dir.exists()

        from clouds_decoded.assets import get_assets_dir
        get_assets_dir()

        assert new_dir.exists()

    def test_config_toml_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLOUDS_DECODED_ASSETS_DIR", raising=False)
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        config_dir.mkdir()
        assets_dir = tmp_path / "from_config"
        config_file.write_text(f'assets_dir = "{assets_dir}"\n')

        import clouds_decoded.assets as assets_mod
        with patch.object(assets_mod, "_CONFIG_FILE", config_file):
            result = assets_mod.get_assets_dir()

        assert result == assets_dir

    def test_platformdirs_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CLOUDS_DECODED_ASSETS_DIR", raising=False)

        import clouds_decoded.assets as assets_mod
        # Point config file to a non-existent path so it's skipped
        with patch.object(assets_mod, "_CONFIG_FILE", tmp_path / "nonexistent.toml"):
            with patch("platformdirs.user_data_dir", return_value=str(tmp_path / "default")):
                result = assets_mod.get_assets_dir()

        assert result == tmp_path / "default"


# ---------------------------------------------------------------------------
# get_asset
# ---------------------------------------------------------------------------

class TestGetAsset:
    def test_returns_correct_subpath(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.assets import get_asset
        result = get_asset("models/cloud_height_emulator/default.pth")

        assert result == tmp_path / "models" / "cloud_height_emulator" / "default.pth"

    def test_does_not_require_existence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.assets import get_asset
        result = get_asset("data/GEBCO_2024.nc")

        assert not result.exists()  # file absent — that's fine


# ---------------------------------------------------------------------------
# require_asset
# ---------------------------------------------------------------------------

class TestRequireAsset:
    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))
        weights = tmp_path / "models" / "cloud_height_emulator" / "default.pth"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"fake weights")

        from clouds_decoded.assets import require_asset
        result = require_asset("emulator")

        assert result == weights

    def test_raises_with_actionable_message_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.assets import require_asset
        with pytest.raises(FileNotFoundError) as exc_info:
            require_asset("emulator")

        msg = str(exc_info.value)
        assert "clouds-decoded download emulator" in msg
        assert "models/cloud_height_emulator/default.pth" in msg

    def test_raises_for_all_known_assets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.assets import require_asset, KNOWN_ASSETS
        for key in KNOWN_ASSETS:
            with pytest.raises(FileNotFoundError) as exc_info:
                require_asset(key)
            assert f"clouds-decoded download {key}" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CloudHeightEmulatorConfig — model_path resolution
# ---------------------------------------------------------------------------

class TestEmulatorConfigPthPath:
    def test_default_points_to_assets_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        # Re-import to pick up env var
        import importlib
        import clouds_decoded.assets
        importlib.reload(clouds_decoded.assets)

        from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
        cfg = CloudHeightEmulatorConfig()

        assert cfg.model_path == str(tmp_path / "models" / "cloud_height_emulator" / "default.pth")

    def test_explicit_path_not_overridden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
        cfg = CloudHeightEmulatorConfig(model_path="/custom/path/model.pth")

        assert cfg.model_path == "/custom/path/model.pth"


# ---------------------------------------------------------------------------
# CloudMaskConfig — model_path resolution
# ---------------------------------------------------------------------------

class TestCloudMaskConfigModelPath:
    def test_default_points_to_assets_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        import importlib
        import clouds_decoded.assets
        importlib.reload(clouds_decoded.assets)

        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        cfg = CloudMaskConfig()

        assert cfg.model_path == str(tmp_path / "models" / "cloud_mask" / "default.pt")

    def test_explicit_path_not_overridden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        cfg = CloudMaskConfig(model_path="/custom/path/model.pt")

        assert cfg.model_path == "/custom/path/model.pt"

    def test_threshold_method_still_resolves_path(self, tmp_path, monkeypatch):
        """model_path validator runs regardless of method."""
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        cfg = CloudMaskConfig(method="threshold")

        assert cfg.model_path == str(tmp_path / "models" / "cloud_mask" / "default.pt")


# ---------------------------------------------------------------------------
# Refl2PropConfig — model_path resolution
# ---------------------------------------------------------------------------

class TestRefl2PropConfigModelPath:
    def test_managed_asset_used_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))
        managed = tmp_path / "models" / "refl2prop" / "default.pth"
        managed.parent.mkdir(parents=True)
        managed.write_bytes(b"fake")

        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig
        cfg = Refl2PropConfig()

        assert cfg.model_path == str(managed)

    def test_managed_path_used_even_when_file_absent(self, tmp_path, monkeypatch):
        """When managed asset file does not yet exist the config still points at it.

        The bundled-model fallback has been removed — callers get a clear
        FileNotFoundError at inference time via require_asset / the processor.
        """
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))
        # managed file does NOT exist

        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig
        cfg = Refl2PropConfig()

        expected = str(tmp_path / "models" / "refl2prop" / "default.pth")
        assert cfg.model_path == expected

    def test_explicit_path_not_overridden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig
        cfg = Refl2PropConfig(model_path="/custom/model.pth")

        assert cfg.model_path == "/custom/model.pth"


# ---------------------------------------------------------------------------
# download_asset — mocked, no network
# ---------------------------------------------------------------------------

class TestDownloadAsset:
    def test_skips_when_file_exists_and_no_force(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))
        weights = tmp_path / "models" / "cloud_height_emulator" / "default.pth"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"existing")

        from clouds_decoded.assets import download_asset
        result = download_asset("emulator", force=False)

        captured = capsys.readouterr()
        assert "already present" in captured.out
        assert result == weights

    def test_raises_when_no_url(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        import clouds_decoded.assets as assets_mod
        from clouds_decoded.assets import download_asset, Asset
        # Temporarily register a fake asset with no URL
        assets_mod.KNOWN_ASSETS["_test_no_url"] = Asset(
            key="_test_no_url", relative_path="models/_test/x.pth",
            url="", size_hint="0", description="test asset",
        )
        try:
            with pytest.raises(ValueError, match="No download URL"):
                download_asset("_test_no_url")
        finally:
            del assets_mod.KNOWN_ASSETS["_test_no_url"]

    def test_downloads_non_zip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))

        import clouds_decoded.assets as assets_mod
        # Patch URL and urlretrieve
        assets_mod.KNOWN_ASSETS["emulator"].url = "http://example.com/model.pth"

        def fake_urlretrieve(url, dest, reporthook=None):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"downloaded")

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with patch("tqdm.tqdm"):
                result = assets_mod.download_asset("emulator", force=True)

        assert result == tmp_path / "models" / "cloud_height_emulator" / "default.pth"
        assert result.read_bytes() == b"downloaded"

        # Reset URL so other tests aren't affected
        assets_mod.KNOWN_ASSETS["emulator"].url = ""
