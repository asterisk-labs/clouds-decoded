"""Tests for the CLI, ProjectConfig, and config-hash integration.

Covers the user-facing surfaces of Phase 3 that live outside the core
DeviceDispatcher class:

  * ``DispatcherSettings`` validation and YAML round-trip.
  * ``Project.resolve_dispatcher_settings`` precedence (CLI > YAML >
    defaults).
  * ``BaseProcessorConfig._semantic_dump`` excludes non-semantic fields.
  * ``Project._config_hash`` is stable across ``device`` changes — this
    is the critical invariant that prevents cached scenes from being
    silently re-processed when flipping CPU↔GPU.
  * CLI parses ``--dispatcher`` / ``--flex-slots-per-gpu`` and passes
    them into ``Project.run`` (smoke via a mocked Project).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.project import DispatcherSettings, ProjectConfig


# --------------------------------------------------------------------- #
# DispatcherSettings model
# --------------------------------------------------------------------- #

class TestDispatcherSettings:
    def test_defaults(self):
        s = DispatcherSettings()
        assert s.mode == "static"
        assert s.flex_slots_per_gpu == 1

    def test_rejects_bad_mode(self):
        with pytest.raises(Exception):
            DispatcherSettings(mode="turbo")

    def test_rejects_zero_slots(self):
        with pytest.raises(Exception):
            DispatcherSettings(flex_slots_per_gpu=0)

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            DispatcherSettings(mode="smart", unknown_field=True)

    def test_yaml_roundtrip(self):
        s = DispatcherSettings(mode="smart", flex_slots_per_gpu=3)
        payload = yaml.dump(s.model_dump(mode="json"))
        loaded = DispatcherSettings(**yaml.safe_load(payload))
        assert loaded.mode == "smart"
        assert loaded.flex_slots_per_gpu == 3


class TestProjectConfigWithDispatcher:
    def test_default_has_no_dispatcher(self):
        cfg = ProjectConfig(name="test")
        assert cfg.dispatcher is None

    def test_omitting_block_loads_cleanly(self, tmp_path: Path):
        """Legacy project.yaml (no dispatcher block) must load."""
        yaml_path = tmp_path / "project.yaml"
        yaml_path.write_text("name: legacy-project\npipeline: full-workflow\n")
        cfg = ProjectConfig.from_yaml(yaml_path)
        assert cfg.dispatcher is None

    def test_dispatcher_block_roundtrip(self, tmp_path: Path):
        yaml_path = tmp_path / "project.yaml"
        yaml_path.write_text(
            "name: smart-project\n"
            "pipeline: full-workflow\n"
            "dispatcher:\n"
            "  mode: smart\n"
            "  flex_slots_per_gpu: 2\n"
        )
        cfg = ProjectConfig.from_yaml(yaml_path)
        assert cfg.dispatcher is not None
        assert cfg.dispatcher.mode == "smart"
        assert cfg.dispatcher.flex_slots_per_gpu == 2


# --------------------------------------------------------------------- #
# Precedence: CLI > project.yaml > defaults
# --------------------------------------------------------------------- #

class TestResolveDispatcherSettings:
    """``Project.resolve_dispatcher_settings`` is the merge point.

    We don't need a fully-materialised Project on disk — a bare stand-in
    with a ``config`` attribute exposing ``.dispatcher`` is enough.
    """

    class _Stub:
        def __init__(self, yaml_cfg):
            self.config = MagicMock()
            self.config.dispatcher = yaml_cfg
        # bind the real method
        from clouds_decoded.project import Project
        resolve_dispatcher_settings = Project.resolve_dispatcher_settings

    def test_defaults_when_everything_unset(self):
        s = self._Stub(yaml_cfg=None)
        mode, slots = s.resolve_dispatcher_settings(cli_mode=None, cli_slots=None)
        assert (mode, slots) == ("static", 1)

    def test_yaml_wins_over_defaults(self):
        s = self._Stub(yaml_cfg=DispatcherSettings(mode="smart", flex_slots_per_gpu=3))
        mode, slots = s.resolve_dispatcher_settings(cli_mode=None, cli_slots=None)
        assert (mode, slots) == ("smart", 3)

    def test_cli_mode_overrides_yaml(self):
        s = self._Stub(yaml_cfg=DispatcherSettings(mode="smart"))
        mode, _ = s.resolve_dispatcher_settings(cli_mode="static", cli_slots=None)
        assert mode == "static"

    def test_cli_slots_override_yaml(self):
        s = self._Stub(yaml_cfg=DispatcherSettings(mode="smart", flex_slots_per_gpu=4))
        _, slots = s.resolve_dispatcher_settings(cli_mode=None, cli_slots=1)
        assert slots == 1

    def test_cli_overrides_cascade_independently(self):
        """Setting only one CLI flag should not clobber the other from YAML."""
        s = self._Stub(yaml_cfg=DispatcherSettings(mode="smart", flex_slots_per_gpu=4))
        mode, slots = s.resolve_dispatcher_settings(cli_mode="static", cli_slots=None)
        assert (mode, slots) == ("static", 4)


# --------------------------------------------------------------------- #
# Semantic dump / config hash stability
# --------------------------------------------------------------------- #

class TestSemanticDump:
    def test_device_excluded_from_dump(self):
        from clouds_decoded.modules.refocus import RefocusConfig
        cpu_cfg = RefocusConfig(device=None)
        gpu_cfg = RefocusConfig(device="cuda:0")
        assert "device" not in BaseProcessorConfig._semantic_dump(cpu_cfg)
        assert "device" not in BaseProcessorConfig._semantic_dump(gpu_cfg)

    def test_device_toggle_does_not_change_dump(self):
        from clouds_decoded.modules.refocus import RefocusConfig
        cpu_cfg = RefocusConfig(device=None)
        gpu_cfg = RefocusConfig(device="cuda:0")
        assert BaseProcessorConfig._semantic_dump(cpu_cfg) == BaseProcessorConfig._semantic_dump(gpu_cfg)

    def test_non_device_change_does_change_dump(self):
        from clouds_decoded.modules.refocus import RefocusConfig
        a = RefocusConfig(device="cuda:0", interpolation_order=1)
        b = RefocusConfig(device="cuda:0", interpolation_order=3)
        assert BaseProcessorConfig._semantic_dump(a) != BaseProcessorConfig._semantic_dump(b)


class TestConfigHashStability:
    """Integration: run ``_config_hash`` through a real on-disk project."""

    def _project(self, tmp_path: Path):
        from clouds_decoded.project import Project
        return Project.init(str(tmp_path / "proj"), pipeline="full-workflow")

    def test_hash_stable_across_device_flip(self, tmp_path: Path):
        project = self._project(tmp_path)
        # Flip the refocus config's device and confirm the hash doesn't move.
        refocus_yaml = project.configs_dir / "refocus.yaml"
        raw = yaml.safe_load(refocus_yaml.read_text())
        raw["device"] = None
        refocus_yaml.write_text(yaml.dump(raw))
        hash_cpu = project._config_hash("refocus")

        raw["device"] = "cuda:0"
        refocus_yaml.write_text(yaml.dump(raw))
        hash_gpu = project._config_hash("refocus")

        assert hash_cpu == hash_gpu, (
            f"device flip changed the semantic hash ({hash_cpu} → {hash_gpu}); "
            "cached scenes would be re-processed on every CPU/GPU toggle."
        )

    def test_hash_changes_on_semantic_field(self, tmp_path: Path):
        project = self._project(tmp_path)
        refocus_yaml = project.configs_dir / "refocus.yaml"
        raw = yaml.safe_load(refocus_yaml.read_text())
        raw["interpolation_order"] = 1
        refocus_yaml.write_text(yaml.dump(raw))
        hash_order1 = project._config_hash("refocus")

        raw["interpolation_order"] = 3
        refocus_yaml.write_text(yaml.dump(raw))
        hash_order3 = project._config_hash("refocus")

        assert hash_order1 != hash_order3


# --------------------------------------------------------------------- #
# CLI flag parsing
# --------------------------------------------------------------------- #

class TestCLIFlagParsing:
    """Verify --dispatcher / --flex-slots-per-gpu thread through to run()."""

    def _invoke(self, argv):
        """Invoke project_run with a mocked Project.load so we don't need a real project on disk."""
        from typer.testing import CliRunner
        from clouds_decoded.cli.entry import app

        runner = CliRunner()
        with patch("clouds_decoded.project.Project.load") as mock_load:
            fake_project = MagicMock()
            fake_project.resolve_dispatcher_settings.return_value = ("static", 1)
            mock_load.return_value = fake_project
            return runner.invoke(app, argv), fake_project

    def test_dispatcher_smart_propagates(self, tmp_path: Path):
        (tmp_path / "dummy").mkdir()
        result, proj = self._invoke([
            "project", "run", str(tmp_path / "dummy"),
            "--dispatcher", "smart", "--flex-slots-per-gpu", "3",
        ])
        # We expect run() to have been called with the passed values (post-resolve).
        proj.resolve_dispatcher_settings.assert_called_once_with(
            cli_mode="smart", cli_slots=3,
        )

    def test_dispatcher_invalid_mode_rejected(self, tmp_path: Path):
        (tmp_path / "dummy").mkdir()
        result, _ = self._invoke([
            "project", "run", str(tmp_path / "dummy"),
            "--dispatcher", "turbo",
        ])
        assert result.exit_code != 0

    def test_no_flags_keeps_static_default(self, tmp_path: Path):
        (tmp_path / "dummy").mkdir()
        result, proj = self._invoke(["project", "run", str(tmp_path / "dummy")])
        proj.resolve_dispatcher_settings.assert_called_once_with(
            cli_mode=None, cli_slots=None,
        )
