"""CLI command tests."""
import pytest
from typer.testing import CliRunner
from clouds_decoded.cli.entry import app

runner = CliRunner()


def test_cli_help():
    """Test that main help text works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cloud-height" in result.stdout or "cloud_height" in result.stdout


def test_cloud_height_help():
    """Test cloud-height command help."""
    result = runner.invoke(app, ["cloud-height", "--help"])
    assert result.exit_code == 0
    # Should mention scene path argument
    assert "scene" in result.stdout.lower() or "path" in result.stdout.lower()


def test_cloud_mask_help():
    """Test cloud-mask command help."""
    result = runner.invoke(app, ["cloud-mask", "--help"])
    assert result.exit_code == 0


def test_full_workflow_help():
    """Test full-workflow command help."""
    result = runner.invoke(app, ["full-workflow", "--help"])
    assert result.exit_code == 0


def test_cloud_height_missing_scene():
    """Test that cloud-height fails gracefully without scene."""
    result = runner.invoke(app, ["cloud-height", "/nonexistent/scene.SAFE"])
    # Should fail but not crash
    assert result.exit_code != 0


if __name__ == "__main__":
    print("=== Running CLI Tests ===\n")

    print("Testing main help...")
    test_cli_help()
    print("✓ Main help works")

    print("Testing cloud-height help...")
    test_cloud_height_help()
    print("✓ cloud-height help works")

    print("Testing cloud-mask help...")
    test_cloud_mask_help()
    print("✓ cloud-mask help works")

    print("Testing full-workflow help...")
    test_full_workflow_help()
    print("✓ full-workflow help works")

    print("Testing missing scene error...")
    test_cloud_height_missing_scene()
    print("✓ Missing scene handled gracefully")

    print("\n=== ✓ All CLI Tests Passed ===")
