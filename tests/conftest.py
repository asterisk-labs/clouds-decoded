"""pytest configuration shared across all test modules."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include slow tests (e.g. albedo on real scenes).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselected by default; run with --run-slow)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # When --run-slow is given, remove the deselection added by addopts.
        config.option.markexpr = ""
