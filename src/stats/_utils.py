"""Shared utilities for stats functions."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from clouds_decoded.project import SceneManifest


def _get_output_path(manifest: "SceneManifest", step_name: str) -> Optional[Path]:
    """Return the output file path for a step if it exists and is complete.

    Args:
        manifest: The scene manifest to inspect.
        step_name: The step name to look up.

    Returns:
        A ``Path`` if the step is complete and its output file exists, else ``None``.
    """
    step = manifest.steps.get(step_name)
    if step is None or step.status != "completed" or not step.output_file:
        return None
    p = Path(step.output_file)
    return p if p.exists() else None
