"""Step-specific stats for the albedo step.

Band names (B01–B12) are already stored in METADATA_TAG by AlbedoData.write(),
so the generic ``mean`` function handles them correctly without any override.
"""
from __future__ import annotations

# Re-export generic mean — no override needed for albedo.
from ._generic import mean  # noqa: F401
