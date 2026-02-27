"""Step-specific stats for the cloud_properties step.

Band names (tau, ice_liq_ratio, r_eff_liq, r_eff_ice) are already stored in
METADATA_TAG by CloudPropertiesData.write(), so the generic ``percentiles``
function handles them correctly without any override needed.
"""
from __future__ import annotations

# Re-export generic percentiles — no override needed for cloud_properties.
from ._generic import percentiles  # noqa: F401
