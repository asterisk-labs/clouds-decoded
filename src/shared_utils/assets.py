"""Asset path resolution and downloading for clouds-decoded.

Large binary assets (model weights, GEBCO) are stored in a user-configured
directory.  Resolution priority:

    1. CLOUDS_DECODED_ASSETS_DIR env var
    2. Value in config.toml  (user_config_dir / "clouds-decoded" / config.toml)
    3. platformdirs.user_data_dir("clouds-decoded", appauthor=False)

Run ``clouds-decoded setup`` once to choose the location, then
``clouds-decoded download <key>`` to fetch individual assets.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import platformdirs

# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(platformdirs.user_config_dir("clouds-decoded", appauthor=False))
_CONFIG_FILE = _CONFIG_DIR / "config.toml"
_CONFIG_KEY = "assets_dir"


def _read_config_assets_dir() -> Optional[str]:
    """Return the assets_dir value from config.toml, or None if absent."""
    if not _CONFIG_FILE.exists():
        return None
    # Minimal TOML parse — strip inline comments before unquoting.
    for line in _CONFIG_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith(_CONFIG_KEY):
            _, _, value = line.partition("=")
            value = value.split("#")[0].strip().strip('"').strip("'")
            return value or None
    return None


def _write_config_assets_dir(path: str) -> None:
    """Write (or overwrite) the assets_dir key in config.toml."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve any other keys that might already exist.
    existing_lines: list[str] = []
    if _CONFIG_FILE.exists():
        existing_lines = [
            l for l in _CONFIG_FILE.read_text().splitlines()
            if not l.strip().startswith(_CONFIG_KEY)
        ]
    existing_lines.insert(0, f'{_CONFIG_KEY} = "{path}"')
    _CONFIG_FILE.write_text("\n".join(existing_lines) + "\n")


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def get_assets_dir() -> Path:
    """Return (and create) the assets directory.

    Priority: env var → config.toml → platformdirs default.
    """
    env = os.environ.get("CLOUDS_DECODED_ASSETS_DIR")
    if env:
        directory = Path(env)
    else:
        from_config = _read_config_assets_dir()
        if from_config:
            directory = Path(from_config)
        else:
            directory = Path(platformdirs.user_data_dir("clouds-decoded", appauthor=False))

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_asset(relative_path: str) -> Path:
    """Return ``get_assets_dir() / relative_path``.

    Does *not* check whether the file exists — callers handle that.
    """
    return get_assets_dir() / relative_path


# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------

@dataclass
class Asset:
    key: str
    relative_path: str
    url: str
    size_hint: str
    description: str
    is_zip: bool = False  # True if the download is a zip that needs extracting


KNOWN_ASSETS: dict[str, Asset] = {
    "cloud_mask": Asset(
        key="cloud_mask",
        relative_path="models/cloud_mask/default.pt",
        url="https://huggingface.co/asterisk-labs/clouds-decoded/resolve/main/cloud_mask/weights.pt",
        size_hint="~105 MB",
        description="Cloud mask SegFormer model weights",
    ),
    "emulator": Asset(
        key="emulator",
        relative_path="models/cloud_height_emulator/default.pth",
        url="https://huggingface.co/asterisk-labs/clouds-decoded/resolve/main/cloud_height_emulator/weights.pth",
        size_hint="~100 MB",
        description="Height emulator model weights",
    ),
    "refl2prop": Asset(
        key="refl2prop",
        relative_path="models/refl2prop/default.pth",
        url="https://huggingface.co/asterisk-labs/clouds-decoded/resolve/main/refl2prop/weights.pth",
        size_hint="~1 MB",
        description="Refl2prop model weights",
    ),
    "albedo_datadriven": Asset(
        key="albedo_datadriven",
        relative_path="models/albedo_datadriven/default.pth",
        url="https://huggingface.co/asterisk-labs/clouds-decoded/resolve/main/albedo_datadriven/weights.pth",
        size_hint="~1 MB",
        description="Data-driven albedo model weights",
    ),
    "gebco": Asset(
        key="gebco",
        relative_path="data/GEBCO_2024.nc",
        url="https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/zip/",
        size_hint="~2.7 GB",
        description="GEBCO 2024 bathymetry",
        is_zip=True,
    ),
}


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _progress_hook(t):  # type: ignore[no-untyped-def]
    """Return a urllib reporthook that updates a tqdm progress bar *t*."""
    last_b = [0]

    def inner(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and t.total is None:
            t.total = total_size
        t.update((count - last_b[0]) * block_size)
        last_b[0] = count

    return inner


def download_asset(key: str, force: bool = False) -> Path:
    """Download *key* to ``get_assets_dir() / asset.relative_path``.

    Skips download if the file already exists and *force* is False.
    For zip assets (GEBCO) the archive is extracted then deleted.

    Returns the final resolved path.
    """
    from tqdm import tqdm  # lazy — not everyone runs downloads

    asset = KNOWN_ASSETS[key]
    dest = get_asset(asset.relative_path)

    if dest.exists() and not force:
        print(f"{asset.description} already present at {dest}. Use --force to re-download.")
        return dest

    if not asset.url:
        raise ValueError(
            f"No download URL configured for '{key}'. "
            "Set one in KNOWN_ASSETS or supply the file manually."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    if asset.is_zip:
        zip_path = dest.parent / f"{dest.stem}.zip"
        print(f"Downloading {asset.description} ({asset.size_hint}) → {zip_path}")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=asset.key) as t:
            urllib.request.urlretrieve(asset.url, zip_path, reporthook=_progress_hook(t))

        print(f"Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only the target .nc file (GEBCO zips contain one file)
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not members:
                members = zf.namelist()
            for member in members:
                zf.extract(member, dest.parent)
                extracted = dest.parent / member
                if extracted != dest:
                    extracted.rename(dest)
        zip_path.unlink(missing_ok=True)
    else:
        print(f"Downloading {asset.description} ({asset.size_hint}) → {dest}")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=asset.key) as t:
            urllib.request.urlretrieve(asset.url, dest, reporthook=_progress_hook(t))

    print(f"Done — saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Require-or-raise helper
# ---------------------------------------------------------------------------

def require_asset(key: str) -> Path:
    """Return the path to *key* if it exists.

    Raises ``FileNotFoundError`` with an actionable message otherwise.
    """
    asset = KNOWN_ASSETS[key]
    path = get_asset(asset.relative_path)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{asset.description} not found.\n"
        f"Run:\n"
        f"  clouds-decoded download {key}\n"
        f"or set CLOUDS_DECODED_ASSETS_DIR to a directory containing "
        f"{asset.relative_path}"
    )
