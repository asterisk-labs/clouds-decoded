"""HTML report generation for clouds-decoded project outputs."""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Categorical slots for tile identity, in fixed order. Four hues come from the
#: bundled logos -- sky from clouds-decoded.webp (#1890f0), yellow/indigo/green
#: from asterisk-labs.svg (#f7cc09, #492ae8, #c4ffc2) -- restepped per surface;
#: orange, magenta, teal and violet fill the wheel.
#:
#: Checked against the colour-vision gates rather than chosen by eye. Worst
#: adjacent pair: CVD dE 11.2 (deutan) / tritan 8.3, normal-vision 20.8, every
#: slot >= 3:1 on its own surface, in BOTH modes (OKLab x100).
#:
#: Do NOT reorder or extend. The ORDERING is the safety mechanism -- it is what
#: holds adjacent pairs apart, and most orderings of these same eight hues fail.
#: A ninth hue cannot clear the gates at all; extra tiles fold to _TILE_OTHER.
_PALETTE_DARK = [
    '#0689e9', '#e26a2b', '#6159ff', '#008400',
    '#da5592', '#b18e00', '#0095a2', '#934cc8',
]
_PALETTE_LIGHT = [
    '#0083e2', '#db6423', '#5c4dff', '#007d00',
    '#d34e8c', '#a98900', '#008e9a', '#8d46c1',
]
#: Neutral for tiles past the palette. Grey is deliberate: it reads as
#: "unassigned", where a recycled hue reads as a specific tile.
_TILE_OTHER = {'dark': '#6b7280', 'light': '#8a9099'}

_STAT_LABELS = {
    'cloud_frac':          'Cloud fraction',
    'tau__mean':           'τ mean',
    'tau__p050':           'τ median',
    'r_eff_liq__p050':     'r_eff liq p50',
    'r_eff_ice__p050':     'r_eff ice p50',
    'ice_liq_ratio__mean': 'Ice/liq ratio',
}


#: Logos shipped with the package and embedded as data URIs in every report.
#: Pass ``brand_logo=""`` / ``project_logo=""`` to generate_report to omit them.
_ASSETS = Path(__file__).parent / "assets"
_LOGO_MIME = {".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp"}


def _asset_uri(filename: str) -> str:
    """Base64 data URI for a bundled asset, or '' if it is not installed."""
    path = _ASSETS / filename
    if not path.exists():
        logger.debug("Report asset missing, omitting: %s", path)
        return ""
    mime = _LOGO_MIME.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _asset_uri_light(filename: str) -> str:
    """Data URI for the light-mode variant of ``filename``, falling back to the
    default when there isn't one.

    Convention: ``asterisk-labs.svg`` -> ``asterisk-labs-light.svg``, alongside
    it in ``assets/``. Drop the file in and it is picked up; nothing else to
    register, since pyproject already ships ``assets/*``.

    A logo is a brand asset, so the light version is a file someone supplies
    rather than something this module derives by recolouring.
    """
    path = _ASSETS / filename
    light = path.with_name(f"{path.stem}-light{path.suffix}")
    if light.exists():
        return _asset_uri(light.name)
    return _asset_uri(filename)


def _extract_date(scene_id: str) -> Optional[str]:
    m = re.search(r"_(\d{4})(\d{2})(\d{2})T", scene_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _tile_colors(scenes: list[dict], mode: str = 'dark') -> dict[str, str]:
    """Assign one categorical slot per tile, in fixed order, never cycling.

    Past the palette every tile gets the same neutral. Recycling hues instead
    (the old ``_PALETTE[i % len]``) silently gave different tiles an identical
    colour on both the map dots and the chart lines -- harmless at 12 tiles with
    a 16-entry list, wrong at the 51 of a full campaign. Colour cannot carry 51
    identities, so the report stops pretending it can and the tile checkboxes
    become the way to isolate one.
    """
    palette = _PALETTE_DARK if mode == 'dark' else _PALETTE_LIGHT
    other = _TILE_OTHER[mode]
    tiles = sorted({s['tile_id'] for s in scenes if s.get('tile_id')})
    return {t: (palette[i] if i < len(palette) else other)
            for i, t in enumerate(tiles)}


def _tile_color_map(scenes: list[dict]) -> dict[str, dict[str, str]]:
    """Both modes at once, so the page can swap palettes without regenerating."""
    return {m: _tile_colors(scenes, m) for m in ('dark', 'light')}


def _load_project_data(project_dir: Path, db_path: Optional[Path] = None):
    import duckdb
    db_path = db_path or (project_dir / "project.db")
    if not db_path.exists():
        raise FileNotFoundError(f"No project.db found at {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}

    df = conn.execute("""
        SELECT r.run_id, r.scene_id,
               sm.sensing_time, sm.tile_id, sm.satellite,
               sm.lat_center, sm.lon_center
        FROM runs r
        LEFT JOIN scene_metadata sm ON r.scene_id = sm.scene_id
        WHERE r.status = 'done'
        ORDER BY sm.sensing_time
    """).df()

    if 'stats_cloud_mask' in tables:
        df = df.merge(conn.execute("SELECT * FROM stats_cloud_mask").df(),
                      on='run_id', how='left')
    if 'stats_cloud_properties' in tables:
        df = df.merge(conn.execute("SELECT * FROM stats_cloud_properties").df(),
                      on='run_id', how='left')
    conn.close()
    return df


def _available_layers(figures_dir: Path, scene_id: str) -> list[str]:
    d = figures_dir / scene_id
    if not d.exists():
        return []
    # overview first, then alphabetical
    files = sorted(d.glob("*.png"))
    overview = [f for f in files if f.name == 'overview.png']
    rest = [f for f in files if f.name != 'overview.png']
    return [f.name for f in overview + rest]


def _build_scenes(df, figures_dir: Path, path_prefix: str = '', compact: bool = False) -> list[dict]:
    """Build scene dicts. ``path_prefix`` is prepended to each layer path so a
    combined report served from a parent dir can reference per-project figures,
    e.g. ``path_prefix='Rama_examples/'`` -> ``Rama_examples/figures/<sid>/...``."""
    skip = {'run_id', 'scene_id', 'sensing_time', 'tile_id', 'satellite',
            'lat_center', 'lon_center'}
    scenes = []
    for _, row in df.iterrows():
        sid = row.scene_id
        stats = {}
        for col in df.columns:
            if col in skip:
                continue
            v = row[col]
            if v is None:
                continue
            try:
                if math.isnan(float(v)):
                    continue
            except (TypeError, ValueError):
                pass
            stats[col] = round(float(v), 4) if isinstance(v, float) else v

        date = str(row.sensing_time)[:10] if row.sensing_time else _extract_date(sid)
        scenes.append({
            'scene_id': sid,
            'date': date,
            'tile_id': str(row.tile_id) if row.tile_id else '',
            'satellite': str(row.satellite) if row.satellite else '',
            'lat': float(row.lat_center) if row.lat_center is not None else None,
            'lon': float(row.lon_center) if row.lon_center is not None else None,
            'layers': (_available_layers(figures_dir, sid) if compact
                       else [f'{path_prefix}figures/{sid}/{l}' for l in _available_layers(figures_dir, sid)]),
            'stats': stats,
        })
    return scenes


def _generate_map(scenes: list[dict], tile_colors: dict[str, str],
                  width_px: int = 500, height_px: int = 380,
                  dpi: int = 96, draw_markers: bool = True,
                  feature_scale: str = '50m'
                  ) -> tuple[Optional[str], Optional[list[dict]]]:
    """Return (base64_png, list_of_pixel_coords) for the map.

    ``draw_markers=False`` renders the basemap only (no dots baked into the PNG)
    so an interactive canvas overlay can fully control dot visibility — used by
    the combined report where tiles are toggled on/off."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        lons = [s['lon'] for s in scenes if s.get('lon') is not None]
        lats = [s['lat'] for s in scenes if s.get('lat') is not None]
        if not lons:
            return None, None

        # Antimeridian-aware framing: find the widest empty longitude arc
        # (largest-gap method) and centre the map on the data. Without this, tiles
        # straddling the date line (e.g. Japan +140.5° and Fiji -170.5°) yield a
        # bogus ~311° span and get cut off at the map edges.
        slon = sorted(lons)
        gaps = [(slon[i + 1] - slon[i], slon[i]) for i in range(len(slon) - 1)]
        gaps.append((360.0 - (slon[-1] - slon[0]), slon[-1]))  # wrap-around gap
        max_gap, gap_start = max(gaps, key=lambda g: g[0])
        gap_center = gap_start + max_gap / 2.0          # middle of the empty arc
        data_center = gap_center + 180.0                # data sits opposite the gap
        central_lon = ((data_center + 180.0) % 360.0) - 180.0  # normalise to [-180,180)
        # express longitudes in the recentred projection frame
        shifted = [((lon - central_lon + 180.0) % 360.0) - 180.0 for lon in lons]
        lon_span = max(shifted) - min(shifted)
        span = max(lon_span, max(lats) - min(lats), 1.0)
        pad = max(2.0, span * 0.25)
        lat_lo = max(min(lats) - pad, -90.0)
        lat_hi = min(max(lats) + pad, 90.0)
        if lon_span > 175.0:
            # near-global spread: use the full recentred world so edge tiles keep
            # a margin instead of being clipped at the map border
            extent = [-180.0, 180.0, lat_lo, lat_hi]
        else:
            extent = [min(shifted) - pad, max(shifted) + pad, lat_lo, lat_hi]

        proj = ccrs.PlateCarree(central_longitude=central_lon)
        fig = plt.figure(figsize=(width_px / dpi, height_px / dpi),
                         facecolor='#1a1a2e', dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1], projection=proj)
        ax.set_facecolor('#0d1b2a')
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale(feature_scale), facecolor='#0d1b2a')
        ax.add_feature(cfeature.LAND.with_scale(feature_scale), facecolor='#1e2d1e')
        ax.add_feature(cfeature.COASTLINE.with_scale(feature_scale),
                       linewidth=0.5, edgecolor='#556655')
        ax.add_feature(cfeature.BORDERS.with_scale(feature_scale),
                       linewidth=0.3, edgecolor='#445544')

        # Plot markers (optional — the interactive canvas draws them for the
        # combined report).
        if draw_markers:
            for s in scenes:
                if s.get('lon') is None:
                    continue
                color = tile_colors.get(s['tile_id'], '#aaaaaa')
                ax.plot(s['lon'], s['lat'], 'o', color=color, markersize=7,
                        markeredgecolor='white', markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=5)

        # Finalise the layout BEFORE reading pixel positions. With aspect='equal'
        # cartopy letterboxes the map inside the figure; the geo→display transform
        # only reflects that final layout after a draw. Computing pixel_coords
        # earlier misplaces the dots (e.g. UK sitting north of Britain).
        fig.canvas.draw()
        trans = ccrs.PlateCarree()._as_mpl_transform(ax)
        pixel_coords = []
        for s in scenes:
            if s.get('lon') is None:
                pixel_coords.append(None)
                continue
            disp = trans.transform((s['lon'], s['lat']))
            # Flip Y: matplotlib origin is bottom-left, HTML canvas is top-left
            pixel_coords.append({'x': round(float(disp[0]), 1),
                                 'y': round(float(height_px - disp[1]), 1)})

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi,
                    facecolor='#1a1a2e', bbox_inches=None)
        plt.close(fig)
        png_b64 = base64.b64encode(buf.getvalue()).decode()
        return png_b64, pixel_coords

    except Exception as e:
        logger.warning(f"Map generation failed: {e}")
        return None, None


def _render_html(scenes: list[dict], tile_colors: dict[str, str],
                 map_png: Optional[str], map_points: Optional[list],
                 project_name: str, regions: Optional[dict] = None,
                 fig_base: str = '', brand_logo: str = '', project_url: str = '',
                 project_logo: str = '',
                 tile_colors_modes: Optional[dict] = None,
                 brand_logo_light: str = '', project_logo_light: str = '') -> str:

    regions = regions or {}
    # Optionally hyperlink the first title segment (before the first " · ") to project_url.
    if project_url:
        _p = project_name.split(' · ', 1)
        title_html = (f'<a href="{project_url}" target="_blank" rel="noopener">{_p[0]}</a>'
                      + (f' · {_p[1]}' if len(_p) > 1 else ''))
    else:
        title_html = project_name
    data_json = json.dumps(scenes)
    # Ship both palettes so the viewer can switch theme without regenerating.
    # ``tile_colors`` stays the dark set (what the baked map PNG was drawn with).
    tile_colors_modes_json = json.dumps(
        tile_colors_modes or {'dark': tile_colors, 'light': tile_colors})
    map_points_json = json.dumps(map_points or [])
    fig_base_json = json.dumps(fig_base)   # if set, layers are basenames → URL = fig_base+scene_id+'/'+name

    # One tile at a time: radio, not checkbox. The chart plots the tile you are
    # looking at and nothing else, so selecting a tile here, clicking its dot on
    # the map, and stepping through its scenes are all the same action.
    def _tile_label(t):
        rg = regions.get(t, '')
        return f'{t} <span class="rgn">({rg})</span>' if rg else t
    _sorted_tiles = sorted(tile_colors.items())
    _default_tile = _sorted_tiles[0][0] if _sorted_tiles else None
    tile_tabs_html = ''.join(
        f'<label class="tchk" data-tile="{t}">'
        f'<input type="radio" name="tile-select" data-tile="{t}"'
        f'{" checked" if t == _default_tile else ""} '
        f'title="plot this tile and browse its scenes">'
        f'<span class="tab-dot" data-tile="{t}" style="background:{c}"></span>'
        f'<span class="tname" data-tile="{t}">{_tile_label(t)}</span></label>'
        for t, c in _sorted_tiles
    )

    map_html = ''
    if map_png:
        map_html = f"""
        <div id="map-wrap" title="scroll to zoom · drag to pan · double-click to reset">
          <div id="map-inner">
            <img id="map-bg" src="data:image/png;base64,{map_png}" alt="scene map">
          </div>
          <canvas id="map-canvas"></canvas>
          <span class="info-i map-i" title="Each dot is one processed scene, coloured by tile; every scene is always shown. Scroll to zoom, drag to pan, double-click to reset. Click a dot to open that scene in the browser below.">i</span>
        </div>"""
    else:
        map_html = '<div id="map-wrap" style="display:none"></div>'

    # Both variants ride along on the element; applyTheme() swaps src. When no
    # -light file exists the two are identical and the swap is a no-op.
    logo_html = (f'<img id="logo" src="{brand_logo}" alt="asterisk labs" '
                 f'data-dark="{brand_logo}" data-light="{brand_logo_light or brand_logo}">'
                 if brand_logo else '')
    logo2_html = (f'<img id="logo2" src="{project_logo}" alt="clouds decoded" '
                  f'data-dark="{project_logo}" data-light="{project_logo_light or project_logo}">'
                  if project_logo else '')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{project_name} — clouds-decoded report</title>
<style>
/* ---- Theme tokens -------------------------------------------------------
   Dark is the default so an existing report keeps its look; [data-theme=light]
   wins both ways and is remembered per viewer. Every colour below is a token so
   the two palettes swap in one place instead of being spread through the file. */
:root {{
  color-scheme: dark;
  --bg: #0f1117; --surface-1: #161a24; --surface-2: #1d2231; --surface-3: #272d40;
  --border: #2a3142; --border-strong: #3b4459;
  --text-1: #e7eaf1; --text-2: #98a2b6; --text-3: #6a7488;
  --accent: #4aa3f0; --accent-ink: #cfe4fb; --accent-weak: #17324d;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 14px rgba(0,0,0,.28);
  --thumb-border: #2b3243; --scrim: rgba(6,8,12,.94);
}}
:root[data-theme="light"] {{
  color-scheme: light;
  --bg: #f6f7fa; --surface-1: #ffffff; --surface-2: #f0f2f7; --surface-3: #e3e7ef;
  --border: #d9dee8; --border-strong: #b6bfcf;
  --text-1: #12161f; --text-2: #545f75; --text-3: #78829a;
  --accent: #0b62c4; --accent-ink: #0a4c99; --accent-weak: #dcebfc;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 4px 14px rgba(16,24,40,.08);
  --thumb-border: #dbe1ea; --scrim: rgba(20,24,32,.85);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        background: var(--bg); color: var(--text-1); -webkit-font-smoothing: antialiased; }}
/* ---- Header: one row, logos inline, title left ---- */
#header {{ position: relative; padding: 9px 14px; background: var(--surface-1);
           border-bottom: 1px solid var(--border); display: flex; align-items: center;
           gap: 12px; flex-shrink: 0; }}
#header h1 {{ font-size: 1.02rem; font-weight: 650; letter-spacing: -0.01em; color: var(--text-1); white-space: nowrap; }}
#brand {{ display: flex; align-items: center; gap: 10px; min-width: 0; }}
#header-controls {{ display: flex; align-items: center; gap: 8px; margin-left: auto; }}
#daterange {{ display: flex; align-items: center; gap: 5px; font-size: 0.75rem; color: var(--text-2); }}
.info-i {{ display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%; border: 1px solid var(--border-strong); color: var(--text-2); font: italic 700 0.62rem Georgia, serif; cursor: help; margin-left: 6px; flex-shrink: 0; background: var(--surface-2); }}
.info-i:hover {{ background: var(--accent-weak); color: var(--accent-ink); border-color: var(--accent); }}
#daterange .dr-label {{ text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.63rem; color: var(--text-3); font-weight: 600; }}
#daterange .dr-sep {{ color: var(--text-3); }}
#daterange input[type=date] {{ background: var(--surface-2); color: var(--text-1); border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-size: 0.73rem; font-family: inherit; }}
#daterange input[type=date]:hover {{ border-color: var(--accent); }}
#date-reset, .tbtn {{ background: var(--surface-2); color: var(--text-2); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 0.72rem; font-family: inherit; cursor: pointer; white-space: nowrap; display: inline-flex; align-items: center; gap: 5px; }}
#date-reset:hover, .tbtn:hover {{ background: var(--surface-3); color: var(--text-1); border-color: var(--border-strong); }}
.tbtn.open {{ background: var(--accent-weak); color: var(--accent-ink); border-color: var(--accent); }}
.tbtn .caret {{ font-size: 0.6rem; opacity: .7; }}
.tbtn .cnt {{ font-variant-numeric: tabular-nums; color: var(--text-3); }}
#header .meta {{ font-size: 0.8rem; color: var(--text-2); }}
#logo, #logo2 {{ height: 26px; width: auto; flex-shrink: 0; }}
#header h1 a {{ color: inherit; text-decoration: none; }}
#header h1 a:hover {{ color: var(--accent); }}
#topbar {{ position: sticky; top: 0; z-index: 30; background: var(--bg); flex-shrink: 0; box-shadow: var(--shadow); }}
.map-i, .chart-i {{ position: absolute; top: 8px; right: 8px; z-index: 6; margin: 0; background: var(--surface-1); }}
.chart-i {{ right: 12px; }}
.row-h {{ width: 46px; background: var(--surface-2); color: var(--text-2); border: 1px solid var(--border); border-radius: 5px; font-size: 0.66rem; font-family: inherit; padding: 2px 4px; margin-left: 5px; }}
.row-h:hover {{ border-color: var(--accent); }}
/* ---- One toolbar; tile / variable / layer pickers live in popovers ---- */
#toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 6px 14px;
            background: var(--surface-1); border-bottom: 1px solid var(--border); flex-shrink: 0; }}
.bar-label {{ font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-3); font-weight: 600; margin-right: 4px; }}
.popwrap {{ position: relative; }}
.popover {{ display: none; position: absolute; top: calc(100% + 6px); left: 0; z-index: 40;
            background: var(--surface-1); border: 1px solid var(--border-strong); border-radius: 10px;
            box-shadow: var(--shadow); padding: 8px; min-width: 240px; max-width: min(560px, 90vw);
            max-height: 60vh; overflow: auto; }}
.popover.open {{ display: flex; flex-wrap: wrap; gap: 4px; align-content: start; }}
.pop-actions {{ display: flex; gap: 6px; width: 100%; padding-bottom: 6px; margin-bottom: 4px; border-bottom: 1px solid var(--border); }}
.pop-actions button {{ background: none; border: none; color: var(--accent); font: inherit; font-size: 0.72rem; cursor: pointer; padding: 0 2px; }}
.pop-actions button:hover {{ text-decoration: underline; }}
.tchk {{ padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface-2); color: var(--text-1); font-size: 0.76rem; white-space: nowrap; display: flex; align-items: center; gap: 6px; user-select: none; cursor: pointer; }}
.tchk:hover {{ border-color: var(--border-strong); background: var(--surface-3); }}
.tchk input {{ margin: 0; cursor: pointer; accent-color: var(--accent); }}
.tchk .tname {{ cursor: pointer; }}
.tchk.focused {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; background: var(--accent-weak); }}
.tchk.focused .tname {{ color: var(--accent-ink); }}
.tchk .rgn {{ color: var(--text-3); }}
.tab-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 2px var(--surface-2); }}
/* ---- Map + chart ---- */
#top-panels {{ display: flex; gap: 10px; padding: 10px 14px; flex-shrink: 0; align-items: stretch; }}
#map-wrap {{ position: relative; flex-shrink: 0; height: 320px; overflow: hidden; border-radius: 10px; cursor: grab; border: 1px solid var(--border); background: var(--surface-1); }}
#map-wrap.grabbing {{ cursor: grabbing; }}
/* #map-inner needs a resolved height, otherwise `height:100%` on #map-bg below
   has an auto-height parent to resolve against, silently falls back to auto, and
   the basemap renders at its natural size while the dots are still positioned
   with clientHeight/naturalHeight -- map and dots then disagree by that ratio. */
#map-inner {{ position: relative; height: 100%; transform-origin: 0 0; will-change: transform; }}
#map-bg {{ display: block; height: 100%; width: auto; }}
#map-canvas {{ position: absolute; top: 0; left: 0; pointer-events: auto; }}
#chart-wrap {{ flex: 1; min-width: 0; position: relative; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1); padding: 6px; }}
#chart-canvas {{ display: block; width: 100%; cursor: pointer; }}
/* ---- Scene browser ---- */
#scene-panel {{ display: flex; align-items: flex-start; gap: 12px; padding: 2px 14px 16px; }}
#nav-prev, #nav-next, #nav-play {{ flex-shrink: 0; width: 34px; height: 34px; font-size: 1.05rem; background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; color: var(--text-1); cursor: pointer; margin-top: 4px; }}
#nav-prev:hover, #nav-next:hover, #nav-play:hover {{ background: var(--surface-3); border-color: var(--border-strong); }}
#nav-play.playing {{ background: var(--accent-weak); border-color: var(--accent); color: var(--accent-ink); }}
#speed-select {{ background: var(--surface-2); color: var(--text-2); border: 1px solid var(--border); border-radius: 6px; font-size: 0.7rem; font-family: inherit; padding: 3px 4px; margin-top: 4px; width: 54px; }}
#scene-info {{ flex-shrink: 0; width: 208px; font-size: 0.78rem; }}
#si-date {{ font-size: 1.05rem; font-weight: 650; color: var(--text-1); letter-spacing: -0.01em; }}
#si-sub {{ font-size: 0.67rem; color: var(--text-3); margin-top: 2px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
/* Stats as aligned tiles: label above, value in tabular figures, so columns
   line up between scenes instead of reflowing as a sentence. */
#si-stats {{ margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
.stat {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 5px 8px; min-width: 0; }}
.stat .k {{ display: block; font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.055em; color: var(--text-3); font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.stat .v {{ display: block; font-size: 0.95rem; font-weight: 600; color: var(--text-1); font-variant-numeric: tabular-nums; line-height: 1.35; }}
/* ---- Layer grid, grouped by role ---- */
#layer-grid {{ flex: 1; align-self: stretch; display: flex; flex-direction: column; gap: 12px; min-width: 0; }}
.layer-group {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px; align-content: start; }}
.group-head {{ grid-column: 1 / -1; display: flex; align-items: center; gap: 8px; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-3); font-weight: 700; }}
.group-head::after {{ content: ""; flex: 1; height: 1px; background: var(--border); }}
.layer-thumb {{ text-align: center; min-width: 0; }}
.layer-thumb.overview {{ grid-column: 1 / -1; }}
.layer-thumb.overview img {{ max-width: 46%; margin: 0 auto; }}
.layer-thumb img {{ width: 100%; height: auto; display: block; border: 1px solid var(--thumb-border); border-radius: 8px; cursor: pointer; background: var(--surface-1); }}
.layer-thumb img:hover {{ border-color: var(--accent); }}
.layer-label {{ font-size: 0.73rem; color: var(--text-2); font-weight: 600; margin-top: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
#progress {{ font-size: 0.72rem; color: var(--text-3); margin-top: 8px; font-variant-numeric: tabular-nums; }}
/* ---- Lightbox ---- */
#lightbox {{ display: none; position: fixed; inset: 0; background: var(--scrim); z-index: 100; align-items: center; justify-content: center; }}
#lightbox.active {{ display: flex; }}
#lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 6px; }}
#lb-close {{ position: fixed; top: 14px; right: 18px; font-size: 1.8rem; color: var(--text-1); cursor: pointer; z-index: 101; }}
@media print {{
  #topbar {{ position: static; box-shadow: none; }}
  .popover, #nav-play, #speed-select {{ display: none !important; }}
}}
</style>
</head>
<body>
<div id="topbar">
  <div id="header">
    {logo_html}
    {logo2_html}
    <div id="brand">
      <h1>{title_html}</h1>
    </div>
    <div id="header-controls">
      <div id="daterange">
        <span class="dr-label">Time range</span>
        <input type="date" id="date-start" title="start date">
        <span class="dr-sep">→</span>
        <input type="date" id="date-end" title="end date">
        <button id="date-reset" title="reset to full range">Reset</button>
      </div>
      <button class="tbtn" id="theme-toggle" title="Switch between light and dark">
        <span id="theme-icon">☾</span>
      </button>
    </div>
  </div>

  <!-- One toolbar. The tile / variable / layer pickers were three permanent
       full-width rows; they are the same controls, now behind popovers so the
       chrome above the data is one row instead of four. -->
  <div id="toolbar">
    <div class="popwrap">
      <button class="tbtn" id="btn-tiles" aria-expanded="false">
        <span class="bar-label">Tiles</span><span class="cnt" id="cnt-tiles"></span><span class="caret">▾</span>
      </button>
      <div class="popover" id="pop-tiles">
        {tile_tabs_html}
      </div>
    </div>
    <div class="popwrap">
      <button class="tbtn" id="btn-vars" aria-expanded="false">
        <span class="bar-label">Plots</span><span class="cnt" id="cnt-vars"></span><span class="caret">▾</span>
      </button>
      <div class="popover" id="pop-vars">
        <div class="pop-actions">
          <button data-act="all" data-for="vars">Select all</button>
          <button data-act="none" data-for="vars">Clear</button>
        </div>
      </div>
    </div>
    <div class="popwrap">
      <button class="tbtn" id="btn-layers" aria-expanded="false">
        <span class="bar-label">Layers</span><span class="cnt" id="cnt-layers"></span><span class="caret">▾</span>
      </button>
      <div class="popover" id="pop-layers">
        <div class="pop-actions">
          <button data-act="all" data-for="layers">Select all</button>
          <button data-act="none" data-for="layers">Clear</button>
        </div>
      </div>
    </div>
    <span class="info-i" title="Tiles selects which tile is plotted and browsed. Plots chooses which cloud-property statistics are plotted, and the number box sets each plot's height. Layers chooses which per-scene thumbnails appear below.">i</span>
  </div>
</div>

<!-- Map + time series -->
<div id="top-panels">
  {map_html}
  <div id="chart-wrap">
    <span class="info-i chart-i" title="Cloud-property statistics over time. Each point is one scene; click a point to open it in the browser below. Lines are coloured by tile — add or remove tiles / variables with the toolbars above.">i</span>
    <canvas id="chart-canvas"></canvas>
  </div>
</div>

<div id="scene-panel">
  <button id="nav-prev">&#8592;</button>
  <button id="nav-play">&#9654;</button>
  <button id="nav-next">&#8594;</button>
  <select id="speed-select" title="Playback speed">
    <option value="2000">0.5×</option>
    <option value="1000" selected>1×</option>
    <option value="500">2×</option>
    <option value="250">4×</option>
  </select>
  <div id="scene-info">
    <div id="si-date"></div>
    <div id="si-sub"></div>
    <div id="si-stats"></div>
    <div class="meta" id="progress"></div>
  </div>
  <div id="layer-grid"></div>
</div>

<div id="lightbox">
  <span id="lb-close">&#10005;</span>
  <img id="lb-img" src="" alt="">
</div>

<script>
const SCENES = {data_json};
const FIG_BASE = {fig_base_json};   // '' = layers hold full paths; else build url = FIG_BASE+scene_id+'/'+name
const layerURL = (s, name) => FIG_BASE ? (FIG_BASE + s.scene_id + '/' + name) : name;
// Both palettes ship; the active one is chosen by theme so a mode switch does
// not need the report regenerating. TILE_COLORS stays the name the chart and
// map read, and is repointed by applyTheme().
const TILE_COLORS_BY_MODE = {tile_colors_modes_json};
let TILE_COLORS = TILE_COLORS_BY_MODE.dark;

// A canvas does not inherit CSS, so anything painted into the chart or onto the
// map dots has to read the theme tokens explicitly -- otherwise the plot stays
// dark after a switch to light mode. Refreshed by applyTheme().
let THEME = {{}};
function refreshTheme() {{
  const cs = getComputedStyle(document.documentElement);
  const g = n => cs.getPropertyValue(n).trim() || '#888';
  THEME = {{
    surface1: g('--surface-1'), surface2: g('--surface-2'),
    border: g('--border'), borderStrong: g('--border-strong'),
    text1: g('--text-1'), text2: g('--text-2'), text3: g('--text-3'),
  }};
}}
refreshTheme();
const MAP_POINTS = {map_points_json};
const CHART_VARS = [
  {{ key: 'cloud_frac',          label: 'Cloud frac' }},
  {{ key: 'tau__mean',           label: 'τ mean' }},
  {{ key: 'tau__p050',           label: 'τ p50' }},
  {{ key: 'r_eff_liq__p050',     label: 'r_eff liq' }},
  {{ key: 'r_eff_ice__p050',     label: 'r_eff ice' }},
  {{ key: 'ice_liq_ratio__mean', label: 'Ice/liq' }},
];
const LAYER_ORDER = ['true_colour', 'cloud_height', 'properties_tau', 'properties_ice_liq_ratio', 'properties_r_eff_liq', 'properties_r_eff_ice'];
const LAYER_TITLES = {{
  true_colour: 'True colour', cloud_height: 'Cloud-top height', cloud_mask: 'Cloud mask',
  ice_composite: 'Ice composite', properties_tau: 'Optical thickness τ',
  properties_r_eff_liq: 'Effective radius (liquid)', properties_r_eff_ice: 'Effective radius (ice)',
  properties_ice_liq_ratio: 'Ice / liquid ratio', properties_uncertainty: 'Retrieval uncertainty',
  overview: 'Overview',
}};
const DEFAULT_LAYERS = ['true_colour', 'cloud_height', 'properties_ice_liq_ratio'];  // landing thumbnails
// Which section of the scene browser each layer belongs to. Unlisted layers
// fall through to 'Other' so a newly added product still shows up.
const LAYER_GROUP = {{
  true_colour: 'Inputs', cloud_mask: 'Inputs', overview: 'Inputs',
  cloud_height: 'Retrieved', properties_tau: 'Retrieved',
  properties_r_eff_liq: 'Retrieved', properties_r_eff_ice: 'Retrieved',
  properties_ice_liq_ratio: 'Retrieved', ice_composite: 'Retrieved',
  properties_uncertainty: 'Quality',
}};

// All retrieval-plot layer types present across scenes, ordered like the grid.
const ALL_LAYERS = (function() {{
  const set = new Set();
  SCENES.forEach(s => (s.layers || []).forEach(p => set.add(p.split('/').pop().replace('.png', ''))));
  const rest = [...set].filter(n => n !== 'overview' && !LAYER_ORDER.includes(n)).sort();
  const ordered = LAYER_ORDER.filter(n => set.has(n)).concat(rest);
  if (set.has('overview')) ordered.push('overview');
  return ordered;
}})();
let visibleLayers = new Set(DEFAULT_LAYERS.filter(n => ALL_LAYERS.includes(n)));
function prettyLayer(n) {{ return LAYER_TITLES[n] || n.replace(/_/g, ' ').replace(/^properties /, ''); }}

// Layer toggles — one checkbox per retrieval plot; filters the scene-browser grid.
(function buildLayerToggles() {{
  const bar = document.getElementById('pop-layers');
  if (!bar) return;
  ALL_LAYERS.forEach(n => {{
    const label = document.createElement('label');
    label.className = 'tchk';
    label.innerHTML = '<input type="checkbox" data-layer="' + n + '"' +
                      (visibleLayers.has(n) ? ' checked' : '') + '>' + prettyLayer(n);
    label.querySelector('input').addEventListener('change', e => {{
      if (e.target.checked) visibleLayers.add(n); else visibleLayers.delete(n);
      showScene(current);
    }});
    bar.appendChild(label);
  }});
}})();

// Global index lookup for map
const SCENE_IDX = {{}};
SCENES.forEach((s, i) => {{ SCENE_IDX[s.scene_id] = i; }});

// ── Time-range filter (timeseries) ────────────────────────────────────────────
const _allT = SCENES.map(s => s.date ? new Date(s.date).getTime() : null).filter(Boolean);
const T_MIN_FULL = _allT.reduce((a, b) => b < a ? b : a, Infinity);
const T_MAX_FULL = _allT.reduce((a, b) => b > a ? b : a, -Infinity);
let dateStart = T_MIN_FULL, dateEnd = T_MAX_FULL;   // default = whole series
const _isoDay = ts => new Date(ts).toISOString().slice(0, 10);
(function initDateRange() {{
  const ds = document.getElementById('date-start'), de = document.getElementById('date-end');
  if (!ds || !de) return;
  ds.min = de.min = _isoDay(T_MIN_FULL); ds.max = de.max = _isoDay(T_MAX_FULL);
  ds.value = _isoDay(T_MIN_FULL); de.value = _isoDay(T_MAX_FULL);
  function apply() {{
    dateStart = ds.value ? new Date(ds.value + 'T00:00:00Z').getTime() : T_MIN_FULL;
    dateEnd   = de.value ? new Date(de.value + 'T23:59:59Z').getTime() : T_MAX_FULL;
    if (dateEnd < dateStart) {{ const t = dateStart; dateStart = dateEnd; dateEnd = t; }}
    setPlaying(false); current = 0;
    showScene(0);          // rescope the scene browser to the range (also redraws chart/map)
    drawChart();           // in case the range is empty (showScene bails early)
  }}
  ds.onchange = apply; de.onchange = apply;
  document.getElementById('date-reset').onclick = () => {{
    ds.value = _isoDay(T_MIN_FULL); de.value = _isoDay(T_MAX_FULL); apply();
  }};
}})();

// ── Tile state ────────────────────────────────────────────────────────────────
let activeTile = null;   // null = first tile is selected by default
let current = 0;         // index within activeScenes()

function activeScenes() {{
  return SCENES.filter(s => {{
    if (activeTile && s.tile_id !== activeTile) return false;
    const t = s.date ? new Date(s.date).getTime() : null;   // scope to the selected time range
    return t != null && t >= dateStart && t <= dateEnd;
  }});
}}


// The one place tile selection is decided. Selecting a tile in the popover,
// clicking its dot on the map and clicking a point on the chart all land here,
// and the chart then plots exactly this tile -- there is no separate notion of
// "which tiles are plotted" any more.
function switchTile(tile) {{
  setPlaying(false);
  activeTile = tile || null;
  current = 0;
  document.querySelectorAll('#pop-tiles .tchk').forEach(l =>
    l.classList.toggle('focused', l.dataset.tile === (tile || ''))
  );
  const radio = document.querySelector(
    '#pop-tiles input[data-tile="' + (tile || '') + '"]');
  if (radio) radio.checked = true;
  const cnt = document.getElementById('cnt-tiles');
  if (cnt) cnt.textContent = tile || '';
  showScene(0);          // redraws chart + map dots
}}

document.querySelectorAll('#pop-tiles input[name="tile-select"]').forEach(rb =>
  rb.addEventListener('change', () => {{
    if (rb.checked) switchTile(rb.dataset.tile);
  }})
);

// ── Chart ────────────────────────────────────────────────────────────────────
const chartCanvas = document.getElementById('chart-canvas');
const chartCtx = chartCanvas.getContext('2d');
let chartPoints = [];
const DEFAULT_TILE = Object.keys(TILE_COLORS).sort()[0];   // landing tile
let visibleVars = new Set([CHART_VARS[0].key]);   // landing: one variable; add more via Plots
let legendItems = [];
const LEGEND_H = 24;
const AXIS_H = 20;     // room for the date axis under the timeseries
const DEFAULT_ROW_H = 150;
let rowHeights = {{}};  // per-variable chart row height in px (set via the Plots height boxes)
const rowH = key => rowHeights[key] || DEFAULT_ROW_H;
const _MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

// Adaptive date ticks: ~6 labels, stepping by years / months / days by span.
function dateTicks(t0, t1) {{
  const D = 86400000, spanD = (t1 - t0) / D, TARGET = 6, out = [];
  const d0 = new Date(t0), d1 = new Date(t1);
  const Y0 = d0.getUTCFullYear(), Y1 = d1.getUTCFullYear();
  if (spanD > 730) {{                                    // yearly
    const step = [1,2,5,10,20,50].find(s => (Y1 - Y0) / s <= TARGET) || 100;
    for (let y = Math.ceil(Y0/step)*step; y <= Y1; y += step)
      out.push({{ t: Date.UTC(y,0,1), lab: '' + y }});
  }} else if (spanD > 75) {{                             // monthly
    const months = (Y1 - Y0)*12 + (d1.getUTCMonth() - d0.getUTCMonth());
    const step = [1,2,3,6].find(s => months / s <= TARGET) || 12;
    let y = Y0, m = Math.floor(d0.getUTCMonth()/step)*step, t = Date.UTC(y,m,1);
    while (t <= t1) {{
      if (t >= t0) {{ const dd = new Date(t);
        out.push({{ t, lab: _MON[dd.getUTCMonth()] + " '" + (''+dd.getUTCFullYear()).slice(2) }}); }}
      m += step; if (m >= 12) {{ m -= 12; y++; }} t = Date.UTC(y,m,1);
    }}
  }} else {{                                             // daily
    const step = [1,2,5,10,15,30].find(s => spanD / s <= TARGET) || 30;
    for (let t = Math.ceil(t0/(step*D))*(step*D); t <= t1; t += step*D) {{
      const dd = new Date(t); out.push({{ t, lab: dd.getUTCDate() + " " + _MON[dd.getUTCMonth()] }});
    }}
  }}
  return out;
}}

// Plot toggles — one checkbox per timeseries variable, built from CHART_VARS.
(function buildVarToggles() {{
  const bar = document.getElementById('pop-vars');
  if (!bar) return;
  CHART_VARS.forEach(v => {{
    const label = document.createElement('label');
    label.className = 'tchk';
    label.innerHTML = '<input type="checkbox" data-var="' + v.key + '"' +
                      (visibleVars.has(v.key) ? ' checked' : '') + '>' + v.label +
                      '<input type="number" class="row-h" data-var="' + v.key +
                      '" value="' + DEFAULT_ROW_H + '" min="50" max="700" step="10" ' +
                      'title="height of this plot in px">';
    label.querySelector('input[type=checkbox]').addEventListener('change', e => {{
      if (e.target.checked) visibleVars.add(v.key); else visibleVars.delete(v.key);
      drawChart();
    }});
    label.querySelector('.row-h').addEventListener('input', e => {{
      const h = parseInt(e.target.value, 10);
      if (h >= 50) {{ rowHeights[v.key] = h; drawChart(); }}
    }});
    bar.appendChild(label);
  }});
}})();

function drawChart() {{
  const vars = CHART_VARS.filter(v => visibleVars.has(v.key));
  const nVars = vars.length;
  const W = chartCanvas.offsetWidth || 400;
  let cum = LEGEND_H;                          // stack rows with per-variable (customisable) heights
  const rowY = {{}};
  vars.forEach(v => {{ rowY[v.key] = cum; cum += rowH(v.key); }});
  const bodyBottom = cum;
  const totalH = bodyBottom + AXIS_H;
  // Bitmap in device pixels, CSS box in CSS pixels, then a transform so the
  // drawing code below stays in CSS pixels. Setting only .width/.height left the
  // bitmap at whatever the width was when it was last drawn while CSS kept
  // stretching it to 100% of the panel -- so after a resize every circle came
  // out as an ellipse and the text sheared.
  //
  // style.width is deliberately NOT pinned: `width:100%` is what makes
  // offsetWidth report the space actually available. Pinning it here would make
  // the next measurement read back this same value and the chart could never
  // grow again.
  const dpr = window.devicePixelRatio || 1;
  chartCanvas.width = Math.round(W * dpr);
  chartCanvas.height = Math.round(totalH * dpr);
  chartCanvas.style.height = totalH + 'px';
  chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const padL = 44, padR = 10, padT = 8, padB = 6;
  const cw = W - padL - padR;

  chartCtx.fillStyle = THEME.surface1;
  chartCtx.fillRect(0, 0, W, totalH);

  // ── Caption ─────────────────────────────────────────────────────────────────
  // A single series needs no legend box -- the caption names it. This replaces
  // the old row of per-tile toggles: the chart now shows the tile being browsed,
  // so a legend listing every tile would describe lines that are not drawn.
  legendItems = [];
  chartCtx.font = '9px monospace';
  if (activeTile) {{
    chartCtx.globalAlpha = 1;
    chartCtx.fillStyle = TILE_COLORS[activeTile] || THEME.text3;
    chartCtx.beginPath();
    chartCtx.arc(padL + 5, LEGEND_H / 2, 4, 0, Math.PI * 2);
    chartCtx.fill();
    chartCtx.fillStyle = THEME.text1;
    chartCtx.textAlign = 'left';
    chartCtx.fillText(activeTile, padL + 12, LEGEND_H / 2 + 4);
    const nS = SCENES.filter(s => s.tile_id === activeTile).length;
    chartCtx.fillStyle = THEME.text3;
    chartCtx.fillText(nS + (nS === 1 ? ' scene' : ' scenes'),
                      padL + 12 + chartCtx.measureText(activeTile).width + 10,
                      LEGEND_H / 2 + 4);
  }}

  // ── Separator ───────────────────────────────────────────────────────────────
  chartCtx.strokeStyle = THEME.border; chartCtx.lineWidth = 1;
  chartCtx.beginPath(); chartCtx.moveTo(0, LEGEND_H); chartCtx.lineTo(W, LEGEND_H); chartCtx.stroke();

  // ── Time series rows ────────────────────────────────────────────────────────
  const allDates = SCENES.map(s => s.date ? new Date(s.date).getTime() : null);
  const inRange = t => t != null && t >= dateStart && t <= dateEnd;
  if (!allDates.some(inRange)) return;
  const tMin = dateStart, tMax = dateEnd;   // x-axis spans the selected range
  const txT = t => padL + ((t - tMin) / (tMax - tMin || 1)) * cw;
  const tx = i => txT(allDates[i]);

  const sc = activeScenes();
  const currentGlobal = sc.length ? SCENE_IDX[sc[current]?.scene_id] : -1;

  chartPoints = [];

  vars.forEach(({{ key, label }}, vi) => {{
    const yOff = rowY[key];
    const ch = rowH(key) - padT - padB;

    // Scale to the tile on screen, not to every tile in the project. Rama's
    // 8 tiles span very different cloud regimes, so a global maximum flattened
    // the line you were actually looking at.
    const allVals = SCENES
      .filter(s => !activeTile || s.tile_id === activeTile)
      .map(s => s.stats[key]).filter(v => v != null);
    if (!allVals.length) return;
    const vMax = Math.max(...allVals) * 1.1 || 1;
    const ty = v => yOff + padT + ch - (v / vMax) * ch;

    if (vi > 0) {{
      chartCtx.strokeStyle = THEME.border; chartCtx.lineWidth = 1;
      chartCtx.beginPath(); chartCtx.moveTo(0, yOff); chartCtx.lineTo(W, yOff); chartCtx.stroke();
    }}

    chartCtx.font = '8px monospace'; chartCtx.textAlign = 'right';
    chartCtx.fillStyle = THEME.text3;
    chartCtx.fillText(vMax.toFixed(2), padL - 3, yOff + padT + 4);
    chartCtx.fillText('0', padL - 3, yOff + padT + ch);
    chartCtx.font = 'bold 10px system-ui'; chartCtx.fillStyle = THEME.text2; chartCtx.textAlign = 'left';
    chartCtx.fillText(label, padL + 4, yOff + padT + 4);

    chartCtx.strokeStyle = THEME.surface2; chartCtx.lineWidth = 1;
    chartCtx.beginPath(); chartCtx.moveTo(padL, yOff + padT + ch); chartCtx.lineTo(padL + cw, yOff + padT + ch); chartCtx.stroke();

    const byTile = {{}};
    SCENES.forEach((s, i) => {{
      const v = s.stats[key];
      if (v == null || !inRange(allDates[i])) return;   // date-range filter
      const tid = s.tile_id || '_';
      (byTile[tid] = byTile[tid] || []).push({{ i, x: tx(i), y: ty(v) }});
    }});

    // Only the tile being browsed is plotted. Drawing the others faintly meant
    // clicking a dot on the map changed several lines at once, and the y-scale
    // was set by tiles you were not looking at.
    Object.entries(byTile).forEach(([tid, pts]) => {{
      if (activeTile && tid !== activeTile) return;
      const color = TILE_COLORS[tid] || THEME.text3;
      chartCtx.strokeStyle = color;
      chartCtx.globalAlpha = 0.55;
      chartCtx.lineWidth = 1.5;
      chartCtx.beginPath();
      pts.forEach((p, j) => j === 0 ? chartCtx.moveTo(p.x, p.y) : chartCtx.lineTo(p.x, p.y));
      chartCtx.stroke();
      pts.forEach(p => {{
        const isCurrent = p.i === currentGlobal;
        chartCtx.globalAlpha = isCurrent ? 1 : 0.8;
        chartCtx.fillStyle = color;
        chartCtx.beginPath();
        chartCtx.arc(p.x, p.y, isCurrent ? 4.5 : 2, 0, Math.PI * 2);
        chartCtx.fill();
        if (isCurrent) {{ chartCtx.strokeStyle = THEME.text1; chartCtx.lineWidth = 1.5; chartCtx.stroke(); }}
        if (vi === 0) chartPoints.push({{ i: p.i, x: p.x, y: p.y, tid }});
      }});
    }});
    chartCtx.globalAlpha = 1;
  }});

  // ── Date axis ─────────────────────────────────────────────────────────────
  const yAxis = bodyBottom;
  chartCtx.strokeStyle = THEME.border; chartCtx.lineWidth = 1;
  chartCtx.beginPath(); chartCtx.moveTo(padL, yAxis); chartCtx.lineTo(padL + cw, yAxis); chartCtx.stroke();
  chartCtx.font = '9px system-ui'; chartCtx.textAlign = 'center';
  for (const {{ t, lab }} of dateTicks(tMin, tMax)) {{
    const x = txT(t);
    if (x < padL - 1 || x > padL + cw + 1) continue;
    chartCtx.strokeStyle = THEME.borderStrong;
    chartCtx.beginPath(); chartCtx.moveTo(x, yAxis); chartCtx.lineTo(x, yAxis + 4); chartCtx.stroke();
    chartCtx.fillStyle = THEME.text3; chartCtx.fillText(lab, x, yAxis + 15);
  }}
}}

chartCanvas.onclick = e => {{
  const r = chartCanvas.getBoundingClientRect();
  // CSS pixels: everything was drawn through the devicePixelRatio transform, so
  // scaling by bitmap/CSS here would put the hit test at dpr times the cursor.
  const mx = e.clientX - r.left;
  const my = e.clientY - r.top;

  // Caption strip is not interactive any more — one tile is plotted, and it is
  // chosen from the Tiles picker or by clicking the map.
  if (my < LEGEND_H) return;

  // Time series area — navigate to nearest scene
  let closest = null, minDx = Infinity, closestTile = null;
  chartPoints.forEach(p => {{
    const d = Math.abs(p.x - mx);
    if (d < minDx) {{ minDx = d; closest = p.i; closestTile = p.tid; }}
  }});
  if (closest !== null) {{
    if (closestTile && closestTile !== activeTile) switchTile(closestTile);
    const sc = activeScenes();
    const localIdx = sc.findIndex(s => SCENE_IDX[s.scene_id] === closest);
    if (localIdx >= 0) showScene(localIdx);
  }}
}};

// ── Map ──────────────────────────────────────────────────────────────────────
const mapCanvas = document.getElementById('map-canvas');
const mapBg = document.getElementById('map-bg');
const mapWrap = document.getElementById('map-wrap');
const mapInner = document.getElementById('map-inner');
let mapDragMoved = false;                 // set while panning → not a click
let mapScale = 1, mapTx = 0, mapTy = 0;   // basemap transform state

// PNG-pixel → viewport-pixel scale at the current zoom.
function mapBaseScale() {{
  return (mapBg && mapBg.naturalHeight) ? (mapWrap.clientHeight / mapBg.naturalHeight) : 1;
}}

// Size the dot canvas to its CSS box AND to the device pixel ratio, then undo
// the ratio with a transform so drawing code stays in CSS pixels.
//
// Without this the canvas bitmap keeps whatever size it had when the page first
// loaded. Any later change to its CSS box -- a browser zoom, a window resize --
// leaves the browser stretching that stale bitmap, so the dots grow with the map
// instead of staying a constant size, and blur as they go.
function sizeMapCanvas() {{
  if (!mapCanvas || !mapWrap) return false;
  const dpr = window.devicePixelRatio || 1;
  const w = mapWrap.clientWidth, h = mapWrap.clientHeight;
  if (!w || !h) return false;
  mapCanvas.style.width = w + 'px';
  mapCanvas.style.height = h + 'px';
  mapCanvas.width = Math.round(w * dpr);
  mapCanvas.height = Math.round(h * dpr);
  mapCanvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  return true;
}}

function initMap() {{
  if (!mapBg || !MAP_POINTS.length || !mapWrap) return;
  if (!sizeMapCanvas()) return;
  drawMapDots();
}}

// Re-size on anything that changes the box: browser zoom, window resize, layout.
if (window.ResizeObserver && mapWrap) {{
  new ResizeObserver(() => {{ if (sizeMapCanvas()) drawMapDots(); }}).observe(mapWrap);
}} else {{
  window.addEventListener('resize', () => {{ if (sizeMapCanvas()) drawMapDots(); }});
}}

// Dots live on a viewport-fixed canvas (NOT inside the transformed image), so they
// keep a constant on-screen size and stay crisp at any zoom.
function drawMapDots() {{
  if (!mapCanvas.width || !mapWrap) return;
  const ctx = mapCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  // Own the transform on every frame instead of trusting whatever sizeMapCanvas
  // last left behind. Clear under the IDENTITY transform using the bitmap's own
  // dimensions: clearing in CSS pixels while the context happened to be at
  // identity wiped only the top-left 1/dpr of the bitmap, so previous frames
  // survived around the edges and dots accumulated as rings on top of each
  // other -- which reads as dots that grow and double when you zoom.
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cw = mapWrap.clientWidth, ch = mapWrap.clientHeight;
  const ds = mapBaseScale() * mapScale;
  const sc = activeScenes();
  const currentGlobal = sc.length ? SCENE_IDX[sc[current]?.scene_id] : -1;

  MAP_POINTS.forEach((p, i) => {{
    if (!p) return;
    const tid = SCENES[i].tile_id;
    // every tile's dots are always drawn — tile toggles only affect the chart lines
    const x = mapTx + p.x * ds, y = mapTy + p.y * ds;
    if (x < -8 || y < -8 || x > cw + 8 || y > ch + 8) return;
    const isActiveTile = !activeTile || tid === activeTile;
    const isCurrent = i === currentGlobal;
    ctx.globalAlpha = isCurrent ? 1 : (isActiveTile ? 0.7 : 0.3);
    ctx.fillStyle = TILE_COLORS[tid] || THEME.text3;
    ctx.beginPath();
    ctx.arc(x, y, isCurrent ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    // The basemap PNG is baked by cartopy and is dark in BOTH themes, so the
    // ring is chosen against the map, not against the page. Theming it to
    // --surface-1 was wrong: in dark mode that painted a near-black ring onto a
    // near-black basemap.
    ctx.strokeStyle = isCurrent ? '#ffffff' : 'rgba(255,255,255,0.55)';
    ctx.lineWidth = isCurrent ? 2 : 1;
    ctx.stroke();
  }});
  ctx.globalAlpha = 1;
}}

if (mapBg) {{ mapBg.onload = initMap; if (mapBg.complete) initMap(); }}
window.addEventListener('resize', initMap);

if (mapCanvas) {{
  mapCanvas.onclick = e => {{
    if (mapDragMoved) return;   // this "click" was actually a pan-drag
    const r = mapCanvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const ds = mapBaseScale() * mapScale;
    let closest = null, minD = 16, closestTile = null;
    MAP_POINTS.forEach((p, i) => {{
      if (!p) return;                          // all dots hoverable (map ignores tile toggles)
      const x = mapTx + p.x * ds, y = mapTy + p.y * ds;
      const d = Math.hypot(x - mx, y - my);
      if (d < minD) {{ minD = d; closest = i; closestTile = SCENES[i].tile_id; }}
    }});
    if (closest !== null) {{
      if (closestTile && closestTile !== activeTile) switchTile(closestTile);
      const sc = activeScenes();
      const localIdx = sc.findIndex(s => SCENE_IDX[s.scene_id] === closest);
      if (localIdx >= 0) showScene(localIdx);
    }}
  }};
}}

// Pan & zoom: scroll to zoom toward the cursor, drag to pan, double-click to reset.
// Only the basemap image is transformed; dots are redrawn (constant size) each frame.
(function() {{
  if (!mapWrap || !mapInner) return;
  // Capped to what the basemap can actually resolve: it is rendered ~7x the
  // displayed width, so allowing 12x only offered blur. The dots stay sharp at
  // any zoom -- they are drawn on the canvas, not baked into the image.
  const MIN = 1, MAX = 7;
  const apply = () => {{
    mapInner.style.transform = 'translate(' + mapTx + 'px,' + mapTy + 'px) scale(' + mapScale + ')';
    drawMapDots();
  }};
  function clamp() {{
    const w = mapWrap.clientWidth, h = mapWrap.clientHeight;
    mapTx = Math.min(0, Math.max(w - w * mapScale, mapTx));
    mapTy = Math.min(0, Math.max(h - h * mapScale, mapTy));
  }}
  mapWrap.addEventListener('wheel', e => {{
    e.preventDefault();
    const r = mapWrap.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top, prev = mapScale;
    mapScale = Math.min(MAX, Math.max(MIN, mapScale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    const k = mapScale / prev;
    mapTx = cx - (cx - mapTx) * k;
    mapTy = cy - (cy - mapTy) * k;
    clamp(); apply();
  }}, {{ passive: false }});
  let dragging = false, sx = 0, sy = 0, startX = 0, startY = 0;
  mapWrap.addEventListener('mousedown', e => {{
    dragging = true; mapDragMoved = false;
    startX = e.clientX; startY = e.clientY; sx = e.clientX - mapTx; sy = e.clientY - mapTy;
    mapWrap.classList.add('grabbing');
  }});
  window.addEventListener('mousemove', e => {{
    if (!dragging) return;
    mapTx = e.clientX - sx; mapTy = e.clientY - sy;
    if (Math.hypot(e.clientX - startX, e.clientY - startY) > 4) mapDragMoved = true;
    clamp(); apply();
  }});
  window.addEventListener('mouseup', () => {{ dragging = false; mapWrap.classList.remove('grabbing'); }});
  mapWrap.addEventListener('dblclick', () => {{ mapScale = 1; mapTx = 0; mapTy = 0; apply(); }});
}})();

// ── Scene browser ─────────────────────────────────────────────────────────────
function labelFromPath(p) {{
  const n = p.split('/').pop().replace('.png', '');
  return LAYER_TITLES[n] || n.replace(/_/g, ' ');
}}

function showScene(idx) {{
  const sc = activeScenes();
  if (!sc.length) return;
  current = (idx + sc.length) % sc.length;
  const s = sc[current];

  document.getElementById('si-date').textContent = s.date || '—';
  document.getElementById('si-sub').textContent = [s.tile_id, s.satellite].filter(Boolean).join(' · ');
  document.getElementById('progress').textContent =
    `${{s.tile_id || ''}}  ·  Scene ${{current + 1}} / ${{sc.length}}`;

  // Stat tiles: label above value, tabular figures, so the numbers line up in
  // the same place from scene to scene instead of reflowing as a sentence.
  const statTiles = Object.entries({{
    cloud_frac: 'Cloud frac', tau__mean: 'τ mean',
    r_eff_liq__p050: 'r_eff liq p50', ice_liq_ratio__mean: 'Ice/liq',
  }}).filter(([k]) => s.stats[k] != null)
    .map(([k, lbl]) => `<div class="stat"><span class="k">${{lbl}}</span>` +
                       `<span class="v">${{s.stats[k].toFixed(3)}}</span></div>`);
  document.getElementById('si-stats').innerHTML = statTiles.join('');

  const grid = document.getElementById('layer-grid');
  grid.innerHTML = '';
  const sortedLayers = [...s.layers].sort((a, b) => {{
    const aName = a.split('/').pop().replace('.png', '');
    const bName = b.split('/').pop().replace('.png', '');
    if (aName === 'overview') return 1;
    if (bName === 'overview') return -1;
    const ai = LAYER_ORDER.indexOf(aName), bi = LAYER_ORDER.indexOf(bName);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return 0;
  }});
  // Group by role rather than one flat auto-fill: what went in, what came out,
  // and how much to trust it. Anything unrecognised falls through to OTHER so a
  // new layer still appears instead of vanishing.
  const groups = [['Inputs', []], ['Retrieved', []], ['Quality', []], ['Other', []]];
  const GI = {{ Inputs: 0, Retrieved: 1, Quality: 2, Other: 3 }};
  sortedLayers
    .filter(path => visibleLayers.has(path.split('/').pop().replace('.png', '')))
    .forEach(path => {{
      const name = path.split('/').pop().replace('.png', '');
      groups[GI[LAYER_GROUP[name] || 'Other']][1].push(path);
    }});
  groups.filter(([, paths]) => paths.length).forEach(([gname, paths]) => {{
    const sec = document.createElement('div');
    sec.className = 'layer-group';
    const head = document.createElement('div');
    head.className = 'group-head';
    head.textContent = gname;
    sec.appendChild(head);
    paths.forEach(path => {{
      const name = path.split('/').pop().replace('.png', '');
      const div = document.createElement('div');
      div.className = 'layer-thumb' + (name === 'overview' ? ' overview' : '');
      const img = document.createElement('img');
      const src = layerURL(s, path);
      img.src = src;
      img.loading = 'lazy';
      img.alt = labelFromPath(path) + ' — ' + (s.scene_id || '');
      img.onclick = () => openLightbox(src);
      const lbl = document.createElement('div');
      lbl.className = 'layer-label';
      lbl.textContent = labelFromPath(path);
      div.appendChild(img); div.appendChild(lbl);
      sec.appendChild(div);
    }});
    grid.appendChild(sec);
  }});

  drawChart();
  drawMapDots();
  preloadScene(current + 1);
  preloadScene(current + 2);
}}

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(src) {{
  document.getElementById('lb-img').src = src;
  document.getElementById('lightbox').classList.add('active');
}}
document.getElementById('lb-close').onclick = () =>
  document.getElementById('lightbox').classList.remove('active');
document.getElementById('lightbox').onclick = e => {{
  if (e.target.id === 'lightbox') document.getElementById('lightbox').classList.remove('active');
}};

// ── Play / Pause ──────────────────────────────────────────────────────────────
let playTimer = null;
const playBtn = document.getElementById('nav-play');

function setPlaying(on) {{
  if (on) {{
    const ms = parseInt(document.getElementById('speed-select').value);
    playTimer = setInterval(() => showScene(current + 1), ms);
    playBtn.textContent = '⏸';
    playBtn.classList.add('playing');
  }} else {{
    clearInterval(playTimer); playTimer = null;
    playBtn.textContent = '▶';
    playBtn.classList.remove('playing');
  }}
}}

playBtn.onclick = () => setPlaying(!playTimer);
document.getElementById('speed-select').onchange = () => {{
  if (playTimer) {{ setPlaying(false); setPlaying(true); }}
}};

// ── Navigation ────────────────────────────────────────────────────────────────
document.getElementById('nav-prev').onclick = () => {{ setPlaying(false); showScene(current - 1); }};
document.getElementById('nav-next').onclick = () => {{ setPlaying(false); showScene(current + 1); }};
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft')  {{ setPlaying(false); showScene(current - 1); }}
  if (e.key === 'ArrowRight') {{ setPlaying(false); showScene(current + 1); }}
  if (e.key === ' ') {{ e.preventDefault(); setPlaying(!playTimer); }}
  if (e.key === 'Escape') {{
    setPlaying(false);
    document.getElementById('lightbox').classList.remove('active');
  }}
}});

window.addEventListener('resize', () => {{ drawChart(); initMap(); }});
// A plain resize listener misses layout-only changes -- a popover opening, the
// scrollbar appearing, a font settling -- which move the panel without resizing
// the window. Observe the panel itself.
if (window.ResizeObserver) {{
  const cw = document.getElementById('chart-wrap');
  if (cw) {{
    let lastW = 0;
    new ResizeObserver(() => {{
      const w = cw.clientWidth;
      if (w && w !== lastW) {{ lastW = w; drawChart(); }}
    }}).observe(cw);
  }}
}}

// ── Preloader ─────────────────────────────────────────────────────────────────
const _preloadCache = {{}};
function preloadScene(idx) {{
  const sc = activeScenes();
  if (!sc.length) return;
  const i = (idx + sc.length) % sc.length;
  if (_preloadCache[sc[i].scene_id]) return;
  _preloadCache[sc[i].scene_id] = true;
  sc[i].layers.forEach(path => {{ const img = new Image(); img.src = layerURL(sc[i], path); }});
}}

// ── Toolbar popovers ─────────────────────────────────────────────────────────
// The tile / plot / layer pickers were three permanent rows. Same checkboxes,
// now one button each; only one popover is open at a time.
const POPS = [['btn-tiles', 'pop-tiles'], ['btn-vars', 'pop-vars'], ['btn-layers', 'pop-layers']];
function closePops(except) {{
  POPS.forEach(([b, p]) => {{
    if (p === except) return;
    document.getElementById(p).classList.remove('open');
    const btn = document.getElementById(b);
    btn.classList.remove('open');
    btn.setAttribute('aria-expanded', 'false');
  }});
}}
POPS.forEach(([b, p]) => {{
  const btn = document.getElementById(b), pop = document.getElementById(p);
  if (!btn || !pop) return;
  btn.addEventListener('click', e => {{
    e.stopPropagation();
    const willOpen = !pop.classList.contains('open');
    closePops(willOpen ? p : null);
    pop.classList.toggle('open', willOpen);
    btn.classList.toggle('open', willOpen);
    btn.setAttribute('aria-expanded', String(willOpen));
  }});
  pop.addEventListener('click', e => e.stopPropagation());
}});
document.addEventListener('click', () => closePops(null));
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closePops(null); }});

// Select all / Clear inside each popover.
document.querySelectorAll('.pop-actions button').forEach(btn =>
  btn.addEventListener('click', () => {{
    const pop = document.getElementById('pop-' + btn.dataset.for);
    const want = btn.dataset.act === 'all';
    pop.querySelectorAll('input[type=checkbox]').forEach(cb => {{
      if (cb.checked !== want) {{ cb.checked = want; cb.dispatchEvent(new Event('change')); }}
    }});
    updateCounts();
  }})
);

// "3 / 12" next to each button, so the collapsed state still says what is on.
function updateCounts() {{
  // 'tiles' is deliberately absent: it is a single choice, so its badge shows
  // the selected tile's name and is owned by switchTile(), not a "n / m" count.
  [['vars', 'cnt-vars'], ['layers', 'cnt-layers']].forEach(([k, id]) => {{
    const pop = document.getElementById('pop-' + k);
    const el = document.getElementById(id);
    if (!pop || !el) return;
    const all = pop.querySelectorAll('input[type=checkbox]');
    const on = pop.querySelectorAll('input[type=checkbox]:checked');
    el.textContent = all.length ? on.length + ' / ' + all.length : '';
  }});
}}
document.querySelectorAll('.popover input[type=checkbox]').forEach(cb =>
  cb.addEventListener('change', updateCounts));

// ── Theme ────────────────────────────────────────────────────────────────────
// Dark stays the default so an existing report looks unchanged; the choice is
// remembered per viewer. Tile colours are restepped per surface, not reused:
// a hue that clears the contrast gate on #0f1117 does not clear it on #f6f7fa.
function applyTheme(mode) {{
  document.documentElement.setAttribute('data-theme', mode);
  TILE_COLORS = TILE_COLORS_BY_MODE[mode] || TILE_COLORS_BY_MODE.dark;
  document.querySelectorAll('.tab-dot[data-tile]').forEach(d => {{
    d.style.background = TILE_COLORS[d.dataset.tile] || '#8a9099';
  }});
  // Logos carry both variants; when no -light file was supplied the two data
  // URIs are identical and this changes nothing.
  document.querySelectorAll('#logo, #logo2').forEach(img => {{
    const next = mode === 'light' ? img.dataset.light : img.dataset.dark;
    if (next && img.getAttribute('src') !== next) img.setAttribute('src', next);
  }});
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = mode === 'light' ? '☀' : '☾';
  try {{ localStorage.setItem('cd-report-theme', mode); }} catch (e) {{}}
  // Re-read the tokens BEFORE repainting: the chart and dots are canvas, so
  // they take no colour from the stylesheet on their own.
  refreshTheme();
  drawChart();
  drawMapDots();
}}
(function initTheme() {{
  let saved = null;
  try {{ saved = localStorage.getItem('cd-report-theme'); }} catch (e) {{}}
  applyTheme(saved === 'light' ? 'light' : 'dark');
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', () => {{
    const now = document.documentElement.getAttribute('data-theme');
    applyTheme(now === 'light' ? 'dark' : 'light');
  }});
}})();

// ── Init ──────────────────────────────────────────────────────────────────────
// Select the first tile by default
const firstTile = Object.keys(TILE_COLORS).sort()[0];
if (firstTile) switchTile(firstTile);
else showScene(0);
updateCounts();
</script>
</body>
</html>"""


def _generate_thumbnail(args: tuple) -> tuple[str, Optional[str]]:
    """Worker: render all thumbnails for one scene. Returns (scene_id, error_or_None)."""
    scene_id, scene_dir, scene_out, dpi = args
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from clouds_decoded.visualisation.static import save_figure
        from clouds_decoded.visualisation.visualiser import Visualiser
        scene_out = Path(scene_out)
        scene_out.mkdir(parents=True, exist_ok=True)
        vis = Visualiser.from_directory(str(scene_dir))
        fig = vis.overview()
        save_figure(fig, str(scene_out / "overview.png"), dpi=dpi)
        plt.close(fig)
        for name in vis.layer_names:
            fname = name.lower().replace(": ", "_").replace(" ", "_") + ".png"
            fig = vis.plot(name)
            save_figure(fig, str(scene_out / fname), dpi=dpi)
            plt.close(fig)
        return scene_id, None
    except Exception as e:
        return scene_id, str(e)


def generate_report(
    project_dir: Path,
    output_path: Optional[Path] = None,
    dpi: int = 72,
    regenerate_figures: bool = False,
    db_path: Optional[Path] = None,
    workers: int = 4,
    brand_logo: Optional[str] = None,
    project_logo: Optional[str] = None,
) -> Path:
    """Generate an HTML report with map, timeseries and scene thumbnail browser.

    ``output_path`` defaults to ``<project_dir>/report.html`` and should stay
    inside ``project_dir``: thumbnail paths in the HTML are relative to it, so a
    report written elsewhere renders with every image broken. The CLI does not
    expose this argument for that reason.
    """
    output_path = output_path or project_dir / "report.html"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    df = _load_project_data(project_dir, db_path=db_path)
    if df.empty:
        raise ValueError("No completed runs found in project.")
    logger.info(f"Building report for {len(df)} scenes")

    # Generate missing thumbnails
    outputs_dir = project_dir / "outputs"
    missing = [
        row.scene_id for _, row in df.iterrows()
        if regenerate_figures or not (figures_dir / row.scene_id / "overview.png").exists()
    ]
    if missing:
        tasks = [
            (sid, str(outputs_dir / sid), str(figures_dir / sid), dpi)
            for sid in missing
            if (outputs_dir / sid).exists()
        ]
        logger.info(f"Generating thumbnails for {len(tasks)} scene(s) at {dpi} DPI "
                    f"using {workers} worker(s)...")
        done_count = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_generate_thumbnail, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                scene_id, err = fut.result()
                done_count += 1
                if err:
                    logger.warning(f"[{scene_id}] thumbnail failed: {err}")
                else:
                    logger.info(f"[{scene_id}] thumbnails done ({done_count}/{len(tasks)})")

    scenes = _build_scenes(df, figures_dir)
    tile_modes = _tile_color_map(scenes)
    tile_colors = tile_modes['dark']

    n_tiles = len({s['tile_id'] for s in scenes if s.get('tile_id')})
    if n_tiles > len(_PALETTE_DARK):
        logger.warning(
            "%d tiles but only %d categorical colours — the remaining %d share a "
            "neutral grey. Use the tile filter to isolate one.",
            n_tiles, len(_PALETTE_DARK), n_tiles - len(_PALETTE_DARK))

    logger.info("Generating map...")
    # Basemap only. With draw_markers=True matplotlib baked a marker per scene
    # INTO the PNG, and the interactive canvas then drew its own dot on top of
    # each one. The baked copy is part of the image, so it scaled with the
    # zoom transform while the canvas dot stayed a constant 4px -- giving a
    # growing disc with a small dot at its centre, and two dots per scene at
    # every zoom level. The canvas markers are the ones that can be positioned,
    # recoloured by theme and clicked, so they are the ones to keep.
    # Rendered ~7x the on-screen size (displayed ~418px wide) so the basemap
    # survives zooming, and at Natural Earth 10m so the extra pixels carry real
    # coastline detail rather than a smoother version of 50m. Measured on
    # cas_meeting_examples: 653 KB and 0.7s, against 57 KB at the old 500x380.
    # Going to 4400px would cover the full 12x but costs 1.1 MB, which is a poor
    # trade on a small project -- the zoom cap is lowered to match instead.
    #
    # The first 10m draw in a process pays a one-time ~18s shapefile load.
    map_png, map_points = _generate_map(
        scenes, tile_colors, draw_markers=False,
        width_px=3000, height_px=2280, dpi=150, feature_scale='10m')

    html = _render_html(
        scenes, tile_colors, map_png, map_points, project_dir.name,
        brand_logo=_asset_uri("asterisk-labs.svg") if brand_logo is None else brand_logo,
        project_logo=_asset_uri("clouds-decoded.webp") if project_logo is None else project_logo,
        brand_logo_light=(_asset_uri_light("asterisk-labs.svg")
                          if brand_logo is None else brand_logo),
        project_logo_light=(_asset_uri_light("clouds-decoded.webp")
                            if project_logo is None else project_logo),
        tile_colors_modes=tile_modes,
    )
    output_path.write_text(html)
    logger.info(f"Report written to {output_path}")
    return output_path
