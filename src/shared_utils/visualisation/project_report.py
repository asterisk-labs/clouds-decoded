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

_PALETTE = [
    '#4fc3f7', '#ffb74d', '#81c784', '#ce93d8',
    '#ff8a65', '#4db6ac', '#e57373', '#fff176',
    '#f06292', '#9575cd', '#a1887f', '#90a4ae',
    '#aed581', '#7986cb', '#dce775', '#4dd0e1',
]

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


def _extract_date(scene_id: str) -> Optional[str]:
    m = re.search(r"_(\d{4})(\d{2})(\d{2})T", scene_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _tile_colors(scenes: list[dict]) -> dict[str, str]:
    tiles = sorted({s['tile_id'] for s in scenes if s.get('tile_id')})
    return {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(tiles)}


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
                  dpi: int = 96, draw_markers: bool = True
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
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#0d1b2a')
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#1e2d1e')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.5, edgecolor='#556655')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),
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
                 project_logo: str = '') -> str:

    regions = regions or {}
    # Optionally hyperlink the first title segment (before the first " · ") to project_url.
    if project_url:
        _p = project_name.split(' · ', 1)
        title_html = (f'<a href="{project_url}" target="_blank" rel="noopener">{_p[0]}</a>'
                      + (f' · {_p[1]}' if len(_p) > 1 else ''))
    else:
        title_html = project_name
    data_json = json.dumps(scenes)
    tile_colors_json = json.dumps(tile_colors)
    map_points_json = json.dumps(map_points or [])
    fig_base_json = json.dumps(fig_base)   # if set, layers are basenames → URL = fig_base+scene_id+'/'+name

    # Tile checkboxes — one per tile, toggle its dots on the map + line on the
    # timeseries. Labelled "TILEID (Region)". All checked by default.
    def _tile_label(t):
        rg = regions.get(t, '')
        return f'{t} <span class="rgn">({rg})</span>' if rg else t
    _sorted_tiles = sorted(tile_colors.items())
    _default_tile = _sorted_tiles[0][0] if _sorted_tiles else None
    tile_tabs_html = ''.join(
        f'<div class="tchk" data-tile="{t}">'
        f'<input type="checkbox" data-tile="{t}"{" checked" if t == _default_tile else ""} '
        f'title="show / hide this tile\'s line on the chart">'
        f'<span class="tab-dot" style="background:{c}"></span>'
        f'<span class="tname" data-tile="{t}" title="view this tile in the browser below">{_tile_label(t)}</span></div>'
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

    logo_html = (f'<img id="logo" src="{brand_logo}" alt="asterisk labs">'
                 if brand_logo else '')
    logo2_html = (f'<img id="logo2" src="{project_logo}" alt="clouds decoded">'
                  if project_logo else '')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{project_name} — clouds-decoded report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #111; color: #ddd; }}
#header {{ position: relative; padding: 10px 16px 8px; background: #1a1a2e; border-bottom: 1px solid #333; display: flex; flex-direction: column; align-items: center; gap: 7px; flex-shrink: 0; }}
#header h1 {{ font-size: 1.5rem; font-weight: 600; letter-spacing: 0.2px; }}
#brand {{ display: flex; justify-content: center; align-items: center; }}
#header-controls {{ display: flex; align-items: center; gap: 18px; }}
#daterange {{ display: flex; align-items: center; gap: 6px; font-size: 0.78rem; color: #888; }}
.info-i {{ display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%; border: 1px solid #4a6fa5; color: #7eb8f7; font: italic 700 0.62rem Georgia, serif; cursor: help; margin-left: 6px; flex-shrink: 0; }}
.info-i:hover {{ background: #1e3050; color: #cfe0f5; }}
#daterange .dr-label {{ text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.68rem; color: #667; }}
#daterange .dr-sep {{ color: #556; }}
#daterange input[type=date] {{ background: #1e1e2e; color: #ccc; border: 1px solid #333; border-radius: 4px; padding: 2px 5px; font-size: 0.75rem; color-scheme: dark; }}
#daterange input[type=date]:hover {{ border-color: #4a6fa5; }}
#date-reset {{ background: #1e1e2e; color: #999; border: 1px solid #333; border-radius: 4px; padding: 2px 9px; font-size: 0.72rem; cursor: pointer; }}
#date-reset:hover {{ background: #2e2e4e; color: #ccc; }}
#header .meta {{ font-size: 0.8rem; color: #888; }}
#logo {{ position: absolute; left: 16px; top: 10px; height: calc(100% - 18px); width: auto; }}
#logo2 {{ position: absolute; right: 16px; top: 10px; height: calc(100% - 18px); width: auto; }}
#header h1 a {{ color: inherit; text-decoration: none; }}
#header h1 a:hover {{ color: #7eb8f7; text-decoration: underline; }}
/* Sticky top bar (header + toolbars) so toggles stay reachable while the page scrolls */
#topbar {{ position: sticky; top: 0; z-index: 30; background: #111; flex-shrink: 0; }}
/* per-section "i" badges positioned over the map / chart */
.map-i, .chart-i {{ position: absolute; top: 6px; right: 6px; z-index: 6; margin: 0; background: rgba(18,18,28,0.85); }}
.chart-i {{ right: 12px; }}
/* per-plot height box */
.row-h {{ width: 44px; background: #12121c; color: #9db4d0; border: 1px solid #2c3346; border-radius: 3px; font-size: 0.66rem; padding: 1px 3px; margin-left: 5px; }}
.row-h:hover {{ border-color: #4a6fa5; }}
/* Fixed top: map + chart */
#top-panels {{ display: flex; gap: 8px; padding: 8px; flex-shrink: 0; }}
#map-wrap {{ position: relative; flex-shrink: 0; height: 200px; overflow: hidden; border-radius: 6px; cursor: grab; }}
#map-wrap.grabbing {{ cursor: grabbing; }}
#map-inner {{ position: relative; transform-origin: 0 0; will-change: transform; }}
#map-bg {{ display: block; height: 200px; width: auto; }}
#map-canvas {{ position: absolute; top: 0; left: 0; pointer-events: auto; }}
#chart-wrap {{ flex: 1; min-width: 0; position: relative; }}
#chart-canvas {{ display: block; width: 100%; cursor: pointer; }}
/* Tile + plot + layer toggle checkboxes */
#tab-bar, #var-bar, #layer-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px; padding: 4px 8px; background: #151520; border-bottom: 1px solid #333; flex-shrink: 0; overflow-x: auto; }}
#tab-bar {{ border-top: 1px solid #222; }}
.bar-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.5px; color: #666; margin-right: 6px; }}
.tchk {{ padding: 4px 12px; border-radius: 4px; border: 1px solid #333; background: #1e1e2e; color: #bbb; font-size: 0.78rem; white-space: nowrap; display: flex; align-items: center; gap: 5px; user-select: none; }}
.tchk input {{ margin: 0; cursor: pointer; accent-color: #4a6fa5; }}
.tchk .tname {{ cursor: pointer; }}
.tchk .tname:hover {{ color: #fff; text-decoration: underline; }}
.tchk.focused {{ border-color: #4a6fa5; box-shadow: 0 0 0 1px #4a6fa5 inset; background: #1e3050; }}
.tchk.focused .tname {{ color: #7eb8f7; }}
.tchk .rgn {{ color: #777; }}
.tab-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
/* Scene browser */
#scene-panel {{ display: flex; align-items: flex-start; gap: 10px; padding: 8px; }}
#nav-prev, #nav-next, #nav-play {{ flex-shrink: 0; width: 36px; height: 36px; font-size: 1.2rem; background: #1e1e2e; border: 1px solid #444; border-radius: 6px; color: #ccc; cursor: pointer; margin-top: 4px; }}
#nav-prev:hover, #nav-next:hover, #nav-play:hover {{ background: #2e2e4e; }}
#nav-play.playing {{ background: #2e1e1e; border-color: #664444; color: #ff8a65; }}
#speed-select {{ background: #1e1e2e; color: #888; border: 1px solid #333; border-radius: 4px; font-size: 0.72rem; padding: 2px 4px; margin-top: 4px; width: 52px; }}
#scene-info {{ flex-shrink: 0; width: 160px; font-size: 0.78rem; line-height: 1.7; }}
#si-date {{ font-size: 0.95rem; font-weight: 600; color: #7eb8f7; }}
#si-sub {{ font-size: 0.68rem; color: #666; margin-top: 1px; word-break: break-all; }}
#si-stats {{ margin-top: 6px; color: #aaa; }}
#si-stats b {{ color: #ccc; }}
#layer-grid {{ flex: 1; align-self: stretch; display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 6px; align-content: start; }}
.layer-thumb {{ text-align: center; }}
.layer-thumb.overview {{ grid-column: 1 / -1; }}
.layer-thumb.overview img {{ max-width: 50%; margin: 0 auto; }}
.layer-thumb img {{ width: 100%; height: auto; display: block; border: 2px solid #2a2a3a; border-radius: 4px; cursor: pointer; }}
.layer-thumb img:hover {{ border-color: #7eb8f7; }}
.layer-label {{ font-size: 0.8rem; color: #b9c2cf; font-weight: 600; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
#progress {{ font-size: 0.75rem; color: #666; }}
/* Lightbox */
#lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.93); z-index: 100; align-items: center; justify-content: center; }}
#lightbox.active {{ display: flex; }}
#lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 4px; }}
#lb-close {{ position: fixed; top: 14px; right: 18px; font-size: 1.8rem; color: #bbb; cursor: pointer; z-index: 101; }}
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
    </div>
  </div>

  <!-- Selection rows (kept on top, above the map / time series / images) -->
  <div id="tab-bar">
    <span class="bar-label">Tiles</span>
    <span class="info-i" title="Add or remove each tile's line on the time series.">i</span>
    {tile_tabs_html}
  </div>
  <div id="var-bar">
    <span class="bar-label">Plots</span>
    <span class="info-i" title="Choose which cloud-property statistics are plotted over time. The number box next to each variable sets that plot's height in pixels.">i</span>
  </div>
  <div id="layer-bar">
    <span class="bar-label">Layers</span>
    <span class="info-i" title="Choose which per-scene retrieval thumbnails appear in the scene browser below.">i</span>
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
const TILE_COLORS = {tile_colors_json};
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
  const bar = document.getElementById('layer-bar');
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

// Scene-browser focus (set by clicking a map dot or chart point) — independent
// of the visibility checkboxes.
function switchTile(tile) {{
  setPlaying(false);
  activeTile = tile || null;
  current = 0;
  document.querySelectorAll('#tab-bar .tchk').forEach(l =>
    l.classList.toggle('focused', l.dataset.tile === (tile || ''))
  );
  showScene(0);
}}

// Tile checkboxes toggle tile visibility on the map + timeseries (via hiddenTiles).
// Scope to #tab-bar so this does NOT also bind to the plot/layer checkboxes (which
// share the .tchk class) — otherwise toggling those would fire this handler too.
document.querySelectorAll('#tab-bar .tchk input').forEach(cb =>
  cb.addEventListener('change', () => {{
    if (cb.checked) hiddenTiles.delete(cb.dataset.tile);
    else hiddenTiles.add(cb.dataset.tile);
    drawChart();
    drawMapDots();
  }})
);

// Click a tile NAME to select it for viewing in the scene browser (independent of
// the checkbox, which only shows/hides it on the map + timeseries).
document.querySelectorAll('#tab-bar .tname').forEach(el =>
  el.addEventListener('click', () => switchTile(el.dataset.tile))
);

// ── Chart ────────────────────────────────────────────────────────────────────
const chartCanvas = document.getElementById('chart-canvas');
const chartCtx = chartCanvas.getContext('2d');
let chartPoints = [];
const DEFAULT_TILE = Object.keys(TILE_COLORS).sort()[0];                          // landing: one tile's line
let hiddenTiles = new Set(Object.keys(TILE_COLORS).filter(t => t !== DEFAULT_TILE));
let visibleVars = new Set([CHART_VARS[0].key]);   // landing: one line; users toggle more via Plots
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
  const bar = document.getElementById('var-bar');
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
  chartCanvas.width = W;
  chartCanvas.height = totalH;
  chartCanvas.style.height = totalH + 'px';

  const padL = 44, padR = 10, padT = 8, padB = 6;
  const cw = W - padL - padR;

  chartCtx.fillStyle = '#0d1117';
  chartCtx.fillRect(0, 0, W, totalH);

  // ── Legend ──────────────────────────────────────────────────────────────────
  legendItems = [];
  let lx = padL;
  chartCtx.font = '9px monospace';
  Object.entries(TILE_COLORS).forEach(([tid, color]) => {{
    const hidden = hiddenTiles.has(tid);
    const tw = chartCtx.measureText(tid).width;
    const itemW = 10 + 4 + tw + 10;
    chartCtx.globalAlpha = hidden ? 0.35 : 1;
    chartCtx.fillStyle = color;
    chartCtx.beginPath();
    chartCtx.arc(lx + 5, LEGEND_H / 2, 4, 0, Math.PI * 2);
    chartCtx.fill();
    chartCtx.fillStyle = hidden ? '#555' : '#ccc';
    chartCtx.textAlign = 'left';
    chartCtx.fillText(tid, lx + 12, LEGEND_H / 2 + 4);
    if (hidden) {{
      chartCtx.strokeStyle = '#555'; chartCtx.lineWidth = 1; chartCtx.globalAlpha = 0.5;
      chartCtx.beginPath(); chartCtx.moveTo(lx, LEGEND_H / 2); chartCtx.lineTo(lx + itemW - 8, LEGEND_H / 2); chartCtx.stroke();
    }}
    chartCtx.globalAlpha = 1;
    legendItems.push({{ tid, x: lx, w: itemW }});
    lx += itemW;
  }});

  // ── Separator ───────────────────────────────────────────────────────────────
  chartCtx.strokeStyle = '#333'; chartCtx.lineWidth = 1;
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

    const allVals = SCENES.map(s => s.stats[key]).filter(v => v != null);
    if (!allVals.length) return;
    const vMax = Math.max(...allVals) * 1.1 || 1;
    const ty = v => yOff + padT + ch - (v / vMax) * ch;

    if (vi > 0) {{
      chartCtx.strokeStyle = '#222'; chartCtx.lineWidth = 1;
      chartCtx.beginPath(); chartCtx.moveTo(0, yOff); chartCtx.lineTo(W, yOff); chartCtx.stroke();
    }}

    chartCtx.font = '8px monospace'; chartCtx.textAlign = 'right';
    chartCtx.fillStyle = '#666';
    chartCtx.fillText(vMax.toFixed(2), padL - 3, yOff + padT + 4);
    chartCtx.fillText('0', padL - 3, yOff + padT + ch);
    chartCtx.font = 'bold 10px system-ui'; chartCtx.fillStyle = '#bbb'; chartCtx.textAlign = 'left';
    chartCtx.fillText(label, padL + 4, yOff + padT + 4);

    chartCtx.strokeStyle = '#1e1e2e'; chartCtx.lineWidth = 1;
    chartCtx.beginPath(); chartCtx.moveTo(padL, yOff + padT + ch); chartCtx.lineTo(padL + cw, yOff + padT + ch); chartCtx.stroke();

    const byTile = {{}};
    SCENES.forEach((s, i) => {{
      const v = s.stats[key];
      if (v == null || !inRange(allDates[i])) return;   // date-range filter
      const tid = s.tile_id || '_';
      (byTile[tid] = byTile[tid] || []).push({{ i, x: tx(i), y: ty(v) }});
    }});

    Object.entries(byTile).forEach(([tid, pts]) => {{
      if (hiddenTiles.has(tid)) return;
      const color = TILE_COLORS[tid] || '#aaa';
      const isActiveTile = !activeTile || tid === activeTile;
      chartCtx.strokeStyle = color;
      chartCtx.globalAlpha = isActiveTile ? 0.4 : 0.12;
      chartCtx.lineWidth = 1;
      chartCtx.beginPath();
      pts.forEach((p, j) => j === 0 ? chartCtx.moveTo(p.x, p.y) : chartCtx.lineTo(p.x, p.y));
      chartCtx.stroke();
      pts.forEach(p => {{
        const isCurrent = p.i === currentGlobal;
        chartCtx.globalAlpha = isCurrent ? 1 : (isActiveTile ? 0.7 : 0.18);
        chartCtx.fillStyle = color;
        chartCtx.beginPath();
        chartCtx.arc(p.x, p.y, isCurrent ? 4 : 1.5, 0, Math.PI * 2);
        chartCtx.fill();
        if (isCurrent) {{ chartCtx.strokeStyle = 'white'; chartCtx.lineWidth = 1.5; chartCtx.stroke(); }}
        if (isActiveTile && vi === 0) chartPoints.push({{ i: p.i, x: p.x, y: p.y, tid }});
      }});
    }});
    chartCtx.globalAlpha = 1;
  }});

  // ── Date axis ─────────────────────────────────────────────────────────────
  const yAxis = bodyBottom;
  chartCtx.strokeStyle = '#333'; chartCtx.lineWidth = 1;
  chartCtx.beginPath(); chartCtx.moveTo(padL, yAxis); chartCtx.lineTo(padL + cw, yAxis); chartCtx.stroke();
  chartCtx.font = '9px system-ui'; chartCtx.textAlign = 'center';
  for (const {{ t, lab }} of dateTicks(tMin, tMax)) {{
    const x = txT(t);
    if (x < padL - 1 || x > padL + cw + 1) continue;
    chartCtx.strokeStyle = '#444';
    chartCtx.beginPath(); chartCtx.moveTo(x, yAxis); chartCtx.lineTo(x, yAxis + 4); chartCtx.stroke();
    chartCtx.fillStyle = '#8a8a99'; chartCtx.fillText(lab, x, yAxis + 15);
  }}
}}

chartCanvas.onclick = e => {{
  const r = chartCanvas.getBoundingClientRect();
  const mx = (e.clientX - r.left) * (chartCanvas.width / r.width);
  const my = (e.clientY - r.top) * (chartCanvas.height / r.height);

  // Legend area — toggle tile visibility
  if (my < LEGEND_H) {{
    for (const item of legendItems) {{
      if (mx >= item.x && mx < item.x + item.w) {{
        if (hiddenTiles.has(item.tid)) hiddenTiles.delete(item.tid);
        else hiddenTiles.add(item.tid);
        const cb = document.querySelector('.tchk input[data-tile="' + item.tid + '"]');
        if (cb) cb.checked = !hiddenTiles.has(item.tid);   // keep checkbox in sync
        drawChart();
        drawMapDots();
        return;
      }}
    }}
    return;
  }}

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

function initMap() {{
  if (!mapBg || !MAP_POINTS.length || !mapWrap) return;
  mapCanvas.width = mapWrap.clientWidth;    // canvas covers the viewport, unscaled
  mapCanvas.height = mapWrap.clientHeight;
  drawMapDots();
}}

// Dots live on a viewport-fixed canvas (NOT inside the transformed image), so they
// keep a constant on-screen size and stay crisp at any zoom.
function drawMapDots() {{
  if (!mapCanvas.width) return;
  const ctx = mapCanvas.getContext('2d');
  ctx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  const ds = mapBaseScale() * mapScale;
  const sc = activeScenes();
  const currentGlobal = sc.length ? SCENE_IDX[sc[current]?.scene_id] : -1;

  MAP_POINTS.forEach((p, i) => {{
    if (!p) return;
    const tid = SCENES[i].tile_id;
    // every tile's dots are always drawn — tile toggles only affect the chart lines
    const x = mapTx + p.x * ds, y = mapTy + p.y * ds;
    if (x < -8 || y < -8 || x > mapCanvas.width + 8 || y > mapCanvas.height + 8) return;
    const isActiveTile = !activeTile || tid === activeTile;
    const isCurrent = i === currentGlobal;
    ctx.globalAlpha = isCurrent ? 1 : (isActiveTile ? 0.7 : 0.3);
    ctx.fillStyle = TILE_COLORS[tid] || '#aaa';
    ctx.beginPath();
    ctx.arc(x, y, isCurrent ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = isCurrent ? 'white' : 'rgba(255,255,255,0.5)';
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
  const MIN = 1, MAX = 12;
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

  const statLines = Object.entries({{
    cloud_frac: 'Cloud frac', tau__mean: 'τ mean',
    r_eff_liq__p050: 'r_eff liq p50', ice_liq_ratio__mean: 'Ice/liq',
  }}).filter(([k]) => s.stats[k] != null)
    .map(([k, lbl]) => `<b>${{lbl}}:</b> ${{s.stats[k].toFixed(3)}}`);
  document.getElementById('si-stats').innerHTML = statLines.join('<br>');

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
  sortedLayers
    .filter(path => visibleLayers.has(path.split('/').pop().replace('.png', '')))
    .forEach(path => {{
    const name = path.split('/').pop().replace('.png', '');
    const div = document.createElement('div');
    div.className = 'layer-thumb' + (name === 'overview' ? ' overview' : '');
    const img = document.createElement('img');
    const src = layerURL(s, path);
    img.src = src;
    img.onclick = () => openLightbox(src);
    const lbl = document.createElement('div');
    lbl.className = 'layer-label';
    lbl.textContent = labelFromPath(path);
    div.appendChild(img); div.appendChild(lbl);
    grid.appendChild(div);
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

// ── Init ──────────────────────────────────────────────────────────────────────
// Select the first tile by default
const firstTile = Object.keys(TILE_COLORS).sort()[0];
if (firstTile) switchTile(firstTile);
else showScene(0);
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
    """Generate an HTML report with map, timeseries and scene thumbnail browser."""
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
    tile_colors = _tile_colors(scenes)

    logger.info("Generating map...")
    map_png, map_points = _generate_map(scenes, tile_colors)

    html = _render_html(
        scenes, tile_colors, map_png, map_points, project_dir.name,
        brand_logo=_asset_uri("asterisk-labs.png") if brand_logo is None else brand_logo,
        project_logo=_asset_uri("clouds-decoded.webp") if project_logo is None else project_logo,
    )
    output_path.write_text(html)
    logger.info(f"Report written to {output_path}")
    return output_path
