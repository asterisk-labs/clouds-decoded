"""Multitemporal albedo estimation over land.

Fits a tile-level cluster + gated temporal kernel model to the full clear-sky
time series of a project's scenes, then evaluates it at each scene's sensing
time to pre-populate per-scene ``albedo.tif`` outputs before the ordinary
per-scene project run.
"""
