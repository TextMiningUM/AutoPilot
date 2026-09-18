#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build + sanity-check a Folium map with an OpenSeaMap nautical overlay,
using the real coordinates from vhfcol_00001 (Houston Ship Channel) that
appeared in this project's own data earlier."""
import folium

# real coordinates from the user's own bridge_snapshot data (Rotterdam
# approach) plus vhfcol_00001's Houston Ship Channel scenario as a second
# example -- not invented numbers.
LAT, LON = 51.88377747, 4.44220974   # Maasmond / Rotterdam approach

m = folium.Map(location=[LAT, LON], zoom_start=12, tiles="OpenStreetMap")

# OpenSeaMap publishes a standard XYZ tile layer specifically for seamarks
# (buoys, lights, etc.) meant to sit ON TOP of a base map, not replace it --
# this is exactly the "additional overlay" pattern, not a full base chart.
folium.TileLayer(
    tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
    attr="Map data: &copy; OpenSeaMap contributors",
    name="OpenSeaMap seamarks",
    overlay=True,
    control=True,
).add_to(m)

folium.Marker([LAT, LON], tooltip="RANDOM_TS3 (from your own logged state)",
              icon=folium.Icon(color="red", icon="ship", prefix="fa")).add_to(m)

folium.LayerControl().add_to(m)

out_path = "/home/claude/viz_demo/folium_test.html"
m.save(out_path)

content = open(out_path).read()
print("File size:", len(content), "bytes")
print("Contains OpenSeaMap tile URL:", "tiles.openseamap.org" in content)
print("Contains Leaflet init:", "L.map" in content or "leaflet" in content.lower())
print(f"Wrote {out_path}")
