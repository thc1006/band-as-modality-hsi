#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for the EMIT region table and acquisition-ID parsing.

Kept deliberately dependency-free (stdlib only). The download scripts need `earthaccess`, but the
ANALYSIS side must be able to read the declared bounding boxes without it: if importing the region
table could fail because a download-only dependency is missing, the geographic crop in
phase8F_multi.extract() would silently switch off and we would be back to analysing whole granules
that merely intersect their box. Bounding boxes are (lon_west, lat_south, lon_east, lat_north).
"""
import os
import re

# Regions chosen to SPAN NDVI: very-high (dense forest) -> mid (savanna/crop/grass) -> low (desert).
# The actual NDVI is measured downstream; these are only search boxes, and earthaccess returns any
# granule that INTERSECTS one, so pixels must still be cropped to the box before a region name means
# anything. (Measured example: the granule returned for `sumatra` had 0.0% of its pixels inside this
# box and sat over water.)
REGIONS = {
    # --- very high NDVI: dense tropical/temperate vegetation ---
    "amazon":        (-65, -6, -60, -1),
    "congo":         (18, -2, 24, 3),
    "borneo":        (110, -1, 116, 3),
    "sumatra":       (102.4, -2, 103.9, -0.5),   # re-declared to south-central Sumatran LAND (Jambi);
                                                 # the old (99,-3,104,2) box's best EMIT granule grazed
                                                 # the ocean edge (0% land in-box) -- this tight box bounds
                                                 # an interior granule footprint (102.6,-1.8..103.7,-0.7)
    "us_midwest":    (-96, 39, -90, 43),      # corn belt (season-dependent)
    # --- mid NDVI: savanna / cerrado / grassland / cropland ---
    "cerrado_br":    (-50, -16, -45, -11),
    "sahel":         (0, 13, 8, 16),
    "e_africa_sav":  (34, -3, 38, 1),
    "pampas_ar":     (-63, -36, -58, -32),
    # --- low-mid NDVI: shrub / semi-arid ---
    "australia_mulga": (140, -30, 146, -25),
    # --- very low NDVI: desert (positive-anchor confirmation) ---
    "sahara":        (10, 22, 18, 27),
    "arabian":       (44, 19, 50, 24),
    # --- 3 already-downloaded granules keyed by the CamelCase NAME phase8F_multi assigns their dirs
    #     (data/emit -> India_crop, data/emit_Africa_veg, data/emit_NAmerica_arid). Without a matching
    #     REGIONS key region_bbox_for() returned None and each was analysed as a WHOLE granule with
    #     inside_pct=NaN (biome label unverified). Boxes derived from each granule's OWN /location
    #     lat/lon footprint (min/max), so the crop is guaranteed inside the scene. ---
    "India_crop":    (73.15, 21.32, 74.27, 22.34),   # Gujarat cropland
    "Africa_veg":    (19.04, -5.49, 20.02, -4.39),    # Congo-basin vegetation
    "NAmerica_arid": (-115.92, 33.34, -114.75, 34.43),  # Mojave / SW-US arid
}


# Strict, anchored: the WHOLE basename must be a real EMIT L2A V001 product name. A `re.search` for
# the id substring accepted anything containing it -- `garbage_001_20240101T000000_1111111_001.tmp`
# parsed as a valid acquisition -- so a partial download, an editor backup or an unrelated file could
# be adopted into the triple. Anchoring is what makes "this file IS product X of acquisition Y" a
# checkable statement rather than a guess.
_EMIT_L2A_NAME = re.compile(
    r"^EMIT_L2A_(?P<product>RFL|RFLUNCERT|MASK)_001_(?P<gid>\d{8}T\d{6}_\d{7}_\d{3})\.nc$")


def _basename(path):
    """Filename component of a local path OR a URL, with any query string / fragment removed.

    Needed because the same parser is applied to `earthaccess` data links, and a presigned S3 URL
    ends in `...RFL_001_<id>.nc?X-Amz-Signature=...`. Anchoring the pattern to the whole basename
    without stripping that would reject every real download URL -- and because the caller now
    refuses a candidate whose id will not parse, the downloader would silently skip everything."""
    s = str(path).split("?", 1)[0].split("#", 1)[0]
    return os.path.basename(s)


def emit_product_and_id(path):
    """(product, acquisition_id) for a strictly-valid EMIT L2A V001 filename, else (None, None)."""
    m = _EMIT_L2A_NAME.fullmatch(_basename(path))
    return (m.group("product"), m.group("gid")) if m else (None, None)


def emit_granule_id(path):
    """The EMIT acquisition ID (timestamp_orbit_scene, e.g. '20220810T034103_2222203_001') for a
    strictly-valid EMIT L2A V001 filename; None otherwise.

    CALLER WARNING: None means "not a recognisable EMIT product", and `None == None` is True. Never
    compare two parsed ids without first rejecting None -- two unparseable filenames would otherwise
    "match" each other and be paired as one acquisition. Use `same_acquisition()` below."""
    return emit_product_and_id(path)[1]


def same_acquisition(a, b):
    """True only if both paths parse AND name the same acquisition. Exists so the None==None trap
    cannot be reintroduced by an innocent-looking `==` at a call site."""
    ga, gb = emit_granule_id(a), emit_granule_id(b)
    return ga is not None and ga == gb
