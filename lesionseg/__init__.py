"""lesionseg: outline EAE lesions from nuclear density + Pu.1+ myeloid cells.

Pipeline (see lesionseg.pipeline.run_sample):

    CZI scan ──► tissue mask ──► tiled nucleus segmentation ──► per-nucleus features
              ──► Pu.1+ classification ──► density maps ──► lesion score
              ──► lesion mask ──► zones (core / rim / peri-lesion / deep / distal)
              ──► per-cell zone + signed distance ──► exports (parquet, GeoJSON, OME-TIFF, LMD)
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lesionseg")
except PackageNotFoundError:  # editable install without metadata
    __version__ = "0.1.0"

ZONE_CODES = {
    "background": 0,  # outside tissue
    "distal": 1,      # tissue, > peri_width from any lesion  (NAWM / control)
    "peri": 2,        # outside lesion, within peri_width of the lesion edge
    "rim": 3,         # inside lesion, within rim_width of the lesion edge
    "core": 4,        # inside lesion, deeper than rim_width
    "deep": 5,        # outside the peri band, a further deep_width outward ("additional step" for DVP)
    "meninges": 6,    # inside the section outline but outside the parenchyma mask (surface / meninges)
}
ZONE_NAMES = {v: k for k, v in ZONE_CODES.items()}
