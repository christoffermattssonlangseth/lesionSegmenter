# Meninges annotation – protocol

**Why.** Today the meninges are placed by a rule (outer 10 µm of the surface plus very compact infiltrate
connected to it). Nothing checks whether that rule is right, and it decides which cells are collected. Hand
annotations in a few boxes let us (1) measure how good the rule is and (2) train a model that learns the
meninges from nuclear shape, alignment, density and markers – and test it on animals it has not seen.

**Effort.** 8 sections (every animal covered), 31 boxes of 300 × 300 µm, roughly 3–5 min per box.

## What to annotate

The boxes are in `results/meninges_annotation/` (one `<slide>_MENINGES_ROI.geojson` per slide, preview PNGs
per section, list in `targets.csv`). Each box is labelled with why it was picked:

| kind | what is there | what we learn |
|---|---|---|
| `core_at_surface` | a manual lesion core within 50 µm of the surface | where meninges stop and subpial lesion starts – decides what is collected |
| `swollen` | the most cellular stretch of meninges in the section | how deep swollen meninges really reach |
| `internal` | meninges-like tissue > 80 µm deep (fissure, flap, root) | whether these are meninges |
| `bare` | surface with (almost) no meningeal nuclei | tear / lost pia vs thin meninges |
| `typical` | an ordinary stretch | the reference |

Per slide, three shapes layers:

1. **`MENINGES_ROI`** – the boxes. Keep the ones you annotated, **delete the ones you skip** (a kept box
   with nothing drawn means "no meninges here"). You may add your own boxes anywhere.
2. **`MENINGES`** – polygons around **all** meningeal tissue inside each kept box: the thin lining,
   swollen infiltrate in the subarachnoid space, meninges in fissures. Polygons may run past the box edge
   (only cells inside a box are used). Inside a box, everything not in `MENINGES` counts as "not meninges"
   – so annotate each box completely.
3. **`MENINGES_UNSURE`** (optional) – anything you cannot call (e.g. infiltrate continuous between
   meninges and lesion with no visible border). These cells are left out of training and testing.

Please decide once and note: are perivascular cuffs that enter the cord from the surface meninges? (Either
is fine for the model – it just has to be consistent.)

Draw on the **slide named in the file** (one replicate per section, chosen to balance the slides). The
previews show the raw image only, without our current boundary, so the annotation is independent of it.

## In napari (same session as the CORE annotation)

```python
import json, numpy as np, napari
slide_name = "20260917_CML_metal_scene1"        # one slide at a time
Interactive(sdata_dict[slide_name])
v = napari.current_viewer()

fc = json.load(open(f"{slide_name}_MENINGES_ROI.geojson"))
boxes = [np.array([[y, x] for x, y in f["geometry"]["coordinates"][0][:4]]) for f in fc["features"]]
v.add_shapes(boxes, shape_type="rectangle", name="MENINGES_ROI", edge_color="yellow", face_color="transparent",
             edge_width=6, features={"target_id": [f["properties"]["target_id"] for f in fc["features"]]},
             text={"string": "{target_id}", "color": "yellow", "size": 12})
v.add_shapes(name="MENINGES", ndim=2, edge_color="cyan", face_color=[0, 1, 1, 0.25])
v.add_shapes(name="MENINGES_UNSURE", ndim=2, edge_color="magenta", face_color=[1, 0, 1, 0.25])
```

Check that the yellow boxes land where the preview PNGs show them (on the surface). If they are offset or
scaled, stop and tell us – the coordinate system differs from the one the CORE polygons were saved in.

Save when done (one GeoJSON per layer, scene px – nothing is written into the SpatialData store):

```python
import geopandas as gpd
from shapely.geometry import Polygon

for name in ("MENINGES_ROI", "MENINGES", "MENINGES_UNSURE"):
    polys = [Polygon([(p[-1], p[-2]) for p in s]) for s in v.layers[name].data]   # napari (y, x) -> (x, y)
    if polys:
        gpd.GeoDataFrame(geometry=polys).to_file(f"{slide_name}_{name}.geojson", driver="GeoJSON")
```

(Writing them as shapes layers into the slide's `.zarr` instead works too – same layer names.)

## Afterwards (our side)

```bash
conda run -n lesionseg python scripts/train_meninges_model.py --annot-dir <folder with the GeoJSON files>
```

reports, per box kind and held out by animal, how well the current rule and the model agree with the
annotation (`results/meninges_model/`), saves the model (`models/meninges_hgb.joblib`) and writes
`meninges_pred.parquet` (probability per cell) next to every scene's `cells.parquet`.
