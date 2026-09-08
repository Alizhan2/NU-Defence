# Google Earth Engine integration

GeoWatch uses Earth Engine as an optional imagery source, not as the detection
model. The offline two-image workflow remains available without Google access.

## Intended pipeline

1. The analyst draws an area of interest and selects `before` and `after` dates.
2. Earth Engine queries `COPERNICUS/S2_SR_HARMONIZED` and applies a cloud mask.
3. The nearest usable scenes are reprojected and exported as aligned tiles.
4. GeoWatch runs the same detector for both dates.
5. Same-class detections are matched by normalized spatial overlap.
6. Appeared and disappeared objects enter human review; they are not treated as
   autonomous conclusions.

## Configuration still required

1. Create or select a Google Cloud project.
2. Enable and register the project for the Earth Engine API.
3. Install the optional `earthengine-api` package.
4. Authenticate outside GeoWatch with `earthengine authenticate`; the app only
   initializes the client with the configured project.
5. Set `GOOGLE_CLOUD_PROJECT` or `EARTHENGINE_PROJECT` to that project ID.

GeoWatch never calls `ee.Authenticate()` and never stores Google credentials.
The adapter imports `ee` lazily, so the offline workflow and tests do not depend
on Google services.

## Adapter API

```python
from datetime import date
from src.earth_engine import EarthEngineAdapter, GeoArea, ScenePairRequest

source = EarthEngineAdapter()
print(source.status())  # local-only check; no Google request
source.initialize()     # uses EARTHENGINE_PROJECT / GOOGLE_CLOUD_PROJECT
pair = source.search_scene_pair(ScenePairRequest(
    aoi=GeoArea.from_bbox(56.8, 50.1, 57.0, 50.3),
    before_date=date(2026, 7, 1),
    after_date=date(2026, 8, 1),
    search_window_days=14,
    max_cloud_percent=25,
))
```

`search_scene_pair()` makes two compact metadata queries and selects the pair
closest to the requested dates while enforcing chronological order. Non-exact
dates are returned in `pair.warnings` and should be shown to the analyst.

The adapter deliberately stops at scene discovery. Exporting aligned RGB tiles
needs a separate bounded job with CRS, resolution, quota, size and audit
controls; never request an unbounded AOI directly from the UI.

Sentinel-2 is suitable for a free competition demonstration, but its 10 m
visible-band resolution and roughly five-day revisit do not support a guaranteed
daily view of small buildings. A production version should support a licensed
high-resolution provider behind the same imagery-source interface.
