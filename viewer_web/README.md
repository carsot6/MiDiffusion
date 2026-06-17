# MiDiffusion Local Scene Viewer

Local web viewer for MiDiffusion predictions with drag-and-drop loading.

## What it does

- Loads `results.pkl` directly in the browser app (via local Flask API).
- Builds a 3D scene with floor, generated walls, lights, and furniture placements.
- Renders furniture as:
  - real GLB assets (if mapped), and/or
  - fallback semantic boxes.
- Lets you switch scenes and toggle walls/floor/assets/boxes/grid.

## Run locally

From repo root:

```bash
python scripts/run_local_scene_viewer.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Input options

### 1) Drag-and-drop `results.pkl`

Drop a MiDiffusion `results.pkl` file exported by `scripts/generate_results.py`.

Optional: paste an asset map in JSON (`viewer_web/asset_map.example.json` shows the format), so labels can resolve to real GLB URLs.

### 2) Drag-and-drop prebuilt manifest JSON

You can pre-export a manifest file and load it directly:

```bash
python scripts/export_viewer_manifest.py \
  output/predicted_results/gpc_smoke_pred/results.pkl \
  --output-file output/predicted_results/gpc_smoke_pred/viewer_manifest.json
```

Then drag `viewer_manifest.json` into the web UI.

## Asset map format

```json
{
  "by_label": {
    "dining_chair": { "asset_url": "https://.../chair.glb" }
  },
  "by_name": {
    "dining_chair_003": { "asset_url": "https://.../specific.glb" }
  },
  "default_asset_url": "https://.../fallback.glb"
}
```

- `by_name` overrides `by_label`.
- `scale` and `offset` are optional per mapping.
- If no asset is mapped, the object is shown as a box.

## Notes

- This app is local-only and intended for inspection/debug workflows.
- `results.pkl` parsing occurs on your local machine; files are not uploaded externally.
- If model import/unpickle fails, ensure your environment has MiDiffusion + ThreedFront dependencies installed.
