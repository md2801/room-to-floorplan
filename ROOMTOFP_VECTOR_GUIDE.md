# RoomToFP Vector Floorplan Overhaul

The implementation is inspired by RoomFormer in one important design choice: instead of relying only on U-Net segmentation masks, the upgraded path uses all panorama RGB-D views to build a scene-level BEV observation and supervises room polygons from `annotation_3d.json`.

## What changed

The old pipeline predicted two raster masks:

- structure/walls
- furniture

That can produce borders, but it often struggles to create clean architectural floorplans because the model has to learn both geometry and vectorization indirectly.

The RoomToFP vector path changes this into:

1. collect all panorama RGB-D evidence for a scene,
2. build a top-down observation tensor,
3. parse polygon room supervision from `annotation_3d.json`,
4. train `RoomToFPPolygonNet` to predict room polygons directly,
5. keep furniture as a separate semantic mask branch.

## Dataset fact-check

The main inspiration point from RoomFormer is its use of Structured3D panorama data. It does not need `perspective/full` or `perspective/empty` for its main Structured3D floorplan reconstruction path. The important difference was not panorama vs perspective. The important difference was the target representation: polygon floorplans instead of only pixel masks.

## New project-facing files

```text
room_to_floorplan_project/
  render_structured3d_roomtofp_floorplan.py
  prepare_roomtofp_dataset.py
  train_roomtofp_stru3d.py
  eval_roomtofp_stru3d.py
  room_to_floorplan/roomtofp_polygon_model.py
```

No added file/function/class uses the RoomFormer naming scheme. Comments mark where the approach is inspired by RoomFormer.

## 1. Prepare RoomToFP polygon training data

From unzipped Structured3D panorama scenes:

```bash
python prepare_roomtofp_dataset.py \
  --structured3d-root /path/to/Structured3D_panorama_root \
  --output data/roomtofp_polygon \
  --grid-size 512 \
  --variants full
```

Or directly from panorama subset zips:

```bash
python prepare_roomtofp_dataset.py \
  --panorama-zips /path/to/Structured3D_panorama_00_subset_0-99.zip \
  --output data/roomtofp_polygon \
  --grid-size 512 \
  --variants full
```

## 3. Train the RoomToFP polygon model

```bash
python train_roomtofp_stru3d.py \
  --dataset-root data/roomtofp_polygon \
  --output-dir output/roomtofp_polygon \
  --epochs 50 \
  --batch-size 2 \
  --device cuda
```

The model saves:

```text
output/roomtofp_polygon/best.pt
output/roomtofp_polygon/last.pt
output/roomtofp_polygon/history.json
```

## 4. Evaluate

```bash
python eval_roomtofp_stru3d.py \
  --dataset-root data/roomtofp_polygon \
  --checkpoint output/roomtofp_polygon/best.pt \
  --output output/roomtofp_eval.json
```

## Furniture detection

The code treats furniture as a separate semantic-mask branch because floorplan polygon prediction and furniture detection are different tasks. For Structured3D scenes, furniture supervision is generated from semantic labels in the `full` panorama variant.
