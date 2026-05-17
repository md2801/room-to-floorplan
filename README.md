# Room to Floorplan Generator

This project converts Structured3D room scene data into top-down floorplan representations. It includes the original baseline pipeline and an upgraded **RoomToFP vector/polygon pipeline** for cleaner outputs.

## Project Overview

The project supports two main workflows:

1. **Baseline segmentation pipeline**
   - Builds BEV training pairs from Structured3D panorama/perspective data.
   - Trains U-Net-style models to predict wall and furniture masks.
   - Can generate preview images and SVG floorplan outputs.

2. **RoomToFP vector pipeline**
   - Uses scene-level panorama aggregation.
   - Builds 8-channel top-down observations.
   - Uses polygon supervision from Structured3D annotations.
   - Trains a CNN–Transformer model to predict room polygons, wall masks, opening masks, and furniture masks.

The RoomToFP vector branch is inspired by RoomFormer’s design idea of using scene-level aggregation and polygon-based supervision instead of relying only on raster segmentation masks.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the Structured3D dataset. The RoomToFP branch mainly uses panorama scenes. Each scene contains an `annotation_3d.json` file and a `2D_rendering` folder.

## Command Examples
### Testing 1 Scene
`
python prepare_roomtofp_dataset.py \
  --structured3d-root /path/to/Structured3D \
  --output output/data/roomtofp_polygon_v5 \
  --grid-size 512 \
  --variants full \
  --furniture-source auto
`

### Preparing the Dataset
`
python prepare_roomtofp_dataset.py \
  --structured3d-root /path/to/Structured3D \
  --output output/data/roomtofp_polygon_v5 \
  --grid-size 512 \
  --variants full \
  --furniture-source auto
`

This script creates the following:
`
samples/,
train.txt,
val.txt,
test.txt,
splits.json,
metadata.json
`

### Training the Model
`
python train_roomtofp_stru3d.py \
  --dataset-root output/data/roomtofp_polygon_v5 \
  --output-dir output/artifacts/roomtofp_polygon_v5 \
  --epochs 80 \
  --batch-size 4 \
  --lr 1e-4 \
  --device cuda
`

### Evaluating the Model
`
python eval_roomtofp_stru3d.py \
  --dataset-root output/data/roomtofp_polygon_v5 \
  --checkpoint output/artifacts/roomtofp_polygon_v5/best.pt \
  --split test \
  --output output/artifacts/roomtofp_eval_test.json
`

### Running Inference
`
python infer_roomtofp_from_scene.py \
  --scene-dir /path/to/Structured3D/scene_00005 \
  --checkpoint output/artifacts/roomtofp_polygon_v5/best.pt \
  --output-dir output/roomtofp_infer_scene_00005 \
  --variants full \
  --grid-size 512
`

### Visualizing
`
python visualize_roomtofp_predictions.py \
  --dataset-root output/data/roomtofp_polygon_v5 \
  --checkpoint output/artifacts/roomtofp_polygon_v5/best.pt \
  --output-dir output/artifacts/roomtofp_prediction_viz \
  --split test \
  --num-samples 12
`

## Outputs
`
floorplan_preview.png,
floorplan.svg,
walls_mask.png,
opening_mask.png,
furniture_mask.png,
metadata.json,
best.pt,
last.pt,
history.json,
evaluation JSON files,
prediction visualisations
`