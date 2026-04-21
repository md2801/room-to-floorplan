# Room to Floorplan Generator

End-to-end baseline pipeline for your university project:

1. **Structured3D data ingestion**
   - supports `perspective/full` scenes
   - supports `panorama/full` scenes
2. **Training pair generation**
   - converts Structured3D depth + semantics + layout into top-down BEV tensors
   - panorama is used for full-room supervision
   - perspective is used for realistic view-conditioned supervision
3. **Model training**
   - small U-Net style floorplan refiner trained on generated BEV maps
4. **Video inference**
   - takes a user-uploaded room/apartment walkthrough video
   - extracts frames
   - reconstructs a sparse 3D point cloud from multi-view geometry
   - builds a top-down observation map
   - optionally refines it using the trained model
   - generates wall mask, furniture mask, combined preview, SVG floorplan, and point cloud PLY

This is a strong **project baseline** designed to be understandable and modifiable. It is not a drop-in reproduction of RoomFormer, but it follows the same core idea of predicting a floorplan from a top-down density/occupancy representation.

## Install

```bash
pip install -r requirements.txt
```

## Build training pairs from your two scene zips

```bash
python build_training_data.py \
  --perspective-zip /path/to/scene_00000_perspective.zip \
  --panorama-zip /path/to/scene_00000_panorama.zip \
  --output-dir data/bev_pairs \
  --grid-size 512
```

## Train the model

```bash
python train_floorplan_model.py \
  --data-dir data/bev_pairs \
  --epochs 20 \
  --batch-size 4 \
  --output-dir artifacts/model
```

## Run inference on a user video

```bash
python infer_floorplan_from_video.py \
  --video /path/to/walkthrough.mp4 \
  --output-dir outputs/run1 \
  --weights artifacts/model/best.pt
```

## Outputs

- `observation.png` – raw top-down map from 3D reconstruction
- `pred_structure.png` – predicted structural floorplan mask
- `pred_furniture.png` – predicted furniture mask
- `combined.png` – colored combined top-down output
- `floorplan.svg` – vectorized wall contours
- `point_cloud.ply` – reconstructed point cloud
- `summary.json`

## Important notes

- **Panorama** is best used for supervision and clean room layout coverage.
- **Perspective/video** is best used for realistic reconstruction.
- The video pipeline uses OpenCV multi-view geometry. It works best for slow walkthrough videos with overlap and stable motion.
- Furniture identification during inference is geometry-driven by default. A hook is provided if you later want to add a real detector/segmenter.
