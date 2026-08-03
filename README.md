# VisionScape - Room-to-Floorplan Generator

> Upload a 3D scene. Get a clean, vector floorplan. Instantly.

VisionScape is a full-stack AI application that takes a Structured3D scene as input and automatically generates an architectural floorplan using a trained deep learning model. The output is a clean, scalable SVG that can be displayed in the browser and downloaded directly.

---

## Overview

Generating floorplans from raw 3D scene data is a tedious and technically demanding task. VisionScape automates this pipeline end-to-end:

1. User uploads a Structured3D scene ZIP via the web interface
2. The frontend validates the ZIP structure in-browser
3. The backend runs a deep learning inference script on the scene
4. A vector floorplan (SVG) is generated and displayed in the GUI
5. The user downloads the result as SVG or PNG

This project integrates a custom-trained polygon-based neural network with a React frontend and a Flask API backend.

---

## Features

- **Scene upload** - Drag and drop a Structured3D scene ZIP
- **In-browser validation** - ZIP structure is verified before inference runs
- **One-click floorplan generation** - Triggers full deep learning inference on the backend
- **SVG display** - The predicted floorplan renders inline in the browser
- **Download SVG** - Saves the clean vector floorplan file
- **Download PNG** - Saves a rasterised preview image
- **Step-by-step UI** - Clear progress indicator across Upload, Validate, Generate, and Results stages

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, react-dropzone, JSZip |
| Backend | Python 3, Flask, flask-cors |
| Inference | PyTorch, OpenCV, NumPy |
| Model | Custom CNN-Transformer polygon prediction network (RoomToFP v7) |
| Output Format | SVG (vector) + PNG (raster preview) |

---

## Model

The inference pipeline is based on **RoomToFP v7**, a polygon-based floorplan prediction model that:

- Builds an 8-channel bird's-eye-view (BEV) observation from panoramic RGB, depth, and semantic inputs using scene-level panorama aggregation
- Predicts room polygons, wall segments, and openings (doors/windows) using a CNN-Transformer decoder with polygon supervision from Structured3D annotations
- Applies Manhattan snapping, multi-candidate layout scoring, and vector SVG rendering as post-processing steps

The model was trained on the [Structured3D](https://structured3d-dataset.org/) dataset and supports two pipelines:

- **Baseline segmentation pipeline** - U-Net-style models predicting wall and furniture masks from BEV training pairs
- **RoomToFP vector pipeline** - Scene-level aggregation with polygon-based supervision for clean vector output (primary pipeline used by VisionScape)

---

## Project Structure

```
deep-learning-project/
├── app.py                              # Flask backend and API routes
├── infer_roomtofp_from_scene.py        # Deep learning inference script
├── train_roomtofp_stru3d.py            # Model training script
├── eval_roomtofp_stru3d.py             # Model evaluation script
├── prepare_roomtofp_dataset.py         # Dataset preparation script
├── visualize_roomtofp_predictions.py   # Prediction visualisation script
├── generate_previews.py                # Preview generation utility
├── best_ema.pt                         # Model checkpoint (not included in repo)
├── requirements.txt                    # Python dependencies
├── room_to_floorplan/                  # Model architecture and utilities
│   ├── roomtofp_polygon_model.py
│   ├── roomtofp_scene_utils.py
│   ├── floorplan_svg.py
│   └── geometry.py
├── outputs/                            # Inference results (generated at runtime)
│   └── scene_XXXXX/
│       ├── pred_floorplan.svg
│       ├── pred_floorplan_preview.png
│       ├── pred_wall_mask.png
│       ├── pred_opening_mask.png
│       └── prediction_metadata.json
├── extracted/                          # Temporary ZIP extraction (generated at runtime)
├── dataset/                            # Place input scene ZIPs here
└── visionscape/                        # React frontend
    ├── src/
    │   ├── pages/
    │   │   └── Studio.jsx
    │   ├── components/
    │   │   ├── SceneUploader.jsx
    │   │   ├── ValidationPanel.jsx
    │   │   ├── ProcessingPanel.jsx
    │   │   └── FloorplanViewer.jsx
    │   └── services/
    │       └── api.js
    ├── vite.config.js
    └── package.json
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- `best_ema.pt` model checkpoint placed in the root `deep-learning-project/` folder

### 1. Backend

Run these commands from the `deep-learning-project/` root directory:

```bash
python3 -m pip install flask flask-cors
python3 -m pip install -r requirements.txt
python3 app.py
```

The Flask server will start at `http://localhost:5000`.

### 2. Frontend

Open a second terminal:

```bash
cd visionscape
npm install
npm run dev
```

The Vite dev server will start at `http://localhost:5173`.

Open `http://localhost:5173` in your browser.

---

## Usage

### Web App

1. **Upload** - Drag and drop a Structured3D scene ZIP (e.g. `scene_00400.zip`) onto the dropzone
2. **Validate** - The app checks the ZIP structure automatically
3. **Generate** - Click **Generate Floorplan** to run inference on the backend
4. **View** - The predicted floorplan SVG appears in the main panel
5. **Download** - Use the **Download SVG** or **Download PNG** buttons to save the result

### Command Line

**Prepare the dataset:**

```bash
python prepare_roomtofp_dataset.py \
  --structured3d-root /path/to/Structured3D \
  --output output/data/roomtofp_polygon_v7 \
  --grid-size 512 \
  --variants full \
  --furniture-source auto
```

**Train the model:**

```bash
python train_roomtofp_stru3d.py \
  --dataset-root output/data/roomtofp_polygon_v7 \
  --output-dir output/artifacts/roomtofp_polygon_v7 \
  --epochs 80 \
  --batch-size 4 \
  --lr 1e-4 \
  --device cuda
```

**Evaluate the model:**

```bash
python eval_roomtofp_stru3d.py \
  --dataset-root output/data/roomtofp_polygon_v7 \
  --checkpoint output/artifacts/roomtofp_polygon_v7/best_ema.pt \
  --split test \
  --output output/artifacts/roomtofp_eval_test.json
```

**Run inference on a single scene:**

```bash
python infer_roomtofp_from_scene.py \
  --scene-dir /path/to/Structured3D/scene_00400 \
  --checkpoint best_ema.pt \
  --output-dir outputs/scene_00400 \
  --grid-size 512 \
  --auto-select-layout \
  --layout-candidates balanced
```

**Visualise predictions:**

```bash
python visualize_roomtofp_predictions.py \
  --dataset-root output/data/roomtofp_polygon_v7 \
  --checkpoint best_ema.pt \
  --output-dir output/artifacts/roomtofp_prediction_viz \
  --split test \
  --num-samples 12
```

---

## Outputs

| File | Description |
|---|---|
| `pred_floorplan.svg` | Clean vector floorplan (primary output) |
| `pred_floorplan_preview.png` | Rasterised colour preview |
| `pred_wall_mask.png` | Predicted wall segmentation mask |
| `pred_opening_mask.png` | Predicted door/window mask |
| `pred_room_union_mask.png` | Union of all predicted room masks |
| `prediction_metadata.json` | Scene metadata and inference settings |
| `best_ema.pt` | Best model checkpoint (EMA weights) |
| `history.json` | Training loss and metric curves |

---

## Notes

- `best_ema.pt` is not included in this repository due to file size. It must be placed manually at `deep-learning-project/best_ema.pt` before running the backend.
- Input ZIPs must follow the Structured3D format, with a top-level `scene_XXXXX/` folder containing `annotation_3d.json` and `2D_rendering/`.
- Inference time varies depending on hardware. GPU is used automatically if available; CPU fallback is supported.
- The `extracted/` directory is wiped and recreated on each request to prevent stale data between runs.

---

## Future Improvements

- Add support for batch scene processing
- Export floorplan with room labels and area measurements
- Integrate a 3D viewer alongside the 2D floorplan output
- Add user authentication and a results history panel
- Containerise the full stack with Docker for easier deployment

---

## Acknowledgements

VisionScape builds upon prior research and datasets, including **RoomFormer** and the **Structured3D** dataset.

- The floorplan generation pipeline is inspired by and derived from concepts introduced in RoomFormer, including transformer-based approaches for structured floorplan reconstruction.
- Training and evaluation data are based on the Structured3D dataset, which provides annotated 3D indoor scenes used for model development.

This repository contains the VisionScape application, including the frontend interface, backend API, inference pipeline, model integration, and custom modifications developed for this project.

---
