# VisionScape — Room-to-Floorplan Generator

A polished React/Vite SaaS application for AI-powered room-to-floorplan conversion.
Supports **Structured3D scene ZIPs** (primary) and video walkthroughs (legacy).

---

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:5173

---

## Project Structure

```
src/
├── components/
│   ├── Navbar.jsx / .css             # Sticky blurred top navigation
│   ├── SceneUploader.jsx / .css      # Upload mode selector (ZIP or Video)
│   ├── VideoUploader.jsx / .css      # Video upload + camera recording
│   ├── ValidationPanel.jsx / .css    # Scene ZIP validation result display
│   ├── ProcessingPanel.jsx / .css    # Generate button + animated progress
│   └── FloorplanViewer.jsx / .css    # 3-mode floorplan display (SVG/CSS)
├── pages/
│   ├── Home.jsx / .css               # Landing page
│   └── Studio.jsx / .css            # Main workspace (full workflow)
├── services/
│   └── api.js                        # All mock API functions
├── styles/
│   └── global.css                    # Design tokens + CSS reset
└── main.jsx / App.jsx
```

---

## Supported Input Modes

### Mode A — Structured3D Scene ZIP (Primary)

Upload a `scene_XXXXX.zip` archive conforming to the Structured3D dataset format.

**Expected ZIP structure:**
```
scene_00400/
├── annotation_3d.json
└── 2D_rendering/
    └── <room_id>/
        └── panorama/
            ├── camera_xyz.txt
            ├── layout.txt
            ├── full/
            │   ├── rgb_rawlight.png    ← required
            │   ├── depth.png           ← required
            │   ├── semantic.png        ← required
            │   ├── normal.png          ← required
            │   ├── rgb_coldlight.png   (optional)
            │   ├── rgb_warmlight.png   (optional)
            │   └── albedo.png          (optional)
            ├── empty/                  (optional)
            └── simple/                 (optional)
```

**Validation rules applied:**
1. ZIP must contain exactly one top-level folder matching `scene_XXXXX`
2. Must contain `annotation_3d.json` at scene root
3. Must contain `2D_rendering/`
4. Detects all room IDs under `2D_rendering/`
5. Per room: checks for `panorama/` folder
6. Per panorama: checks all 4 required files (`camera_xyz.txt`, `layout.txt`, `full/rgb_rawlight.png`, `full/depth.png`, `full/semantic.png`, `full/normal.png`)
7. Mac metadata ignored: `__MACOSX/`, `.DS_Store`, `._*`, dotfiles

**Status badges:**
- Icon - ✅ **Valid Scene** — all rooms pass all required checks
- Icon - ⚠️ **Partially Valid** — at least one valid room, some issues
- Icon - ❌ **Invalid Scene** — missing critical structure

### Mode B — Video Walkthrough (Legacy)

Upload MP4/MOV/WebM or record live via device camera.

**Note:** This mode is retained for future work, where the model can be extended to support direct floorplan generation from real-world video inputs.
---

## User Workflow (Scene ZIP mode)

```
Open Studio
  → Select "Structured3D Scene ZIP"
  → Upload scene_XXXXX.zip
  → [Auto] ZIP is validated in-browser via JSZip
  → Validation summary shown:
       scene name · rooms found · valid/invalid per room
       missing files · optional available
       status badge (Valid / Partial / Invalid)
  → Select a room from the room chip selector
  → Click "Generate Floorplan"
  → Animated progress bar runs (mock 7-step pipeline)
  → FloorplanViewer renders:
       · Without Furniture (dimensioned SVG)
```

---

## API Services (`src/services/api.js`)

| Function | Description |
|---|---|
| `validateSceneZip(file)` | Reads ZIP in-browser with JSZip, validates Structured3D structure, returns full result object |
| `extractPreviewImage(zip, path)` | Extracts a PNG from ZIP as an object URL for preview thumbnails |
| `generateFloorplanFromScene(metadata, roomId, onProgress)` | Mock async generation with 7 progress steps |
| `uploadVideo(file)` | Legacy mock video upload |
| `generateFloorplan(fileId, onProgress)` | Legacy mock video floorplan generation |

---

## Design System

| Token | Value |
|---|---|
| Background | `#0d0d0e` (base) / `#141416` (surface) |
| Text | `#f0ede8` (primary) / `#9b9791` (secondary) |
| Accent | `#c9a96e` (warm gold) |
| Success | `#6db890` |
| Error | `#c96e6e` |
| Font (headings) | DM Serif Display |
| Font (body/UI) | DM Sans |

---

## Tech Stack

- **React 18** + **Vite 5**
- **React Router v6**
- **React Dropzone** — drag & drop upload
- **JSZip 3.10** — in-browser ZIP parsing + validation
- **Lucide React** — icons
- **Pure CSS** with custom properties (no Tailwind)

---

## Testing the ZIP Validator

1. Run `npm run dev`
2. Navigate to Studio → **Structured3D Scene ZIP**
3. Upload `scene_00400.zip` (reference file)
4. Expected output:
   - Scene name: `scene_00400`
   - 6 rooms detected: `201`, `343735`, `365962`, `366378`, `406`, `458`
   - All 6 rooms: ✅ Valid
   - Status badge: **Valid Scene**
   - Each room expandable → shows available files + panorama preview images
5. Select any room → click **Generate Floorplan** → watch progress bar
6. FloorplanViewer renders with scene + room metadata in subtitle
