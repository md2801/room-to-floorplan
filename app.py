"""
VisionScape Backend — app.py

Directory layout (all relative to this file):
  deep-learning-project/
    app.py
    infer_roomtofp_from_scene.py
    best_ema.pt                  ← must be placed here manually
    extracted/                   ← per-request extraction (cleared each run)
    outputs/                     ← inference output: outputs/scene_XXXXX/
    room_to_floorplan/
    visionscape/
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# FROM AAQIL: commented the og checkpoint with my checkpoint path for testing.
# If youre testing this, comment my path instead of overwriting it and use your own path
CHECKPOINT = BASE_DIR / "output" / "v7" / "artifacts" / "roomtofp_polygon_v7_512_c16_final" / "best_ema.pt"
# CHECKPOINT = BASE_DIR / "best_ema.pt"

EXTRACT_DIR = BASE_DIR / "extracted"
OUTPUT_ROOT = BASE_DIR / "outputs"
INFER_SCRIPT = BASE_DIR / "infer_roomtofp_from_scene.py"

app = Flask(__name__)
CORS(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_scene_dir(root: Path) -> Path | None:
    """
    Return the first directory matching scene_\\d+ found under root.
    Skips __MACOSX and other junk entries.
    Also searches one level deep in case the ZIP extracted into a
    same-named subfolder (e.g. extracted/scene_00400/scene_00400/).
    """
    candidates = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("__") or entry.name.startswith("."):
            continue
        import re
        if re.match(r"^scene_\d+$", entry.name):
            candidates.append(entry)

    if candidates:
        return candidates[0]

    # One level deeper (ZIP extracted into a wrapper folder)
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("__") or entry.name.startswith("."):
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and re.match(r"^scene_\d+$", sub.name):
                return sub

    return None


def _validate_prereqs() -> str | None:
    if not INFER_SCRIPT.exists():
        return f"Inference script not found: {INFER_SCRIPT}"
    if not CHECKPOINT.exists():
        return (
            f"Checkpoint not found: {CHECKPOINT}. "
            "Place best_ema.pt directly inside deep-learning-project/."
        )
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    err = _validate_prereqs()
    if err:
        return jsonify({"status": "error", "message": err}), 500
    return jsonify({"status": "ok", "checkpoint": str(CHECKPOINT)}), 200


@app.route("/inferFloorplan", methods=["POST"])
def infer_floorplan():
    # 1. Pre-flight
    err = _validate_prereqs()
    if err:
        return jsonify({"success": False, "error": err}), 500

    uploaded_zip = request.files.get("sceneZip")
    if uploaded_zip is None:
        return jsonify({"success": False, "error": "No file uploaded (field: sceneZip)."}), 400

    filename = uploaded_zip.filename or "scene.zip"
    if not filename.lower().endswith(".zip"):
        return jsonify({"success": False, "error": "Uploaded file must be a .zip."}), 400

    # 2. Wipe & recreate extraction dir to avoid stale scene folders
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 3. Save & extract ZIP
    zip_save_path = EXTRACT_DIR / filename
    uploaded_zip.save(str(zip_save_path))

    try:
        with zipfile.ZipFile(zip_save_path, "r") as zf:
            # Filter out __MACOSX and .DS_Store entries
            members = [
                m for m in zf.infolist()
                if not any(
                    part.startswith("__MACOSX") or part == ".DS_Store"
                    for part in m.filename.split("/")
                )
            ]
            zf.extractall(EXTRACT_DIR, members=[m.filename for m in members])
    except zipfile.BadZipFile:
        return jsonify({"success": False, "error": "Uploaded file is not a valid ZIP."}), 400

    # 4. Locate scene_XXXXX folder
    scene_dir = _find_scene_dir(EXTRACT_DIR)
    if scene_dir is None:
        return jsonify({
            "success": False,
            "error": "No scene_XXXXX folder found in the extracted ZIP.",
        }), 400

    scene_name = scene_dir.name  # e.g. "scene_00400"

    # 5. Output directory
    output_dir = OUTPUT_ROOT / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Build inference command
    command = [
        sys.executable,
        str(INFER_SCRIPT),
        "--scene-dir",                  str(scene_dir),
        "--checkpoint",                 str(CHECKPOINT),
        "--output-dir",                 str(output_dir),
        "--grid-size",                  "512",
        "--auto-select-layout",
        "--layout-candidates",          "balanced",
        "--mask-threshold",             "0.50",
        "--min-room-area-px",           "80",
        "--opening-snap-distance-px",   "22",
        "--min-opening-area-px",        "18",
        "--min-opening-length-px",      "20",
        "--bev-opening-weight",         "0.35",
        "--merge-opening-distance-px",  "18",
        "--max-opening-length-px",      "80",
        "--snap-axis-threshold-px",     "8",
        "--simplify-eps-px",            "2",
        "--debug-layout-candidates",
        "--debug-layout-candidate-previews",
    ]

    print(f"\n[VisionScape] Running inference for {scene_name}")
    print(f"[VisionScape] scene_dir  : {scene_dir}")
    print(f"[VisionScape] output_dir : {output_dir}")
    print(f"[VisionScape] command    : {' '.join(command)}\n")

    # 7. Run
    try:
        proc = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Inference timed out (10 min)."}), 500
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to launch script: {exc}"}), 500

    # Always print subprocess output so you can see real errors in terminal
    if proc.stdout:
        print("[infer stdout]\n" + proc.stdout[-4000:])
    if proc.stderr:
        print("[infer stderr]\n" + proc.stderr[-4000:])

    if proc.returncode != 0:
        return jsonify({
            "success": False,
            "error": "Inference script exited with an error. See terminal for details.",
            "stderr": proc.stderr[-3000:] if proc.stderr else "",
            "stdout": proc.stdout[-3000:] if proc.stdout else "",
        }), 500

    # 8. Verify outputs
    svg_path = output_dir / "pred_floorplan.svg"
    png_path = output_dir / "pred_floorplan_preview.png"

    missing = [str(p) for p in [svg_path, png_path] if not p.exists()]
    if missing:
        return jsonify({
            "success": False,
            "error": "Inference finished but output files are missing.",
            "missing_files": missing,
        }), 500

    # 9. Return URLs (served via /outputs/ route below)
    base_url = request.host_url.rstrip("/")
    return jsonify({
        "success":    True,
        "scene_name": scene_name,
        "svg_url":    f"{base_url}/outputs/{scene_name}/pred_floorplan.svg",
        "png_url":    f"{base_url}/outputs/{scene_name}/pred_floorplan_preview.png",
    }), 200


# ── Static serving ────────────────────────────────────────────────────────────

@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(str(OUTPUT_ROOT), filename)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[VisionScape] BASE_DIR: {BASE_DIR}")
    print(f"[VisionScape] CHECKPOINT: {CHECKPOINT}")
    print(f"[VisionScape] CHECKPOINT exists: {CHECKPOINT.exists()}")
    print(f"[VisionScape] OUTPUT_ROOT: {OUTPUT_ROOT}")
    app.run(debug=True, port=5000)
