import os
import torch
import subprocess
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)

'''Path Configurations:
- DEVICE: The device to run inference on (GPU if available, otherwise CPU).
- BASE_DIR: The directory where this app.py file is located.
- OUTPUT_ROOT: The root directory where inference outputs will be saved.
- SCENE_DIR: The directory of the input scene for inference.
- CHECKPOINT: The path to the model checkpoint used for inference.
'''
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, 'output')
scene_folder_name = "scene_00000"
SCENE_DIR = os.path.join(BASE_DIR, 'dataset', 'Structured3D', 'scene_00000')
CHECKPOINT = os.path.join(
    BASE_DIR,
    'output',
    'v7',
    'artifacts',
    'best.pt'
)

'''Defining the endpoint for floorplan inference 
which will run the infer_roomtofp_from_scene.py script 
with the specified parameters and return the generated floorplan preview and SVG paths as JSON response.
'''
@app.route('/inferFloorplan', methods=['POST'])
def infer_floorplan():
    script = 'infer_roomtofp_from_scene.py'
    output_dir = os.path.join(
        OUTPUT_ROOT,
        "inference",
        "v7",
        f"{scene_folder_name}_best_clean"
    )

    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python", script,

        "--scene-dir", SCENE_DIR,
        "--checkpoint", CHECKPOINT,
        "--output-dir", output_dir,

        "--variants", "full",
        "--grid-size", "512",
        "--device", DEVICE,

        "--room-threshold", "0.35",
        "--corner-threshold", "0.45",
        "--simplify-eps-px", "2",
        "--snap-axis-threshold-px", "8",
        "--opening-snap-distance-px", "45",
        "--min-opening-area-px", "4",
        "--min-opening-length-px", "14",
        "--bev-opening-weight", "1.0"
    ]

    # Run the command using subprocess
    # and throw process error if script failed.
    try:
        print(command)
        subprocess.run(command, check=True, cwd=BASE_DIR, shell=True)
    
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error":"Inference script failed",
            "details": str(e)
        }), 500
    
    # Extract Output files from the path specified 
    preview_file = "pred_floorplan_preview.png"
    preview_path = os.path.join(output_dir, preview_file)
    svg_file = "pred_floorplan.svg"
    svg_path = os.path.join(output_dir, svg_file)

    if not os.path.exists(preview_path) or not os.path.exists(svg_path):
        return jsonify({
            "error": "Output files not found",
            "expected_preview": preview_path,
            "expected_svg": svg_path
        }), 500
    
    relative_output_dir = os.path.relpath(output_dir, OUTPUT_ROOT).replace("\\", "/")

    return jsonify({
        "preview_png": f"http://localhost:5000/output/{relative_output_dir}/{preview_file}",
        "svg": f"http://localhost:5000/output/{relative_output_dir}/{svg_file}"
    })

@app.route("/output/<path:filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_ROOT, filename)

def main():
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()