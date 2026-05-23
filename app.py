import os
import torch
import subprocess
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)

'''Path Configurations:
- BASE_DIR: The directory where this app.py file is located.
- OUTPUT_ROOT: The root directory where inference outputs will be saved.
- SCENE_DIR: The directory of the input scene for inference.
- CHECKPOINT: The path to the model checkpoint used for inference.
'''
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, 'output')

'''Change the SCENE_DIR and CHECKPOINT to desired paths,
but make sure the SCENE_DIR and CHECKPOINT paths are valid and exist on your system before running the app.
Also add the corresponding scene folder name in the scene_folder_name variable for correct output path generation.
Example: scene_00000
'''
scene_folder_name = "SCENE_FOLDER_NAME"
SCENE_DIR = os.path.join(BASE_DIR, 'dataset','Structured3D', scene_folder_name)
CHECKPOINT = os.path.join(
    BASE_DIR,
    'output',
    'v7',
    'artifacts',
    'best_ema.pt'
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
        f"{scene_folder_name}_best_ema_v7"
    )

    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python", script,
        "--scene-dir", SCENE_DIR,
        "--checkpoint", CHECKPOINT,
        "--output-dir", output_dir,
        "--grid-size", "512",
        "--auto-select-layout",
        "--layout-candidates", "balanced",
        "--mask-threshold", "0.50",
        "--min-room-area-px", "80",
        "--opening-snap-distance-px", "28",
        "--min-opening-area-px", "12",
        "--min-opening-length-px", "18",
        "--bev-opening-weight", "0.5",
        "--snap-axis-threshold-px", "8",
        "--simplify-eps-px", "2"
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
    preview_png = "pred_floorplan_preview.png"
    preview_path = os.path.join(output_dir, preview_png)
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
        "png": f"http://localhost:5000/output/{relative_output_dir}/{preview_png}",
        "svg": f"http://localhost:5000/output/{relative_output_dir}/{svg_file}"
    })

@app.route("/output/<path:filename>")
def get_output(filename):
    return send_from_directory(
        OUTPUT_ROOT, 
        filename
    )

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(
        OUTPUT_ROOT,
        filename,
        as_attachment=True
    )

def main():
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()