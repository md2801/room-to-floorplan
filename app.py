import os
import zipfile
import subprocess
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

'''Change CHECKPOINT to desired paths,
but make sure that the CHECKPOINT path is valid and exist on your system before running the app.
'''
CHECKPOINT = os.path.join(
    BASE_DIR,
    'best_ema.pt'
)

OUTPUT_ROOT = os.path.join(BASE_DIR, "output", "inference")

'''Defining the endpoint for floorplan inference 
which will run the infer_roomtofp_from_scene.py script 
with the specified parameters and return the generated floorplan preview and SVG paths as JSON response.
'''
@app.route('/inferFloorplan', methods=['POST'])
def infer_floorplan():

    uploaded_zip = request.files.get("sceneZip")

    if uploaded_zip is None:
        return jsonify({
            "error": "No file uploaded"
        }), 400
    
    extract_dir = os.path.join(BASE_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    script = 'infer_roomtofp_from_scene.py'

    zip_path = os.path.join(BASE_DIR, uploaded_zip.filename)
    uploaded_zip.save(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    SCENE_DIR = None

    for item in os.listdir(extract_dir):
        item_path = os.path.join(extract_dir, item)

        if os.path.isdir(item_path) and item.startswith("scene_"):
            SCENE_DIR = item_path
            scene_folder_name = item
            break

    if SCENE_DIR is None:
        return jsonify({
            "error": "No valid scene directory found in the uploaded zip"
        }), 400
    
    output_dir = os.path.join(
        OUTPUT_ROOT,
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
        "svg": f"http://localhost:5000/output/{relative_output_dir}/{svg_file}",
        "download_png": f"http://localhost:5000/download/{relative_output_dir}/{preview_png}",
        "download_svg": f"http://localhost:5000/download/{relative_output_dir}/{svg_file}"
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