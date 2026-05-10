from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from room_to_floorplan.geometry import (
    make_bev,
    panorama_depth_to_world,
    save_bev_preview,
    stack_observation,
)
from room_to_floorplan.model import FloorplanUNet, FloorplanUNetAttention, FloorplanUNetPlusPlus
from room_to_floorplan.floorplan_svg import (
    clean_mask,
    save_floorplan_overlay_png,
    save_floorplan_svg,
    save_mask_png,
    save_probability_png,
)


def read_image(path: Path, unchanged: bool = False) -> np.ndarray:
    if unchanged:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    img = Image.open(path)
    return np.array(img)


def load_panorama_folder(panorama_dir: str | Path, rgb_name: str = "rgb_rawlight.png"):
    """
    Load one Structured3D panorama folder.

    Expected structure:
        panorama/
          camera_xyz.txt
          layout.txt
          full/
            depth.png
            semantic.png
            rgb_rawlight.png
    """
    pano = Path(panorama_dir)
    full = pano / "full"

    if not pano.exists():
        raise FileNotFoundError(f"Panorama folder does not exist: {pano}")
    if not full.exists():
        raise FileNotFoundError(f"Missing 'full' folder inside panorama path: {full}")

    rgb_path = full / rgb_name
    if not rgb_path.exists():
        # Fallbacks based on the screenshot / Structured3D variants.
        for candidate in ["rgb_rawlight.png", "rgb_warmlight.png", "rgb_coldlight.png", "albedo.png"]:
            if (full / candidate).exists():
                rgb_path = full / candidate
                break

    depth_path = full / "depth.png"
    semantic_path = full / "semantic.png"
    camera_path = pano / "camera_xyz.txt"

    missing = [p for p in [rgb_path, depth_path, semantic_path, camera_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(str(p) for p in missing))

    rgb = read_image(rgb_path)
    depth_mm = read_image(depth_path, unchanged=True)

    # IMPORTANT:
    # Structured3D semantic.png is usually a palette PNG.
    # PIL preserves the palette index values, which are the actual semantic class IDs.
    # OpenCV can return palette colours instead, which breaks STRUCTURE_IDS/FURNITURE_IDS matching.
    semantic_img = Image.open(semantic_path)
    semantic = np.array(semantic_img)
    
    camera_xyz_mm = np.loadtxt(camera_path, dtype=np.float32)
    
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]

    if semantic.ndim == 3:
        raise ValueError(
            f"Expected semantic.png to load as class IDs, but got shape {semantic.shape}. "
            "Do not load semantic.png through OpenCV. Use PIL so palette IDs are preserved."
        )

    return {
        "rgb": rgb,
        "depth_mm": depth_mm,
        "semantic": semantic,
        "camera_xyz_mm": camera_xyz_mm,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "semantic_path": str(semantic_path),
        "camera_path": str(camera_path),
    }


def build_model(model_name: str, device: str):
    model_name = model_name.lower().strip()
    if model_name in {"unet", "u-net"}:
        model = FloorplanUNet()
    elif model_name in {"unet_attention", "attention", "attention_unet", "unetattention"}:
        model = FloorplanUNetAttention()
    elif model_name in {"unet_plusplus", "unet++", "unetpp", "unetplusplus"}:
        model = FloorplanUNetPlusPlus()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def infer_model_name_from_path(checkpoint_path: Path) -> str:
    s = str(checkpoint_path).lower()
    if "plusplus" in s or "unet++" in s or "unetpp" in s:
        return "unet_plusplus"
    if "attention" in s:
        return "unet_attention"
    return "unet"


def load_model(checkpoint_path: str | Path, model_name: str, device: str):
    checkpoint_path = Path(checkpoint_path)
    if model_name == "auto":
        model_name = infer_model_name_from_path(checkpoint_path)

    print(f"[MODEL] {model_name}")
    model = build_model(model_name, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, model_name


def predict_masks(model, obs: np.ndarray, device: str, threshold: float):
    x = torch.from_numpy(obs.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Training targets are [structure, furniture], so keep that convention.
    structure_prob = probs[0]
    furniture_prob = probs[1] if probs.shape[0] > 1 else np.zeros_like(structure_prob)

    structure_mask = structure_prob >= threshold
    furniture_mask = furniture_prob >= threshold

    return structure_prob, furniture_prob, structure_mask, furniture_mask


def main():
    ap = argparse.ArgumentParser(description="Run panorama -> BEV -> model prediction -> downloadable SVG floorplan.")
    ap.add_argument("--panorama-dir", required=True,
                    help="Path to one Structured3D panorama folder, e.g. .../scene_00005/2D_rendering/827475/panorama")
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt")
    ap.add_argument("--output-dir", required=True, help="Output folder for PNG/SVG results")
    ap.add_argument("--model", default="auto",
                    choices=["auto", "unet", "unet_attention", "unet_plusplus"],
                    help="Model architecture. Use auto if the checkpoint path contains the model name.")
    ap.add_argument("--rgb-name", default="rgb_rawlight.png",
                    help="RGB panorama file inside full/. Fallbacks are tried if missing.")
    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--z-min", type=float, default=-0.5)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--height-lo", type=float, default=0.15)
    ap.add_argument("--height-hi", type=float, default=1.60)
    ap.add_argument("--xy-padding-m", type=float, default=0.35)
    ap.add_argument("--splat-radius", type=int, default=2)

    # cleanup / vectorisation controls
    ap.add_argument("--structure-min-area", type=int, default=120)
    ap.add_argument("--furniture-min-area", type=int, default=80)
    ap.add_argument("--svg-min-area", type=float, default=40.0)
    ap.add_argument("--svg-simplify", type=float, default=1.5)
    ap.add_argument("--no-furniture-svg", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("PANORAMA TO FLOORPLAN INFERENCE")
    print(f"Panorama dir: {args.panorama_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Device: {device}")
    print("=" * 80)

    data = load_panorama_folder(args.panorama_dir, rgb_name=args.rgb_name)

    semantic_unique = np.unique(data["semantic"])
    print(f"[DEBUG] semantic unique values: {semantic_unique[:80]}")
    print(f"[DEBUG] semantic min/max: {data['semantic'].min()} / {data['semantic'].max()}")

    expected_structure = {1, 2, 8, 9, 22}
    expected_furniture = {
        5, 6, 10, 11, 12, 14, 15, 16, 17, 18, 19,
        24, 25, 28, 29, 30, 31, 32, 33, 34
    }

    has_structure = any(int(v) in expected_structure for v in semantic_unique)
    has_furniture = any(int(v) in expected_furniture for v in semantic_unique)

    print(f"[DEBUG] semantic contains structure IDs: {has_structure}")
    print(f"[DEBUG] semantic contains furniture IDs: {has_furniture}")

    if not has_structure:
        raise ValueError(
            "semantic.png does not contain expected Structured3D structure IDs. "
            "This usually means the semantic image was loaded incorrectly."
        )

    pts, colors, sem = panorama_depth_to_world(
        data["depth_mm"],
        data["rgb"],
        data["semantic"],
        data["camera_xyz_mm"],
        stride=args.stride,
    )

    bev = make_bev(
        pts,
        sem,
        colors,
        grid_size=args.grid_size,
        meters_per_pixel=None,
        z_clip=(args.z_min, args.z_max),
        height_band_above_floor=(args.height_lo, args.height_hi),
        xy_padding_m=args.xy_padding_m,
        splat_radius=args.splat_radius,
    )
    obs = stack_observation(bev)

    print("[DEBUG] observation channel stats:")
    for i, name in enumerate(["occupancy", "structure", "furniture", "rgb_r", "rgb_g", "rgb_b"]):
        ch = obs[..., i]
        print(
            f"  {i} {name}: "
            f"min={ch.min():.4f}, max={ch.max():.4f}, "
            f"mean={ch.mean():.6f}, nonzero={np.count_nonzero(ch)}"
        )

    if np.count_nonzero(obs[..., 1]) == 0:
        raise ValueError("BEV structure channel is empty. The model input is invalid.")

    if np.count_nonzero(obs[..., 2]) == 0:
        print("[WARNING] BEV furniture channel is empty. This may be okay for empty rooms, but check semantic loading.")

    model, model_name = load_model(args.checkpoint, args.model, device=device)
    structure_prob, furniture_prob, raw_structure_mask, raw_furniture_mask = predict_masks(
        model, obs, device=device, threshold=args.threshold
    )

    clean_structure = clean_mask(
        structure_prob,
        threshold=args.threshold,
        close_kernel=5,
        open_kernel=3,
        min_area_px=args.structure_min_area,
        fill_holes=True,
    )
    clean_furniture = clean_mask(
        furniture_prob,
        threshold=args.threshold,
        close_kernel=3,
        open_kernel=3,
        min_area_px=args.furniture_min_area,
        fill_holes=False,
    )

    # Save intermediate and final outputs.
    np.save(out_dir / "observation.npy", obs)
    np.save(out_dir / "structure_probability.npy", structure_prob.astype(np.float32))
    np.save(out_dir / "furniture_probability.npy", furniture_prob.astype(np.float32))

    # Target argument is only used by save_bev_preview for colour display. Use raw model masks here.
    preview_target = np.stack([raw_structure_mask.astype(np.float32), raw_furniture_mask.astype(np.float32)], axis=0)
    save_bev_preview(obs, preview_target, out_dir / "bev_observation_preview.png")

    save_probability_png(structure_prob, out_dir / "structure_probability.png")
    save_probability_png(furniture_prob, out_dir / "furniture_probability.png")
    save_mask_png(raw_structure_mask, out_dir / "raw_structure_mask.png")
    save_mask_png(raw_furniture_mask, out_dir / "raw_furniture_mask.png")
    save_mask_png(clean_structure, out_dir / "clean_structure_mask.png")
    save_mask_png(clean_furniture, out_dir / "clean_furniture_mask.png")
    save_floorplan_overlay_png(clean_structure, clean_furniture, out_dir / "floorplan_preview.png")

    meters_per_pixel = float(bev["meters_per_pixel"][0])
    save_floorplan_svg(
        structure_mask=clean_structure,
        furniture_mask=clean_furniture,
        out_svg=out_dir / "floorplan.svg",
        meters_per_pixel=meters_per_pixel,
        min_area_px=args.svg_min_area,
        simplify_px=args.svg_simplify,
        include_furniture=not args.no_furniture_svg,
        title="Predicted floorplan",
    )

    metadata = {
        "panorama_dir": str(args.panorama_dir),
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "device": device,
        "grid_size": args.grid_size,
        "stride": args.stride,
        "threshold": args.threshold,
        "meters_per_pixel": meters_per_pixel,
        "origin_xy": bev["origin_xy"].tolist(),
        "center_xy": bev["center_xy"].tolist(),
        "floor_z": float(bev["floor_z"][0]),
        "height_band_above_floor": bev["height_band_above_floor"].tolist(),
        "input_files": {
            "rgb": data["rgb_path"],
            "depth": data["depth_path"],
            "semantic": data["semantic_path"],
            "camera_xyz": data["camera_path"],
        },
        "outputs": {
            "svg": str(out_dir / "floorplan.svg"),
            "preview_png": str(out_dir / "floorplan_preview.png"),
            "bev_preview_png": str(out_dir / "bev_observation_preview.png"),
        },
    }
    (out_dir / "inference_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 80)
    print("DONE")
    print(f"SVG floorplan: {out_dir / 'floorplan.svg'}")
    print(f"Preview PNG: {out_dir / 'floorplan_preview.png'}")
    print(f"BEV preview: {out_dir / 'bev_observation_preview.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
