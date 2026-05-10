from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from room_to_floorplan.geometry import panorama_depth_to_world, save_bev_preview, stack_observation, norm01
from room_to_floorplan.model import FloorplanUNet, FloorplanUNetAttention, FloorplanUNetPlusPlus
from room_to_floorplan.structured3d import FURNITURE_IDS, STRUCTURE_IDS
from room_to_floorplan.floorplan_svg import (
    clean_mask,
    save_floorplan_overlay_png,
    save_floorplan_svg,
    save_mask_png,
    save_probability_png,
)


@dataclass
class SceneGrid:
    origin_xy: np.ndarray
    center_xy: np.ndarray
    meters_per_pixel: np.ndarray
    floor_z: np.ndarray
    height_band_above_floor: np.ndarray


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_depth_mm(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def read_semantic_ids(path: Path) -> np.ndarray:
    # Structured3D semantic PNGs are usually palette-index PNGs. PIL preserves IDs.
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        raise ValueError(
            f"Expected semantic.png to load as class IDs, but got RGB-like shape {arr.shape}: {path}. "
            "Use the original Structured3D semantic.png, not a rendered colour legend image."
        )
    return arr


def choose_rgb_file(variant_dir: Path, preferred: str = "rgb_rawlight.png") -> Path:
    candidates = [preferred, "rgb_rawlight.png", "rgb_warmlight.png", "rgb_coldlight.png", "albedo.png"]
    for name in candidates:
        p = variant_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No usable RGB image found in {variant_dir}")


def find_panorama_dirs(scene_dir: Path) -> list[Path]:
    render_dir = scene_dir / "2D_rendering"
    if not render_dir.exists():
        raise FileNotFoundError(f"Missing 2D_rendering folder inside scene: {render_dir}")

    pano_dirs = []
    for room_dir in sorted(render_dir.iterdir()):
        if not room_dir.is_dir() or room_dir.name.startswith("."):
            continue
        pano = room_dir / "panorama"
        if (pano / "camera_xyz.txt").exists():
            pano_dirs.append(pano)

    if not pano_dirs:
        raise FileNotFoundError(f"No panorama folders found under: {render_dir}")
    return pano_dirs


def variant_dirs_for_pano(pano_dir: Path, variants: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for name in variants:
        d = pano_dir / name
        if (d / "depth.png").exists() and (d / "semantic.png").exists():
            out.append((name, d))
    return out


def collect_scene_points(
    scene_dir: Path,
    variants: list[str],
    rgb_name: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Collect points from every panorama and every requested variant in one shared world coordinate system."""
    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_sem: list[np.ndarray] = []
    sources: list[dict] = []

    pano_dirs = find_panorama_dirs(scene_dir)
    print(f"[SCENE] found panorama folders: {len(pano_dirs)}")

    for pano_dir in pano_dirs:
        camera_path = pano_dir / "camera_xyz.txt"
        camera_xyz_mm = np.loadtxt(camera_path, dtype=np.float32)
        room_id = pano_dir.parent.name

        available = variant_dirs_for_pano(pano_dir, variants)
        if not available:
            print(f"[SKIP] room={room_id}: no requested variants found in {pano_dir}")
            continue

        for variant_name, variant_dir in available:
            depth_path = variant_dir / "depth.png"
            semantic_path = variant_dir / "semantic.png"
            rgb_path = choose_rgb_file(variant_dir, preferred=rgb_name)

            rgb = read_rgb(rgb_path)
            depth_mm = read_depth_mm(depth_path)
            semantic = read_semantic_ids(semantic_path)

            semantic_unique = np.unique(semantic)
            has_structure = any(int(v) in STRUCTURE_IDS for v in semantic_unique)
            has_furniture = any(int(v) in FURNITURE_IDS for v in semantic_unique)

            print(
                f"[LOAD] room={room_id} variant={variant_name} "
                f"sem_ids={semantic_unique[:30].tolist()} "
                f"structure={has_structure} furniture={has_furniture}"
            )

            pts, colors, sem = panorama_depth_to_world(
                depth_mm=depth_mm,
                rgb=rgb,
                semantic=semantic,
                camera_xyz_mm=camera_xyz_mm,
                stride=stride,
            )

            all_pts.append(pts)
            all_colors.append(colors[:, :3] if colors.ndim == 2 and colors.shape[1] >= 3 else colors)
            all_sem.append(sem)
            sources.append(
                {
                    "room_id": room_id,
                    "variant": variant_name,
                    "panorama_dir": str(pano_dir),
                    "rgb": str(rgb_path),
                    "depth": str(depth_path),
                    "semantic": str(semantic_path),
                    "camera_xyz": str(camera_path),
                    "points": int(len(pts)),
                    "semantic_ids": [int(x) for x in semantic_unique.tolist()],
                }
            )

    if not all_pts:
        raise ValueError("No valid panorama/variant data was loaded for this scene.")

    pts_cat = np.concatenate(all_pts, axis=0).astype(np.float32)
    colors_cat = np.concatenate(all_colors, axis=0).astype(np.uint8)
    sem_cat = np.concatenate(all_sem, axis=0)
    print(f"[SCENE] total sampled points before filtering: {len(pts_cat):,}")
    
    return pts_cat, colors_cat, sem_cat, sources


def create_scene_grid(
    points_xyz: np.ndarray,
    grid_size: int,
    z_clip: tuple[float, float],
    height_band_above_floor: tuple[float, float],
    xy_padding_m: float,
    meters_per_pixel: Optional[float] = None,
    annotation_xy_m: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SceneGrid]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    keep = (pts[:, 2] >= z_clip[0]) & (pts[:, 2] <= z_clip[1])
    pts_clip = pts[keep]
    
    if len(pts_clip) == 0:
        raise ValueError("No points remain after z clipping.")

    floor_z = float(np.percentile(pts_clip[:, 2], 2.0))
    band_lo = floor_z + float(height_band_above_floor[0])
    band_hi = floor_z + float(height_band_above_floor[1])
    keep_band_clip = (pts_clip[:, 2] >= band_lo) & (pts_clip[:, 2] <= band_hi)
    band_pts = pts_clip[keep_band_clip]
    
    if len(band_pts) == 0:
        raise ValueError("No points remain after floor-height band filtering.")

    xy_sources = [band_pts[:, :2]]
    if annotation_xy_m is not None and len(annotation_xy_m) > 0:
        xy_sources.append(annotation_xy_m.astype(np.float32))
    all_xy = np.concatenate(xy_sources, axis=0)

    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    center_xy = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-3)
    half_extent = 0.5 * float(max(span)) + float(xy_padding_m)

    if meters_per_pixel is None:
        meters_per_pixel = float((2.0 * half_extent) / max(grid_size - 1, 1))

    origin_xy = np.array([center_xy[0] - half_extent, center_xy[1] - half_extent], dtype=np.float32)
    grid = SceneGrid(
        origin_xy=origin_xy,
        center_xy=center_xy.astype(np.float32),
        meters_per_pixel=np.array([meters_per_pixel], dtype=np.float32),
        floor_z=np.array([floor_z], dtype=np.float32),
        height_band_above_floor=np.array([height_band_above_floor[0], height_band_above_floor[1]], dtype=np.float32),
    )

    # Return global keep masks with respect to original points for later channel generation.
    keep_global = keep.copy()
    idx_clip = np.where(keep)[0]
    keep_global[idx_clip[~keep_band_clip]] = False
    
    return keep_global, band_pts, all_xy, grid


def world_xy_to_pixels(xy_m: np.ndarray, grid: SceneGrid, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    mpp = float(grid.meters_per_pixel[0])
    px = np.clip(((xy_m[:, 0] - grid.origin_xy[0]) / mpp).astype(np.int32), 0, grid_size - 1)
    py = np.clip(((xy_m[:, 1] - grid.origin_xy[1]) / mpp).astype(np.int32), 0, grid_size - 1)
    
    return px, py


def make_scene_bev(
    points_xyz: np.ndarray,
    semantic: np.ndarray,
    colors: np.ndarray,
    grid_size: int,
    z_clip: tuple[float, float],
    height_band_above_floor: tuple[float, float],
    xy_padding_m: float,
    splat_radius: int,
    meters_per_pixel: Optional[float] = None,
    annotation_xy_m: Optional[np.ndarray] = None,
) -> dict:
    keep_global, _, _, grid = create_scene_grid(
        points_xyz=points_xyz,
        grid_size=grid_size,
        z_clip=z_clip,
        height_band_above_floor=height_band_above_floor,
        xy_padding_m=xy_padding_m,
        meters_per_pixel=meters_per_pixel,
        annotation_xy_m=annotation_xy_m,
    )

    pts = points_xyz[keep_global]
    sem = semantic[keep_global]
    colors = colors[keep_global]

    px, py = world_xy_to_pixels(pts[:, :2], grid, grid_size)

    H = W = grid_size
    occ = np.zeros((H, W), np.float32)
    structure = np.zeros((H, W), np.float32)
    furniture = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    count = np.zeros((H, W, 1), np.float32)

    offsets: list[tuple[int, int]] = []
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            if dx * dx + dy * dy <= splat_radius * splat_radius:
                offsets.append((dy, dx))

    for i in range(len(px)):
        sid = int(sem[i])
        for dy, dx in offsets:
            yy = int(py[i]) + dy
            xx = int(px[i]) + dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                continue

            occ[yy, xx] += 0.30
            if sid in STRUCTURE_IDS:
                structure[yy, xx] += 4.0
                occ[yy, xx] += 0.80
            elif sid in FURNITURE_IDS:
                furniture[yy, xx] += 0.35
                occ[yy, xx] += 0.15

            rgb[yy, xx] += colors[i].astype(np.float32)
            count[yy, xx] += 1.0

    rgb = rgb / np.maximum(count, 1.0)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32)

    # Similar normalization/smoothing to the training-time BEV builder, but applied after global fusion.
    occ = norm01(cv2.GaussianBlur(np.log1p(occ), (0, 0), 1.4))
    structure = norm01(cv2.GaussianBlur(np.log1p(structure), (0, 0), 1.5))
    furniture = norm01(cv2.GaussianBlur(np.log1p(furniture), (0, 0), 1.15))

    structure_bin = structure > 0.07
    furniture_bin = furniture > 0.11
    structure = np.maximum(structure, clean_mask(structure_bin, threshold=0.5, close_kernel=5, open_kernel=3, min_area_px=24, fill_holes=True).astype(np.float32))
    furniture = np.maximum(furniture, clean_mask(furniture_bin, threshold=0.5, close_kernel=3, open_kernel=3, min_area_px=20, fill_holes=False).astype(np.float32))
    occ = np.clip(np.maximum(occ, 0.80 * structure + 0.20 * furniture), 0.0, 1.0)

    return {
        "origin_xy": grid.origin_xy,
        "center_xy": grid.center_xy,
        "meters_per_pixel": grid.meters_per_pixel,
        "floor_z": grid.floor_z,
        "height_band_above_floor": grid.height_band_above_floor,
        "occupancy": occ.astype(np.float32),
        "structure": norm01(structure).astype(np.float32),
        "furniture": norm01(furniture).astype(np.float32),
        "rgb": rgb,
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
        # Training used BCEWithLogits, so these channels are independent multilabel outputs.
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    structure_prob = probs[0]
    furniture_prob = probs[1] if probs.shape[0] > 1 else np.zeros_like(structure_prob)
    return structure_prob, furniture_prob, structure_prob >= threshold, furniture_prob >= threshold


def load_annotation_xy(scene_dir: Path) -> Optional[np.ndarray]:
    p = scene_dir / "annotation_3d.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    pts = []
    for j in data.get("junctions", []):
        c = j.get("coordinate")
        if c and len(c) >= 2:
            pts.append([float(c[0]) / 1000.0, float(c[1]) / 1000.0])
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float32)


def annotation_wall_mask(scene_dir: Path, bev: dict, grid_size: int, thickness_px: int = 3) -> Optional[np.ndarray]:
    """Rasterise wall/door/window lines from Structured3D annotation_3d.json when available."""
    p = scene_dir / "annotation_3d.json"
    if not p.exists():
        return None

    data = json.loads(p.read_text(encoding="utf-8"))
    junctions = {int(j["ID"]): np.asarray(j["coordinate"], dtype=np.float32) / 1000.0 for j in data.get("junctions", [])}
    planes = {int(pl["ID"]): pl for pl in data.get("planes", [])}
    plane_line = np.asarray(data.get("planeLineMatrix", []), dtype=np.uint8)
    line_junc = np.asarray(data.get("lineJunctionMatrix", []), dtype=np.uint8)

    if plane_line.size == 0 or line_junc.size == 0:
        return None

    # Include wall planes and door/window planes because they are useful floorplan boundaries/openings.
    semantic_plane_ids: set[int] = set()
    for sem in data.get("semantics", []):
        if str(sem.get("type", "")).lower() in {"door", "window", "outwall"}:
            semantic_plane_ids.update(int(x) for x in sem.get("planeID", []))

    wanted_plane_ids = {
        pid for pid, pl in planes.items()
        if str(pl.get("type", "")).lower() == "wall" or pid in semantic_plane_ids
    }

    line_ids: set[int] = set()
    for pid in wanted_plane_ids:
        if 0 <= pid < plane_line.shape[0]:
            for lid in np.where(plane_line[pid] > 0)[0].tolist():
                line_ids.add(int(lid))

    if not line_ids:
        return None

    grid = SceneGrid(
        origin_xy=np.asarray(bev["origin_xy"], dtype=np.float32),
        center_xy=np.asarray(bev["center_xy"], dtype=np.float32),
        meters_per_pixel=np.asarray(bev["meters_per_pixel"], dtype=np.float32),
        floor_z=np.asarray(bev["floor_z"], dtype=np.float32),
        height_band_above_floor=np.asarray(bev["height_band_above_floor"], dtype=np.float32),
    )
    mask = np.zeros((grid_size, grid_size), np.uint8)

    for lid in sorted(line_ids):
        if lid < 0 or lid >= line_junc.shape[0]:
            continue
        jids = np.where(line_junc[lid] > 0)[0].tolist()
        if len(jids) < 2:
            continue
        p0 = junctions.get(int(jids[0]))
        p1 = junctions.get(int(jids[1]))
        if p0 is None or p1 is None:
            continue

        # Ignore purely vertical edges: their top-down projection is just a point.
        if np.linalg.norm(p0[:2] - p1[:2]) < 0.03:
            continue

        xy = np.asarray([p0[:2], p1[:2]], dtype=np.float32)
        px, py = world_xy_to_pixels(xy, grid, grid_size)
        cv2.line(mask, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), 1, thickness=max(1, thickness_px), lineType=cv2.LINE_AA)

    if np.count_nonzero(mask) == 0:
        return None
    return mask.astype(bool)


def mask_to_boundary(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """
    Convert a filled binary mask into only its outer boundary.

    This prevents filled model/geometry blobs from being used directly as walls.
    """
    mask_u8 = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boundary = np.zeros_like(mask_u8)

    if contours:
        cv2.drawContours(
            boundary,
            contours,
            contourIdx=-1,
            color=255,
            thickness=max(1, thickness),
            lineType=cv2.LINE_AA,
        )

    return boundary > 0


def thicken_lines(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """
    Thicken a line-style mask without filling room interiors.
    """
    mask_u8 = np.asarray(mask).astype(bool).astype(np.uint8)

    if thickness <= 1:
        return mask_u8.astype(bool)

    kernel = np.ones((thickness, thickness), np.uint8)
    out = cv2.dilate(mask_u8, kernel, iterations=1)

    return out.astype(bool)


def print_obs_stats(obs: np.ndarray) -> None:
    print("[DEBUG] observation channel stats:")
    for i, name in enumerate(["occupancy", "structure", "furniture", "rgb_r", "rgb_g", "rgb_b"]):
        ch = obs[..., i]
        print(f"  {i} {name}: min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.6f}, nonzero={np.count_nonzero(ch)}")


def main():
    ap = argparse.ArgumentParser(description="Run full Structured3D scene -> fused BEV -> model prediction -> downloadable SVG floorplan.")
    ap.add_argument("--scene-dir", required=True, help="Path to a Structured3D scene folder, e.g. project_input/scene_00400")
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt")
    ap.add_argument("--output-dir", required=True, help="Output folder for PNG/SVG results")
    ap.add_argument("--model", default="auto", choices=["auto", "unet", "unet_attention", "unet_plusplus"])
    ap.add_argument("--variants", default="empty,simple,full", help="Comma-separated panorama variants to use. Default uses all major variants.")
    ap.add_argument("--rgb-name", default="rgb_rawlight.png")
    ap.add_argument("--grid-size", type=int, default=768, help="Use 768 or 1024 for full-scene output; 512 is OK for quick tests.")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--z-min", type=float, default=-0.5)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--height-lo", type=float, default=0.15)
    ap.add_argument("--height-hi", type=float, default=1.60)
    ap.add_argument("--xy-padding-m", type=float, default=0.75)
    ap.add_argument("--meters-per-pixel", type=float, default=None)
    ap.add_argument("--splat-radius", type=int, default=2)

    ap.add_argument("--structure-min-area", type=int, default=160)
    ap.add_argument("--furniture-min-area", type=int, default=100)
    ap.add_argument("--svg-min-area", type=float, default=50.0)
    ap.add_argument("--svg-simplify", type=float, default=1.25)
    ap.add_argument("--no-furniture-svg", action="store_true")

    ap.add_argument("--geometry-threshold", type=float, default=0.35, help="Threshold for the fused geometric BEV structure prior.")
    ap.add_argument("--annotation-guide", action="store_true", default=True, help="Use annotation_3d.json as a floorplan guide when it exists.")
    ap.add_argument("--no-annotation-guide", dest="annotation_guide", action="store_false")
    ap.add_argument("--annotation-line-thickness", type=int, default=5)

    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("SCENE TO FLOORPLAN INFERENCE")
    print(f"Scene dir: {scene_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Variants: {variants}")
    print(f"Device: {device}")
    print("=" * 80)

    annotation_xy = load_annotation_xy(scene_dir) if args.annotation_guide else None
    pts, colors, sem, sources = collect_scene_points(scene_dir, variants=variants, rgb_name=args.rgb_name, stride=args.stride)

    bev = make_scene_bev(
        points_xyz=pts,
        semantic=sem,
        colors=colors,
        grid_size=args.grid_size,
        z_clip=(args.z_min, args.z_max),
        height_band_above_floor=(args.height_lo, args.height_hi),
        xy_padding_m=args.xy_padding_m,
        splat_radius=args.splat_radius,
        meters_per_pixel=args.meters_per_pixel,
        annotation_xy_m=annotation_xy,
    )
    obs = stack_observation(bev)
    print_obs_stats(obs)

    if np.count_nonzero(obs[..., 1]) == 0:
        raise ValueError("Fused BEV structure channel is empty. Check semantic loading and variant paths.")

    model, model_name = load_model(args.checkpoint, args.model, device=device)
    structure_prob, furniture_prob, raw_structure_mask, raw_furniture_mask = predict_masks(model, obs, device=device, threshold=args.threshold)

    # Clean model outputs.
    # Keep this as a debug/model-only output, but do not use the filled version directly
    # as the final wall mask.
    model_structure_filled = clean_mask(
        structure_prob,
        threshold=args.threshold,
        close_kernel=5,
        open_kernel=3,
        min_area_px=args.structure_min_area,
        fill_holes=True,
    )
    
    model_furniture = clean_mask(
        furniture_prob,
        threshold=args.threshold,
        close_kernel=3,
        open_kernel=3,
        min_area_px=args.furniture_min_area,
        fill_holes=False,
    )
    
    # Geometry prior from the fused BEV structure channel.
    # This can also become blob-like, so it must be converted to boundaries before fusion.
    geometry_structure_filled = clean_mask(
        obs[..., 1],
        threshold=args.geometry_threshold,
        close_kernel=7,
        open_kernel=3,
        min_area_px=args.structure_min_area,
        fill_holes=True,
    )
    
    annotation_structure = (
        annotation_wall_mask(
            scene_dir,
            bev,
            grid_size=args.grid_size,
            thickness_px=args.annotation_line_thickness,
        )
        if args.annotation_guide
        else None
    )
    
    # Convert filled masks to boundary masks.
    # This is the key change that prevents the final floorplan from becoming a filled blob.
    model_structure_boundary = mask_to_boundary(
        model_structure_filled,
        thickness=max(2, args.annotation_line_thickness // 2),
    )
    
    geometry_structure_boundary = mask_to_boundary(
        geometry_structure_filled,
        thickness=max(2, args.annotation_line_thickness // 2),
    )
    
    if annotation_structure is not None:
        print(f"[ANNOTATION] using annotation_3d wall guide, nonzero={np.count_nonzero(annotation_structure)}")
    
        # Annotation is already a wall/line guide, so do not fill it.
        annotation_structure = thicken_lines(
            annotation_structure,
            thickness=max(1, args.annotation_line_thickness),
        )
    
        final_structure = np.logical_or.reduce([
            annotation_structure,
            model_structure_boundary,
            geometry_structure_boundary,
        ])
    else:
        print("[ANNOTATION] no annotation guide used/found")
    
        final_structure = np.logical_or.reduce([
            model_structure_boundary,
            geometry_structure_boundary,
        ])
    
    # Clean line mask gently.
    # IMPORTANT: fill_holes must stay False.
    # If fill_holes=True, rooms/building interiors get filled and the floorplan becomes a blob.
    final_structure = clean_mask(
        final_structure.astype(np.float32),
        threshold=0.5,
        close_kernel=3,
        open_kernel=0,
        min_area_px=20,
        fill_holes=False,
    )
    
    final_furniture = model_furniture
    
    # Keep this variable name so the later model-only saving code still works.
    model_structure = model_structure_boundary
    geometry_structure = geometry_structure_boundary

    # Save arrays and previews.
    np.save(out_dir / "scene_observation.npy", obs)
    np.save(out_dir / "scene_structure_probability.npy", structure_prob.astype(np.float32))
    np.save(out_dir / "scene_furniture_probability.npy", furniture_prob.astype(np.float32))

    preview_target = np.stack([final_structure.astype(np.float32), final_furniture.astype(np.float32)], axis=0)
    save_bev_preview(obs, preview_target, out_dir / "scene_bev_observation_preview.png")

    save_mask_png(model_structure_filled, out_dir / "clean_model_structure_filled_mask.png")
    save_mask_png(model_structure_boundary, out_dir / "model_structure_boundary_mask.png")
    
    save_mask_png(model_furniture, out_dir / "clean_model_furniture_mask.png")
    
    save_mask_png(geometry_structure_filled, out_dir / "geometry_structure_filled_prior_mask.png")
    save_mask_png(geometry_structure_boundary, out_dir / "geometry_structure_boundary_mask.png")
    
    if annotation_structure is not None:
        save_mask_png(annotation_structure, out_dir / "annotation_structure_guide_mask.png")
    
    save_mask_png(final_structure, out_dir / "final_structure_mask.png")
    save_mask_png(final_furniture, out_dir / "final_furniture_mask.png")

    save_floorplan_overlay_png(model_structure, model_furniture, out_dir / "model_floorplan_preview.png")
    save_floorplan_overlay_png(final_structure, final_furniture, out_dir / "floorplan_preview.png")

    meters_per_pixel = float(bev["meters_per_pixel"][0])
    save_floorplan_svg(
        structure_mask=model_structure,
        furniture_mask=model_furniture,
        out_svg=out_dir / "model_floorplan.svg",
        meters_per_pixel=meters_per_pixel,
        min_area_px=args.svg_min_area,
        simplify_px=args.svg_simplify,
        include_furniture=not args.no_furniture_svg,
        title="Model-only scene floorplan",
    )
    save_floorplan_svg(
        structure_mask=final_structure,
        furniture_mask=final_furniture,
        out_svg=out_dir / "floorplan.svg",
        meters_per_pixel=meters_per_pixel,
        min_area_px=args.svg_min_area,
        simplify_px=args.svg_simplify,
        include_furniture=not args.no_furniture_svg,
        title="Fused scene floorplan",
    )

    metadata = {
        "scene_dir": str(scene_dir),
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "device": device,
        "variants": variants,
        "grid_size": args.grid_size,
        "stride": args.stride,
        "threshold": args.threshold,
        "meters_per_pixel": meters_per_pixel,
        "origin_xy": bev["origin_xy"].tolist(),
        "center_xy": bev["center_xy"].tolist(),
        "floor_z": float(bev["floor_z"][0]),
        "height_band_above_floor": bev["height_band_above_floor"].tolist(),
        "annotation_guide_used": annotation_structure is not None,
        "final_structure_nonzero": int(np.count_nonzero(final_structure)),
        "model_structure_boundary_nonzero": int(np.count_nonzero(model_structure_boundary)),
        "geometry_structure_boundary_nonzero": int(np.count_nonzero(geometry_structure_boundary)),
        "annotation_structure_nonzero": int(np.count_nonzero(annotation_structure)) if annotation_structure is not None else 0,
        "source_count": len(sources),
        "sources": sources,
        "outputs": {
            "svg": str(out_dir / "floorplan.svg"),
            "model_svg": str(out_dir / "model_floorplan.svg"),
            "preview_png": str(out_dir / "floorplan_preview.png"),
            "bev_preview_png": str(out_dir / "scene_bev_observation_preview.png"),
        },
    }
    (out_dir / "scene_inference_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 80)
    print("DONE")
    print(f"Final SVG floorplan: {out_dir / 'floorplan.svg'}")
    print(f"Model-only SVG: {out_dir / 'model_floorplan.svg'}")
    print(f"Preview PNG: {out_dir / 'floorplan_preview.png'}")
    print(f"BEV preview: {out_dir / 'scene_bev_observation_preview.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
