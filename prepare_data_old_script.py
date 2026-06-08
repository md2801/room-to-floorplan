from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from PIL import Image
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from render_structured3d_roomtofp_floorplan import (
    build_roomtofp_scene_outputs,
    grid_from_bev,
    load_annotation,
    parse_room_polygons,
    cleanup_room_labels,
    xy_to_poly_pixels,
)
from room_to_floorplan.roomtofp_scene_utils import (
    collect_scene_points,
    load_annotation_xy,
    make_scene_bev,
    stack_roomtofp_observation,
)

ROOM_TYPES = [
    "undefined",
    "living room",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "corridor",
    "dining room",
    "study",
    "office",
    "storage",
    "other",
]
ROOM_TYPE_TO_ID = {name: i for i, name in enumerate(ROOM_TYPES)}

OBSERVATION_CHANNEL_ORDER = ["occupancy", "wall", "floor", "opening", "furniture", "rgb_r", "rgb_g", "rgb_b"]


def extract_zips(zip_paths: list[str], extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    for zp in zip_paths:
        zpath = Path(zp)
        print(f"--Extracting-- {zpath} -> {extract_root}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_root)
    return extract_root


def find_scene_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("scene_*") if p.is_dir() and (p / "annotation_3d.json").exists()])


def canonicalize_polygon_pixels(points: np.ndarray, max_corners: int | None = None, simplify_eps_px: float = 1.0) -> np.ndarray:
    """
    Improvement v4.1: normalise target polygon vertices before padding/rasterising.
    This removes duplicate points, simplifies tiny zigzags, enforces a consistent clockwise order, and starts the sequence at the top-left-most corner.
    Cleaner target sequences are critical for RoomFormer-like ordered polygon learning.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.float32)

    rounded = np.round(pts).astype(np.int32)
    _, keep_idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[np.sort(keep_idx)]
    if len(pts) < 3:
        return pts.astype(np.float32)

    if simplify_eps_px > 0 and len(pts) > 3:
        approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), float(simplify_eps_px), True).reshape(-1, 2)
        if len(approx) >= 3:
            pts = approx.astype(np.float32)

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    # Image coordinates have y downward; this signed area check gives a consistent clockwise-like winding in pixel space.
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if signed_area < 0:
        pts = pts[::-1]

    start = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    pts = np.roll(pts, -start, axis=0)

    if max_corners is not None and len(pts) > max_corners:
        # Uniformly sample around the polygon instead of taking the first N corners only.
        idx = np.linspace(0, len(pts) - 1, int(max_corners), dtype=np.int32)
        pts = pts[idx]

    return pts.astype(np.float32)


def pad_polygons(
        rooms: list[dict],
        bev: dict,
        grid_size: int,
        max_rooms: int,
        max_corners: int
):
    """
    converts a variable num of polygons into fixed size training arrays that model learns from
    """
    grid = grid_from_bev(bev)

    corners = np.zeros((max_rooms, max_corners, 2), dtype=np.float32)
    corner_valid = np.zeros((max_rooms, max_corners), dtype=np.float32)

    room_valid = np.zeros((max_rooms,), dtype=np.float32)
    room_class = np.zeros((max_rooms,), dtype=np.int64)
    room_area = np.zeros((max_rooms,), dtype=np.float32)

    # Larger polygons first makes the fixed-size target more stable if a scene has more than max_rooms.
    rooms_sorted = sorted(rooms, key=lambda r: float(r.get("area_m2", 0.0)), reverse=True)
    for ri, room in enumerate(rooms_sorted[:max_rooms]):
        pts = xy_to_poly_pixels(room["xy"], grid, grid_size).astype(np.float32)
        pts = canonicalize_polygon_pixels(pts, max_corners=max_corners, simplify_eps_px=1.0)

        # normalizing xy coords into range 0-1
        pts[:, 0] /= max(1, grid_size - 1)
        pts[:, 1] /= max(1, grid_size - 1)

        # limiting num of corners after Improvement v4.1 canonicalisation/uniform downsampling
        n = min(len(pts), max_corners)
        corners[ri, :n] = pts[:n]  # stores first n corner coords in fixed-sized array
        corner_valid[ri, :n] = 1.0  # marking valid corners

        room_valid[ri] = 1.0  # marking room slot as valid
        room_class[ri] = ROOM_TYPE_TO_ID.get(room["type"], ROOM_TYPE_TO_ID["other"])  # stores room class as numeric
        room_area[ri] = float(room.get("area_m2", 0.0))  # stores room area

    return corners, corner_valid, room_valid, room_class, room_area


def rasterize_room_instance_targets(rooms: list[dict], bev: dict, grid_size: int, max_rooms: int, max_corners: int, corner_radius_px: int = 4):
    """
    Improvement v3: rasterise each annotation room polygon into per-query room masks, a shared boundary mask, and a corner heatmap.

    These targets let the Transformer room queries learn full room extents, not only corner coordinates.
    The room mask target is used for query-mask supervision, while boundary/corner heatmaps provide dense structural hints.
    """
    grid = grid_from_bev(bev)
    room_masks = np.zeros((max_rooms, grid_size, grid_size), dtype=np.float32)
    boundary_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    corner_heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    rooms_sorted = sorted(rooms, key=lambda r: float(r.get("area_m2", 0.0)), reverse=True)
    for ri, room in enumerate(rooms_sorted[:max_rooms]):
        pts = xy_to_poly_pixels(room["xy"], grid, grid_size).astype(np.float32)
        pts = canonicalize_polygon_pixels(pts, max_corners=max_corners, simplify_eps_px=1.0).astype(np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(room_masks[ri], [pts], 1.0)
        cv2.polylines(boundary_mask, [pts], True, 1.0, thickness=3, lineType=cv2.LINE_AA)
        for x, y in pts[:max_corners]:
            cv2.circle(corner_heatmap, (int(x), int(y)), int(corner_radius_px), 1.0, thickness=-1)

    boundary_mask = np.clip(boundary_mask, 0.0, 1.0).astype(np.float32)
    corner_heatmap = np.clip(cv2.GaussianBlur(corner_heatmap, (0, 0), 1.0), 0.0, 1.0).astype(np.float32)
    return room_masks, boundary_mask[None, ...], corner_heatmap[None, ...]


def load_preview_outputs(preview_out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Load auxiliary masks created by the renderer.

    Furniture is optional and disabled by default in the floorplan-first pipeline.
    When no furniture mask exists, we return an all-zero mask with the same shape
    as the wall/opening target so older dataset/model code stays compatible.

    - Loads the wall mask from .npy or PNG.
    - Requires wall mask because it is central to floorplan training.
    - Loads the opening mask if available, otherwise creates a zero mask.
    - Loads the furniture mask if available, otherwise creates a zero mask.
    - Reads metadata.json if present.
    - Extracts furniture component metadata.
    - Adds a channel dimension to each mask.
    - Returns furniture, wall, opening masks, and component list.
    """
    # Check for wall masks; if exists, convert to float32; else fallback to PNG and conv to grayscale numpy array binary mask
    if (preview_out / "wall_mask.npy").exists():
        wall_mask = np.load(preview_out / "wall_mask.npy").astype(np.float32)
    elif (preview_out / "walls_mask.png").exists():
        wall_mask = (np.array(Image.open(preview_out / "walls_mask.png").convert("L")) > 0).astype(np.float32)
    else:
        raise FileNotFoundError(f"Missing wall mask in preview folder: {preview_out}")

    # Check for opening mask
    if (preview_out / "opening_mask.npy").exists():
        opening_mask = np.load(preview_out / "opening_mask.npy").astype(np.float32)
    elif (preview_out / "opening_mask.png").exists():
        opening_mask = (np.array(Image.open(preview_out / "opening_mask.png").convert("L")) > 0).astype(np.float32)
    else:
        opening_mask = np.zeros_like(wall_mask, dtype=np.float32)

    # Check furniture mask
    if (preview_out / "furniture_mask.npy").exists():
        furniture_mask = np.load(preview_out / "furniture_mask.npy").astype(np.float32)
    elif (preview_out / "furniture_mask.png").exists():
        furniture_mask = (np.array(Image.open(preview_out / "furniture_mask.png").convert("L")) > 0).astype(np.float32)
    else:
        furniture_mask = np.zeros_like(wall_mask, dtype=np.float32)

    # Load metadata
    metadata = json.loads((preview_out / "metadata.json").read_text(encoding="utf-8")) if (preview_out / "metadata.json").exists() else {}
    components = metadata.get("furniture_components", [])  # extract furniture components

    # ... -> ellipsis: adds None to the front of the mask and keeps rest as is.
    return furniture_mask[None, ...], wall_mask[None, ...], opening_mask[None, ...], components


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare RoomToFP polygon training data from Structured3D panorama scenes.")
    ap.add_argument(
        "--structured3d-root",
        nargs="+",
        default=None,
        help="One or more unzipped Structured3D panorama roots or parent folders containing scene_* folders.",
    )
    # Dataset args
    ap.add_argument("--panorama-zips", nargs="*", default=[], help="Optional Structured3D panorama subset zips to extract first")
    ap.add_argument("--extract-dir", default="data/structured3d_panorama_extracted")
    ap.add_argument("--output", default="data/roomtofp_polygon")

    # Dataset size & polygon args
    ap.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes")
    ap.add_argument("--max-rooms", type=int, default=20)
    ap.add_argument("--max-corners", type=int, default=40)

    # Scene rendering/view args
    ap.add_argument("--variants", default="full")
    ap.add_argument("--rgb-name", default="rgb_rawlight.png")

    # BEV generation args
    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--z-min", type=float, default=-0.5)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--height-lo", type=float, default=0.15)
    ap.add_argument("--height-hi", type=float, default=1.60)
    ap.add_argument("--xy-padding-m", type=float, default=0.75)
    ap.add_argument("--splat-radius", type=int, default=2)
    ap.add_argument("--wall-thickness", type=int, default=5)

    # Room filtering & label args
    ap.add_argument("--min-room-area-m2", type=float, default=0.20)
    ap.add_argument("--exclude-balcony", action="store_true")
    ap.add_argument("--keep-undefined-label", action="store_true", help="Keep raw Structured3D 'undefined' room labels instead of remapping them to 'other'.")

    # Opening args
    ap.add_argument("--min-opening-length-m", type=float, default=0.35)
    ap.add_argument("--max-opening-wall-distance-m", type=float, default=0.20)

    # Furniture args
    ap.add_argument("--min-furniture-area-px", type=int, default=120)
    ap.add_argument("--min-furniture-width-px", type=int, default=8)
    ap.add_argument("--min-furniture-height-px", type=int, default=8)
    ap.add_argument("--max-furniture-aspect-ratio", type=float, default=12.0)
    ap.add_argument("--furniture-room-erosion-px", type=int, default=1)
    ap.add_argument("--include-furniture", action="store_true", help="Generate furniture masks/components. Off by default so the main dataset focuses on floorplan structure.")
    ap.add_argument("--furniture-source", choices=["auto", "annotation", "semantic", "both"], default="auto", help="Furniture source used only when --include-furniture is passed.")

    # Split & utility args
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true", help="Do not regenerate existing .npz samples")
    ap.add_argument("--keep-previews", action="store_true", help="Keep per-scene preview folders. Otherwise only first few are kept.")

    args = ap.parse_args()

    if not args.include_furniture:
        args.furniture_source = "off"

    # If zips provided, extract them, else get path for each panorama folder provided
    here = Path(__file__).resolve().parent
    if args.panorama_zips:
        roots = [extract_zips(args.panorama_zips, (here / args.extract_dir).resolve())]
    elif args.structured3d_root:
        roots = [Path(p).resolve() for p in args.structured3d_root]
    else:
        raise ValueError("Provide either --structured3d-root or --panorama-zips")

    out = (here / args.output).resolve()
    samples_dir = out / "samples"
    previews_dir = out / "previews"
    samples_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs: list[Path] = []
    seen_scene_paths: set[str] = set()
    for root in roots:
        found = find_scene_dirs(root)
        print(f"--Scene Info-- {root}: found {len(found)} scenes")
        for scene_dir in found:
            key = str(scene_dir.resolve())
            if key not in seen_scene_paths:
                scene_dirs.append(scene_dir)
                seen_scene_paths.add(key)
    scene_dirs = sorted(scene_dirs)
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]
    print(f"--Scene Info-- Total unique scenes: {len(scene_dirs)}")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    records = []

    # Inspired by RoomFormer: supervise polygons from annotation_3d.json, not only raster masks.
    for idx, scene_dir in enumerate(tqdm(scene_dirs, desc="prepare RoomToFP polygon data")):
        sample_name = f"{scene_dir.name}.npz"
        sample_path = samples_dir / sample_name

        if args.skip_existing and sample_path.exists():
            records.append({"scene": scene_dir.name, "sample": str(sample_path), "skipped_existing": True})
            continue

        try:
            annotation = load_annotation(scene_dir)
            rooms = parse_room_polygons(annotation, min_area_m2=args.min_room_area_m2, keep_balcony=not args.exclude_balcony)
            rooms = cleanup_room_labels(rooms, keep_undefined=getattr(args, "keep_undefined_label", False))
            if not rooms:
                continue

            pts, colors, sem, _sources = collect_scene_points(scene_dir, variants=variants, rgb_name=args.rgb_name, stride=args.stride)
            bev = make_scene_bev(
                points_xyz=pts,
                semantic=sem,
                colors=colors,
                grid_size=args.grid_size,
                z_clip=(args.z_min, args.z_max),
                height_band_above_floor=(args.height_lo, args.height_hi),
                xy_padding_m=args.xy_padding_m,
                splat_radius=args.splat_radius,
                annotation_xy_m=load_annotation_xy(scene_dir),
            )
            obs = stack_roomtofp_observation(bev).astype(np.float32)
            if not args.include_furniture and obs.shape[-1] >= 5:
                # Keep the 8-channel shape for compatibility, but zero the furniture evidence
                # so the floorplan model is not trained to rely on noisy furniture projections.
                obs[..., 4] = 0.0

            preview_out = previews_dir / scene_dir.name
            metadata = build_roomtofp_scene_outputs(scene_dir, preview_out, variants, args)

            furniture_mask, wall_mask, opening_mask, furniture_components = load_preview_outputs(preview_out)
            corners, corner_valid, room_valid, room_class, room_area = pad_polygons(
                rooms, bev, args.grid_size, args.max_rooms, args.max_corners
            )
            room_masks, boundary_mask, corner_heatmap = rasterize_room_instance_targets(
                rooms, bev, args.grid_size, args.max_rooms, args.max_corners
            )

            np.savez_compressed(
                sample_path,
                observation=obs,
                polygon_xy=corners,
                corner_valid=corner_valid,
                room_valid=room_valid,
                room_class=room_class,
                room_area=room_area,
                furniture_mask=furniture_mask.astype(np.float32),
                wall_mask=wall_mask.astype(np.float32),
                opening_mask=opening_mask.astype(np.float32),
                # Improvement v3: save room instance masks and dense structural heatmaps for query-mask and boundary/corner supervision.
                room_mask=room_masks.astype(np.float32),
                boundary_mask=boundary_mask.astype(np.float32),
                corner_heatmap=corner_heatmap.astype(np.float32),
            )
            records.append(
                {
                    "scene": scene_dir.name,
                    "sample": str(sample_path),
                    "sample_name": sample_name,
                    "num_rooms": int(room_valid.sum()),
                    "num_openings": int(metadata.get("num_openings", 0)),
                    "num_furniture_components": int(metadata.get("num_furniture_components", len(furniture_components))),
                    "include_furniture": bool(getattr(args, "include_furniture", False)),
                    "furniture_source": metadata.get("furniture_source", getattr(args, "furniture_source", "off")),
                    "observation_channels": int(obs.shape[-1]),
                }
            )

            if not args.keep_previews and idx >= 10 and preview_out.exists():
                shutil.rmtree(preview_out, ignore_errors=True)

        except Exception as e:
            print(f"--Warning-- Skipped {scene_dir}: {type(e).__name__}: {e}")

    usable_records = [r for r in records if r.get("sample") and Path(r.get("sample", "")).exists()]
    sample_names = [Path(r["sample"]).name for r in usable_records]
    rng = random.Random(int(args.split_seed))
    rng.shuffle(sample_names)

    total_ratio = max(1e-8, float(args.train_ratio + args.val_ratio + args.test_ratio))
    train_ratio = float(args.train_ratio) / total_ratio
    val_ratio = float(args.val_ratio) / total_ratio

    n = len(sample_names)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    splits = {
        "train": sample_names[:n_train],
        "val": sample_names[n_train:n_train + n_val],
        "test": sample_names[n_train + n_val:],
    }

    if n >= 3 and not splits["test"] and splits["val"]:
        splits["test"] = splits["val"][-1:]
        splits["val"] = splits["val"][:-1]

    if n >= 2 and not splits["val"] and splits["train"]:
        splits["val"] = splits["train"][-1:]
        splits["train"] = splits["train"][:-1]

    for split_name, names in splits.items():
        (out / f"{split_name}.txt").write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    (out / "metadata.json").write_text(
        json.dumps(
            {
                "room_types": ROOM_TYPES,
                "observation_channel_order": OBSERVATION_CHANNEL_ORDER,
                "include_furniture": bool(args.include_furniture),
                "split_seed": int(args.split_seed),
                "split_counts": {k: len(v) for k, v in splits.items()},
                "splits": splits,
                "samples": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Dataset Preprocessing Complete: {len(records)} samples written to {samples_dir}")
    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
