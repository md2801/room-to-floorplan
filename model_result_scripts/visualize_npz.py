from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Basic helpers
def norm01(x: np.ndarray) -> np.ndarray:
    """
    Normalize an array into [0, 1].
    """
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x

    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))

    if mx - mn <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mn) / (mx - mn)).astype(np.float32)


def as_hw(x: np.ndarray) -> np.ndarray:
    """
    Convert common mask shapes into [H, W].

    Supports:
        [H, W]
        [1, H, W]
        [H, W, 1]
    """
    x = np.asarray(x)

    if x.ndim == 2:
        return x

    if x.ndim == 3:
        if x.shape[0] == 1:
            return x[0]
        if x.shape[-1] == 1:
            return x[..., 0]

    raise ValueError(f"Cannot convert shape to HW: {x.shape}")


def as_chw_bev(data: Dict[str, Any]) -> np.ndarray:
    """
    Returns BEV as [C, H, W].

    Preferred key:
        bev

    Fallback:
        observation [H, W, C]
    """
    if "bev" in data:
        bev = np.asarray(data["bev"], dtype=np.float32)

        if bev.ndim != 3:
            raise ValueError(f"bev must be 3D. Got: {bev.shape}")

        # Already CHW.
        if bev.shape[0] <= 32:
            return bev

        # HWC fallback.
        return np.transpose(bev, (2, 0, 1)).astype(np.float32)

    if "observation" in data:
        obs = np.asarray(data["observation"], dtype=np.float32)

        if obs.ndim != 3:
            raise ValueError(f"observation must be 3D. Got: {obs.shape}")

        # HWC.
        if obs.shape[-1] <= 32:
            return np.transpose(obs, (2, 0, 1)).astype(np.float32)

        # CHW.
        return obs.astype(np.float32)

    raise KeyError("Sample has neither 'bev' nor 'observation'.")


def to_uint8_rgb(x: np.ndarray) -> np.ndarray:
    """
    Convert [H, W] or [H, W, 3] float image into uint8 RGB.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = norm01(x)
        u = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return np.stack([u, u, u], axis=-1)

    if x.ndim == 3 and x.shape[-1] == 3:
        if float(x.max()) <= 1.5:
            x = x * 255.0
        return np.clip(x, 0, 255).astype(np.uint8)

    raise ValueError(f"Cannot convert to RGB: {x.shape}")


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Overlay a single-channel mask onto an RGB image.
    """
    img = image.astype(np.float32).copy()
    m = as_hw(mask).astype(np.float32)
    m = norm01(m)
    active = m > threshold

    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)

    img[active] = (1.0 - alpha) * img[active] + alpha * color_arr

    return np.clip(img, 0, 255).astype(np.uint8)


def draw_text_bar(
    img: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    bar_height: int = 34,
) -> np.ndarray:
    """
    Add a black title bar above an image.
    """
    h, w = img.shape[:2]
    out = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    out[bar_height:] = img

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        small = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw.text((6, 4), title, fill=(255, 255, 255), font=font)

    if subtitle:
        draw.text((6, 19), subtitle, fill=(190, 190, 190), font=small)

    return np.asarray(pil)


def resize_panel(img: np.ndarray, panel_size: int) -> np.ndarray:
    """
    Resize panel to square.
    """
    return cv2.resize(img, (panel_size, panel_size), interpolation=cv2.INTER_AREA)


def hstack_with_gap(images: List[np.ndarray], gap: int = 6) -> np.ndarray:
    """
    Horizontally stack images with a gap.
    """
    h = max(img.shape[0] for img in images)
    w = sum(img.shape[1] for img in images) + gap * (len(images) - 1)

    out = np.full((h, w, 3), 18, dtype=np.uint8)

    x = 0
    for img in images:
        out[: img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + gap

    return out


def vstack_with_gap(images: List[np.ndarray], gap: int = 8) -> np.ndarray:
    """
    Vertically stack images with a gap.
    """
    h = sum(img.shape[0] for img in images) + gap * (len(images) - 1)
    w = max(img.shape[1] for img in images)

    out = np.full((h, w, 3), 18, dtype=np.uint8)

    y = 0
    for img in images:
        out[y : y + img.shape[0], : img.shape[1]] = img
        y += img.shape[0] + gap

    return out


def make_instance_color(index: int) -> Tuple[int, int, int]:
    """
    Deterministic color for room instance visualization.
    """
    palette = [
        (88, 180, 255),
        (96, 220, 130),
        (255, 190, 80),
        (255, 110, 130),
        (190, 130, 255),
        (120, 230, 230),
        (230, 230, 100),
        (255, 150, 220),
        (160, 220, 90),
        (100, 150, 255),
        (230, 160, 110),
        (150, 255, 180),
    ]
    return palette[index % len(palette)]


# Sample extraction helpers
def get_mask(data: Dict[str, Any], key: str, fallback_shape: Tuple[int, int]) -> np.ndarray:
    """
    Safe mask getter.
    """
    if key not in data:
        return np.zeros(fallback_shape, dtype=np.float32)

    arr = np.asarray(data[key], dtype=np.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]

    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]

    # If someone stored per-room masks under a merged key, merge them.
    if arr.ndim == 3:
        return np.clip(np.sum(arr, axis=0), 0, 1).astype(np.float32)

    return np.zeros(fallback_shape, dtype=np.float32)


def get_room_masks(data: Dict[str, Any], fallback_shape: Tuple[int, int]) -> np.ndarray:
    """
    Return per-room masks [R, H, W].

    Supports both old and new keys:
        room_masks = per-room
        room_mask  = merged or per-room depending on old scripts
    """
    if "room_masks" in data:
        arr = np.asarray(data["room_masks"], dtype=np.float32)
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            return arr[None, ...]

    if "room_mask" in data:
        arr = np.asarray(data["room_mask"], dtype=np.float32)
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            return arr[None, ...]

    return np.zeros((0, fallback_shape[0], fallback_shape[1]), dtype=np.float32)


def get_room_valid(data: Dict[str, Any], room_count: int) -> np.ndarray:
    """
    Return room_valid mask.
    """
    if "room_valid" in data:
        rv = np.asarray(data["room_valid"], dtype=np.float32).reshape(-1)
        if len(rv) >= room_count:
            return rv[:room_count]
        out = np.zeros((room_count,), dtype=np.float32)
        out[: len(rv)] = rv
        return out

    if "num_rooms" in data:
        n = int(np.asarray(data["num_rooms"]).reshape(-1)[0])
        out = np.zeros((room_count,), dtype=np.float32)
        out[: min(n, room_count)] = 1.0
        return out

    return np.ones((room_count,), dtype=np.float32)


def draw_boundary_points(
    img: np.ndarray,
    points_norm: np.ndarray,
    color: Tuple[int, int, int],
    closed: bool = True,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw normalized polygon/boundary points on an RGB image.
    """
    out = img.copy()
    h, w = out.shape[:2]

    pts = np.asarray(points_norm, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return out

    # If points are normalized, scale them.
    if float(np.nanmax(pts)) <= 1.5:
        pts_px = pts.copy()
        pts_px[:, 0] *= w - 1
        pts_px[:, 1] *= h - 1
    else:
        pts_px = pts

    pts_px[:, 0] = np.clip(pts_px[:, 0], 0, w - 1)
    pts_px[:, 1] = np.clip(pts_px[:, 1], 0, h - 1)

    pts_int = np.round(pts_px).astype(np.int32)

    cv2.polylines(
        out,
        [pts_int],
        isClosed=closed,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return out


# Quality checks
def connected_components_count(mask: np.ndarray) -> int:
    """
    Count connected components in a binary mask, excluding background.
    """
    m = as_hw(mask)
    m = (m > 0.5).astype(np.uint8)

    num, _labels = cv2.connectedComponents(m, connectivity=8)

    return max(0, int(num) - 1)


def compute_quality_stats(npz_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute useful dataset sanity metrics for one sample.
    """
    bev = as_chw_bev(data)
    _, h, w = bev.shape

    room_masks = get_room_masks(data, (h, w))
    room_valid = get_room_valid(data, room_masks.shape[0])

    valid_indices = np.where(room_valid > 0.5)[0]
    valid_masks = room_masks[valid_indices] if len(valid_indices) else np.zeros((0, h, w), dtype=np.float32)

    room_sum = np.sum(valid_masks > 0.5, axis=0) if len(valid_masks) else np.zeros((h, w), dtype=np.float32)
    overlap_pixels = int(np.sum(room_sum > 1))
    room_pixels = int(np.sum(room_sum > 0))
    overlap_pct = 0.0 if room_pixels == 0 else 100.0 * overlap_pixels / max(1, room_pixels)

    empty_valid_rooms = 0
    multi_component_rooms = 0

    for rm in valid_masks:
        if float(np.sum(rm > 0.5)) <= 0:
            empty_valid_rooms += 1
            continue

        cc = connected_components_count(rm)
        if cc > 1:
            multi_component_rooms += 1

    stats = {
        "file": npz_path.name,
        "source": str(np.asarray(data.get("source", "unknown"))),
        "bev_shape": list(bev.shape),
        "valid_rooms": int(len(valid_indices)),
        "room_pixels": int(room_pixels),
        "overlap_pixels": int(overlap_pixels),
        "overlap_pct": float(overlap_pct),
        "empty_valid_rooms": int(empty_valid_rooms),
        "multi_component_rooms": int(multi_component_rooms),
        "has_input_corner_prior": "input_corner_prior" in data,
        "has_distance_to_wall_target": "distance_to_wall_target" in data,
    }

    if "corner_heatmap" in data and bev.shape[0] > 8:
        ch = get_mask(data, "corner_heatmap", (h, w))
        stats["max_abs_bev8_minus_corner_heatmap"] = float(np.max(np.abs(bev[8] - ch)))

    if "input_corner_prior" in data and bev.shape[0] > 8:
        inp = get_mask(data, "input_corner_prior", (h, w))
        stats["max_abs_bev8_minus_input_corner_prior"] = float(np.max(np.abs(bev[8] - inp)))

    if "distance_to_wall_target" in data and bev.shape[0] > 9:
        dt_t = get_mask(data, "distance_to_wall_target", (h, w))
        stats["max_abs_bev9_minus_distance_target"] = float(np.max(np.abs(bev[9] - dt_t)))

    if "distance_to_wall" in data and bev.shape[0] > 9:
        dt = get_mask(data, "distance_to_wall", (h, w))
        stats["max_abs_bev9_minus_distance_to_wall"] = float(np.max(np.abs(bev[9] - dt)))

    return stats


# Panel builders
def make_observed_input_panel(data: Dict[str, Any], panel_size: int) -> np.ndarray:
    """
    Panel 1:
        Observed model input channels.

    Visual convention:
        gray = occupancy
        red = observed wall
        green = observed floor
        blue = observed openings
        magenta = observed corner prior
    """
    bev = as_chw_bev(data)
    c, h, w = bev.shape

    occ = bev[0] if c > 0 else np.zeros((h, w), dtype=np.float32)
    wall = bev[1] if c > 1 else np.zeros((h, w), dtype=np.float32)
    floor = bev[2] if c > 2 else np.zeros((h, w), dtype=np.float32)
    opening = bev[3] if c > 3 else np.zeros((h, w), dtype=np.float32)
    corner_prior = bev[8] if c > 8 else np.zeros((h, w), dtype=np.float32)

    img = to_uint8_rgb(occ)
    img = overlay_mask(img, floor, color=(50, 220, 80), alpha=0.25, threshold=0.10)
    img = overlay_mask(img, wall, color=(255, 60, 40), alpha=0.60, threshold=0.10)
    img = overlay_mask(img, opening, color=(70, 130, 255), alpha=0.70, threshold=0.10)
    img = overlay_mask(img, corner_prior, color=(255, 0, 255), alpha=0.75, threshold=0.10)

    img = resize_panel(img, panel_size)

    return draw_text_bar(
        img,
        "Observed input BEV",
        "gray occ | red wall | green floor | blue opening | magenta corner prior",
    )


def make_target_mask_panel(data: Dict[str, Any], panel_size: int) -> np.ndarray:
    """
    Panel 2:
        Merged target masks.

    Visual convention:
        green = target room region
        red = target boundary/wall
        blue = target opening
        magenta = target corner heatmap
    """
    bev = as_chw_bev(data)
    _, h, w = bev.shape

    room_mask = get_mask(data, "room_mask", (h, w))
    boundary = get_mask(data, "edge_mask", (h, w))
    if float(boundary.max()) <= 1e-8:
        boundary = get_mask(data, "boundary_mask", (h, w))

    opening = get_mask(data, "opening_mask", (h, w))
    corner_heatmap = get_mask(data, "corner_heatmap", (h, w))

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img = overlay_mask(img, room_mask, color=(80, 220, 100), alpha=0.70, threshold=0.05)
    img = overlay_mask(img, boundary, color=(255, 60, 30), alpha=0.95, threshold=0.05)
    img = overlay_mask(img, opening, color=(40, 120, 255), alpha=0.95, threshold=0.05)
    img = overlay_mask(img, corner_heatmap, color=(255, 0, 255), alpha=0.85, threshold=0.05)

    img = resize_panel(img, panel_size)

    return draw_text_bar(
        img,
        "Target masks",
        "green room | red boundary | blue opening | magenta GT corners",
    )


def make_instance_polygon_panel(data: Dict[str, Any], panel_size: int) -> np.ndarray:
    """
    Panel 3:
        Per-room instance masks + dense polygon boundaries.
    """
    bev = as_chw_bev(data)
    _, h, w = bev.shape

    room_masks = get_room_masks(data, (h, w))
    room_valid = get_room_valid(data, room_masks.shape[0])

    img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(room_masks.shape[0]):
        if i >= len(room_valid) or room_valid[i] <= 0.5:
            continue

        color = make_instance_color(i)
        img = overlay_mask(img, room_masks[i], color=color, alpha=0.55, threshold=0.5)

    # Draw boundary_64 if available.
    if "boundary_64" in data:
        boundary_64 = np.asarray(data["boundary_64"], dtype=np.float32)

        for i in range(min(room_masks.shape[0], boundary_64.shape[0])):
            if i >= len(room_valid) or room_valid[i] <= 0.5:
                continue

            img = draw_boundary_points(
                img,
                boundary_64[i],
                color=(255, 255, 255),
                closed=True,
                thickness=1,
            )

    # Draw sparse polygon corners if available.
    if "polygon_xy" in data and "corner_valid" in data:
        polygon_xy = np.asarray(data["polygon_xy"], dtype=np.float32)
        corner_valid = np.asarray(data["corner_valid"], dtype=np.float32)

        for i in range(min(polygon_xy.shape[0], corner_valid.shape[0])):
            if i >= len(room_valid) or room_valid[i] <= 0.5:
                continue

            n = int(np.sum(corner_valid[i] > 0.5))
            if n >= 3:
                img = draw_boundary_points(
                    img,
                    polygon_xy[i, :n],
                    color=(0, 0, 0),
                    closed=True,
                    thickness=2,
                )

    img = resize_panel(img, panel_size)

    return draw_text_bar(
        img,
        "Room instances + polygons",
        "colored rooms | white boundary_64 | black sparse polygon",
    )


def make_leakage_debug_panel(data: Dict[str, Any], panel_size: int) -> np.ndarray:
    """
    Panel 4:
        Compare input-derived priors against GT targets.

    Left half:
        input corner prior and input distance-to-wall.

    Right half:
        GT corner heatmap and GT distance-to-wall target.
    """
    bev = as_chw_bev(data)
    _, h, w = bev.shape

    input_corner = get_mask(data, "input_corner_prior", (h, w))
    if float(input_corner.max()) <= 1e-8 and bev.shape[0] > 8:
        input_corner = bev[8]

    input_dist = get_mask(data, "distance_to_wall", (h, w))
    if float(input_dist.max()) <= 1e-8 and bev.shape[0] > 9:
        input_dist = bev[9]

    gt_corner = get_mask(data, "corner_heatmap", (h, w))
    gt_dist = get_mask(data, "distance_to_wall_target", (h, w))

    left = to_uint8_rgb(input_dist)
    left = overlay_mask(left, input_corner, color=(255, 0, 255), alpha=0.85, threshold=0.05)

    right = to_uint8_rgb(gt_dist)
    right = overlay_mask(right, gt_corner, color=(255, 0, 255), alpha=0.85, threshold=0.05)

    combined = np.concatenate([left, right], axis=1)
    combined = resize_panel(combined, panel_size)

    return draw_text_bar(
        combined,
        "Input prior vs GT target",
        "left=input-derived | right=GT target",
    )


def make_info_panel(
    npz_path: Path,
    data: Dict[str, Any],
    stats: Dict[str, Any],
    panel_size: int,
) -> np.ndarray:
    """
    Panel 5:
        Textual sanity summary.
    """
    img = np.full((panel_size, panel_size, 3), 28, dtype=np.uint8)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    y = 8
    lines = [
        npz_path.name,
        f"source: {stats.get('source', 'unknown')}",
        f"bev: {stats.get('bev_shape')}",
        f"valid rooms: {stats.get('valid_rooms')}",
        f"overlap: {stats.get('overlap_pct', 0):.2f}%",
        f"empty rooms: {stats.get('empty_valid_rooms')}",
        f"multi-comp rooms: {stats.get('multi_component_rooms')}",
    ]

    if "max_abs_bev8_minus_input_corner_prior" in stats:
        lines.append(f"bev8-input_prior: {stats['max_abs_bev8_minus_input_corner_prior']:.4f}")

    if "max_abs_bev8_minus_corner_heatmap" in stats:
        lines.append(f"bev8-GT_corner: {stats['max_abs_bev8_minus_corner_heatmap']:.4f}")

    if "max_abs_bev9_minus_distance_to_wall" in stats:
        lines.append(f"bev9-input_dist: {stats['max_abs_bev9_minus_distance_to_wall']:.4f}")

    if "max_abs_bev9_minus_distance_target" in stats:
        lines.append(f"bev9-GT_dist: {stats['max_abs_bev9_minus_distance_target']:.4f}")

    for idx, line in enumerate(lines):
        fill = (255, 255, 255) if idx == 0 else (210, 210, 210)
        draw.text((8, y), str(line), fill=fill, font=font if idx == 0 else small)
        y += 20 if idx == 0 else 17

    return np.asarray(pil)


def make_sample_row(npz_path: Path, panel_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a full visualization row for one .npz file.
    """
    loaded = np.load(npz_path, allow_pickle=True)
    data = {k: loaded[k] for k in loaded.files}

    stats = compute_quality_stats(npz_path, data)

    panels = [
        make_observed_input_panel(data, panel_size),
        make_target_mask_panel(data, panel_size),
        make_instance_polygon_panel(data, panel_size),
        make_leakage_debug_panel(data, panel_size),
        make_info_panel(npz_path, data, stats, panel_size),
    ]

    row = hstack_with_gap(panels, gap=6)

    return row, stats


# File discovery
def collect_npz_files(inputs: List[str], recursive: bool = True) -> List[Path]:
    """
    Collect .npz files from files or folders.

    If a folder has a samples/ subfolder, this script will include files inside it.
    """
    files: List[Path] = []

    for item in inputs:
        p = Path(item)

        if p.is_file() and p.suffix.lower() == ".npz":
            files.append(p)
            continue

        if p.is_dir():
            if (p / "samples").exists():
                search_root = p / "samples"
            else:
                search_root = p

            if recursive:
                files.extend(sorted(search_root.rglob("*.npz")))
            else:
                files.extend(sorted(search_root.glob("*.npz")))

    # Deduplicate
    files = sorted(set(files), key=lambda x: str(x))

    return files


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize RoomToFP V9 .npz samples as a montage."
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more .npz files or folders containing .npz samples.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output montage PNG path.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum number of samples to visualize. Default: 30.",
    )

    parser.add_argument(
        "--panel-size",
        type=int,
        default=220,
        help="Panel size in pixels. Default: 220.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search folders for .npz files.",
    )

    parser.add_argument(
        "--stats-json",
        type=str,
        default="",
        help="Optional path to save per-sample quality stats as JSON.",
    )

    args = parser.parse_args()

    npz_files = collect_npz_files(args.input, recursive=args.recursive)

    if not npz_files:
        raise FileNotFoundError("No .npz files found from the provided input paths.")

    npz_files = npz_files[: max(1, int(args.max_samples))]

    rows = []
    all_stats = []

    print(f"Found {len(npz_files)} sample(s) to visualize.")

    for i, npz_path in enumerate(npz_files, start=1):
        print(f"[{i}/{len(npz_files)}] {npz_path}")

        try:
            row, stats = make_sample_row(npz_path, panel_size=int(args.panel_size))
            rows.append(row)
            all_stats.append(stats)
        except Exception as e:
            print(f"  WARNING: failed to visualize {npz_path}: {type(e).__name__}: {e}")

    if not rows:
        raise RuntimeError("No rows were successfully visualized.")

    montage = vstack_with_gap(rows, gap=10)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(montage).save(out_path)

    print(f"Saved montage to: {out_path}")

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
        print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()