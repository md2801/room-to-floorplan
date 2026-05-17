from __future__ import annotations

"""Scene-level Structured3D utilities for RoomToFP.

Inspired by RoomFormer-style scene fusion: use all available panorama RGB-D evidence
from a Structured3D scene, fuse it into a shared world coordinate frame, then project
it into a top-down BEV representation. This is custom project code and does not depend
on the old U-Net inference scripts.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

from room_to_floorplan.floorplan_svg import clean_mask
from room_to_floorplan.geometry import norm01, panorama_depth_to_world
from room_to_floorplan.structured3d import FURNITURE_IDS

# Structured3D semantic IDs used by this project.
# The old code grouped floor/ceiling/wall/door/window into one "structure" channel.
# For the RoomToFP vector branch we split them because the polygon model needs wall
# evidence separately from floor/free-space and openings.
WALL_IDS = {1}
FLOOR_IDS = {2}
CEILING_IDS = {22}
DOOR_IDS = {8}
WINDOW_IDS = {9}
OPENING_IDS = DOOR_IDS | WINDOW_IDS


@dataclass
class SceneGrid:
    origin_xy: np.ndarray  # real-world coords of starting point of grid
    center_xy: np.ndarray  # real-world centre of bev
    meters_per_pixel: np.ndarray  # real-world distance each BEV pixel represents
    floor_z: np.ndarray  # estimated floor height of scene
    height_band_above_floor: np.ndarray  # vertical height range for filtering points above floor


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
    # Structured3D semantic PNGs are normally palette-index PNGs. PIL preserves class IDs.
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

    pano_dirs: list[Path] = []
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
    """Collect points from every panorama and requested variant in one world coordinate frame."""
    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_sem: list[np.ndarray] = []
    sources: list[dict] = []

    pano_dirs = find_panorama_dirs(scene_dir)
    print(f"--Scene-- found panorama folders: {len(pano_dirs)}")

    for pano_dir in pano_dirs:
        camera_path = pano_dir / "camera_xyz.txt"
        camera_xyz_mm = np.loadtxt(camera_path, dtype=np.float32)
        room_id = pano_dir.parent.name

        available = variant_dirs_for_pano(pano_dir, variants)
        if not available:
            print(f"--Skipping Room-- room={room_id}: no requested variants found in {pano_dir}")
            continue

        for variant_name, variant_dir in available:
            depth_path = variant_dir / "depth.png"
            semantic_path = variant_dir / "semantic.png"
            rgb_path = choose_rgb_file(variant_dir, preferred=rgb_name)

            rgb = read_rgb(rgb_path)
            depth_mm = read_depth_mm(depth_path)
            semantic = read_semantic_ids(semantic_path)

            semantic_unique = np.unique(semantic)
            has_walls = any(int(v) in WALL_IDS for v in semantic_unique)
            has_floor = any(int(v) in FLOOR_IDS for v in semantic_unique)
            has_openings = any(int(v) in OPENING_IDS for v in semantic_unique)
            has_furniture = any(int(v) in FURNITURE_IDS for v in semantic_unique)

            print(
                f"--Loading Room-- room={room_id} variant={variant_name} "
                f"sem_ids={semantic_unique[:30].tolist()} "
                f"wall={has_walls} floor={has_floor} openings={has_openings} furniture={has_furniture}"
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
    print(f"--Scene-- total sampled points before filtering: {len(pts_cat):,}")

    return pts_cat, colors_cat, sem_cat, sources


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


def create_scene_grid(
    points_xyz: np.ndarray,  # 3D point cloud created from panorama depth
    grid_size: int,
    z_clip: tuple[float, float],  # vertical filter
    height_band_above_floor: tuple[float, float],  # height band above estimated floor
    xy_padding_m: float,  # adds extra margin around scenes, prevents walls near edges from being clipped
    meters_per_pixel: Optional[float] = None,
    annotation_xy_m: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SceneGrid]:
    """
    Defins top-down coord system for one scene
    3D scene point cloud -> Filter useful points -> estimate floor height -> calculate scene bounds -> create BEV grid metadata
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    # takes z coords of each point and creates boolean mask if point is within z_clip
    keep = (pts[:, 2] >= z_clip[0]) & (pts[:, 2] <= z_clip[1])
    pts_clip = pts[keep]  # keeps only valid points

    if len(pts_clip) == 0:
        raise ValueError("No points remain after z clipping.")

    # estimates floor height using 2nd percentile of z value
    # floor is usually near the lowest visible points, but taking minimum is risky, hence 2 percentile
    floor_z = float(np.percentile(pts_clip[:, 2], 2.0))

    # calculate lower and upper height limits relative to floor
    band_lo = floor_z + float(height_band_above_floor[0])
    band_hi = floor_z + float(height_band_above_floor[1])

    # filtering points to height band
    keep_band_clip = (pts_clip[:, 2] >= band_lo) & (pts_clip[:, 2] <= band_hi)
    band_pts = pts_clip[keep_band_clip]

    if len(band_pts) == 0:
        raise ValueError("No points remain after floor-height band filtering.")

    # Building xy scene bounds
    xy_sources = [band_pts[:, :2]]  # takes x, y coords

    # if annotation_3d.json coords available, add them if point cloud misses some parts, this can help
    if annotation_xy_m is not None and len(annotation_xy_m) > 0:
        xy_sources.append(annotation_xy_m.astype(np.float32))

    # combines all xy into 1 array
    all_xy = np.concatenate(xy_sources, axis=0)

    # calculate min and max and center
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    center_xy = 0.5 * (mins + maxs)

    # calculate scene width and height in metres
    span = np.maximum(maxs - mins, 1e-3)  # prevents span from being 0
    half_extent = 0.5 * float(max(span)) + float(xy_padding_m)  # creates square BEV area

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

    keep_global = keep.copy()
    idx_clip = np.where(keep)[0]
    keep_global[idx_clip[~keep_band_clip]] = False

    return keep_global, band_pts, all_xy, grid


def world_xy_to_pixels(xy_m: np.ndarray, grid: SceneGrid, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts xy coords in metres into BEV pixel coords
    """
    mpp = float(grid.meters_per_pixel[0])
    px = np.clip(
        (
                (
                    # takes x-coord of each point and subtracts the BEV origin, converting world pos into distance
                    xy_m[:, 0] - grid.origin_xy[0]
                ) / mpp
        ).astype(np.int32), 0, grid_size - 1)
    py = np.clip(((xy_m[:, 1] - grid.origin_xy[1]) / mpp).astype(np.int32), 0, grid_size - 1)
    return px, py


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    """
    Creates list of pixel offsets inside a circular disk
    Used for splatting -> makes map denser and smoother
    Instead of projecting 3D points into exactly 1 pixel, we paint it into a small circular area around that pixel
    Since pt clouds can be sparse, if each point only hits 1 pixel, the BEV map can look dotted and fragmented.
    """
    # dy -> vertical shift; dx -> horizontal shift
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:  # keeps points inside the circle (x^2 + y^2 = r^2)
                offsets.append((dy, dx))
    return offsets


def _clean_density_channel(channel: np.ndarray, threshold: float, close_kernel: int, open_kernel: int, min_area: int, fill_holes: bool) -> np.ndarray:
    """
    raw density -> log compression -> gauss blur -> 0-1 norm -> mask cleanup -> combines soft + cleaned result
    """
    channel = norm01(cv2.GaussianBlur(np.log1p(channel), (0, 0), 1.2))
    cleaned = clean_mask(channel, threshold=threshold, close_kernel=close_kernel, open_kernel=open_kernel, min_area_px=min_area, fill_holes=fill_holes)
    return np.maximum(channel, cleaned.astype(np.float32)).astype(np.float32)


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
    """
    Build a RoomToFP BEV dictionary with separated semantic channels.

    Output channels are intentionally split as:
      occupancy, wall, floor, opening, furniture, rgb
    because a single mixed "structure" channel made floor/ceiling evidence look like walls.

    3D panorama -> filter useful pts -> place them on 2D top-down grid -> separate by semantic type -> clean channels -> return BEV dict
    """
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
    wall = np.zeros((H, W), np.float32)
    floor = np.zeros((H, W), np.float32)
    opening = np.zeros((H, W), np.float32)
    furniture = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    count = np.zeros((H, W, 1), np.float32)

    offsets = _disk_offsets(max(1, int(splat_radius)))

    for i in range(len(px)):
        sid = int(sem[i])
        for dy, dx in offsets:  # splatting each pt over nearby pixels
            yy = int(py[i]) + dy
            xx = int(px[i]) + dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W:  # checks if pixel is inside image
                continue

            occ[yy, xx] += 0.20
            if sid in WALL_IDS:
                wall[yy, xx] += 4.0
                occ[yy, xx] += 0.85
            elif sid in FLOOR_IDS:
                floor[yy, xx] += 1.0
                occ[yy, xx] += 0.25
            elif sid in OPENING_IDS:
                opening[yy, xx] += 3.0
                occ[yy, xx] += 0.60
            elif sid in FURNITURE_IDS:
                furniture[yy, xx] += 0.75
                occ[yy, xx] += 0.25

            rgb[yy, xx] += colors[i].astype(np.float32)
            count[yy, xx] += 1.0

    rgb = rgb / np.maximum(count, 1.0)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32)

    occ = norm01(cv2.GaussianBlur(np.log1p(occ), (0, 0), 1.4))
    wall = _clean_density_channel(wall, threshold=0.07, close_kernel=5, open_kernel=3, min_area=24, fill_holes=False)
    floor = _clean_density_channel(floor, threshold=0.05, close_kernel=7, open_kernel=3, min_area=80, fill_holes=True)
    opening = _clean_density_channel(opening, threshold=0.08, close_kernel=3, open_kernel=1, min_area=10, fill_holes=False)
    furniture = _clean_density_channel(furniture, threshold=0.10, close_kernel=3, open_kernel=3, min_area=20, fill_holes=False)

    # Keep the old key for compatibility with any helper still expecting "structure".
    structure = np.maximum(wall, opening)
    occ = np.clip(np.maximum(occ, 0.85 * wall + 0.35 * opening + 0.20 * furniture), 0.0, 1.0)

    return {
        "origin_xy": grid.origin_xy,
        "center_xy": grid.center_xy,
        "meters_per_pixel": grid.meters_per_pixel,
        "floor_z": grid.floor_z,
        "height_band_above_floor": grid.height_band_above_floor,
        "occupancy": occ.astype(np.float32),
        "wall": norm01(wall).astype(np.float32),
        "floor": norm01(floor).astype(np.float32),
        "opening": norm01(opening).astype(np.float32),
        "furniture": norm01(furniture).astype(np.float32),
        "structure": norm01(structure).astype(np.float32),
        "rgb": rgb,
    }


def stack_roomtofp_observation(bev: dict) -> np.ndarray:
    """Return the 8-channel RoomToFP observation tensor: H x W x 7.

    Channel order:
      0 occupancy
      1 wall density
      2 floor/free-space density
      3 door/window opening density
      4 furniture density
      5 RGB red
      6 RGB green
      7 RGB blue

    Note: this returns 8 channels (5 semantic/density + 3 RGB). The model/training
    scripts infer the channel count from the saved samples, so old hard-coded 6-channel
    assumptions no longer apply.
    """
    occ = bev["occupancy"]
    wall = bev.get("wall", bev.get("structure"))
    floor = bev.get("floor", np.zeros_like(occ))
    opening = bev.get("opening", np.zeros_like(occ))
    furniture = bev["furniture"]
    rgb = bev.get("rgb")
    if rgb is None:
        rgb = np.zeros((*occ.shape, 3), np.float32)
    return np.dstack([occ, wall, floor, opening, furniture, rgb]).astype(np.float32)
