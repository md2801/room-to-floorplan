"""
1. Converts depth images into 3D point clouds
2. Separates points into things like structure and furniture using semantic labels
3. Projects those points into a 2D top-down map
4. Prepares input tensors and target masks for a learning system
5. Saves outputs (preview images, SVG floorplan contours and point clouds)

This file performs geometry and preprocessing for our pipeline. It takes indoor scene data like depth maps and RGB
images, semantic labels and camera information and converts them into 3D world points so we can project them into a
top-down bird's eye view grid which we can then use as inputs for training visualization and floorplan extraction.
Essentially this file turns raw data into something a floorplan model can use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label,
)
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure

from .structured3d import FURNITURE_IDS, STRUCTURE_IDS


def perspective_depth_to_world(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    semantic: np.ndarray,
    pose: np.ndarray,
    intr: np.ndarray,
    stride: int = 4,
):
    h, w = depth_mm.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]

    d = depth_mm[0:h:stride, 0:w:stride].astype(np.float32) / 1000.0
    valid = d > 1e-6

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    d = d[valid]

    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])

    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    Z = d

    cam = np.stack([X, Y, Z], axis=1).astype(np.float32)
    world = (pose[:3, :3] @ cam.T).T + pose[:3, 3]

    colors = rgb[0:h:stride, 0:w:stride][valid]
    sem = semantic[0:h:stride, 0:w:stride][valid]

    return world.astype(np.float32), colors, sem


def panorama_depth_to_world(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    semantic: np.ndarray,
    camera_xyz_mm: np.ndarray,
    stride: int = 4,
):
    h, w = depth_mm.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]

    d = depth_mm[0:h:stride, 0:w:stride].astype(np.float32) / 1000.0
    valid = d > 1e-6

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    d = d[valid]

    xsn = (xs + 0.5) / float(w)
    ysn = (ys + 0.5) / float(h)

    theta = xsn * (2.0 * np.pi) - np.pi
    phi = (0.5 - ysn) * np.pi

    cos_phi = np.cos(phi)
    dir_x = np.sin(theta) * cos_phi
    dir_y = -np.cos(theta) * cos_phi
    dir_z = np.sin(phi)

    dirs = np.stack([dir_x, dir_y, dir_z], axis=1).astype(np.float32)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)

    camera_xyz_m = camera_xyz_mm.astype(np.float32) / 1000.0
    world = dirs * d[:, None] + camera_xyz_m[None, :]

    colors = rgb[0:h:stride, 0:w:stride][valid]
    sem = semantic[0:h:stride, 0:w:stride][valid]

    return world.astype(np.float32), colors, sem


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask

    labeled, num = label(mask)
    if num == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labeled]


def make_bev(
    points_xyz: np.ndarray,
    semantic: np.ndarray,
    colors: Optional[np.ndarray] = None,
    grid_size: int = 512,
    meters_per_pixel: Optional[float] = None,
    z_clip: Tuple[float, float] = (-0.5, 3.0),
    height_band_above_floor: Tuple[float, float] = (0.15, 1.6),
    xy_padding_m: float = 0.35,
    splat_radius: int = 2,
) -> Dict[str, np.ndarray]:
    if len(points_xyz) == 0:
        raise ValueError("No points given")

    pts = np.asarray(points_xyz, dtype=np.float32).copy()
    sem = np.asarray(semantic)

    keep = (pts[:, 2] >= z_clip[0]) & (pts[:, 2] <= z_clip[1])
    pts = pts[keep]
    sem = sem[keep]

    if colors is not None:
        colors = np.asarray(colors)
        colors = colors[keep]
        if colors.ndim == 2 and colors.shape[1] >= 4:
            colors = colors[:, :3]

    if len(pts) == 0:
        raise ValueError("No points remaining after z clipping")

    floor_z = float(np.percentile(pts[:, 2], 2.0))
    band_lo = floor_z + float(height_band_above_floor[0])
    band_hi = floor_z + float(height_band_above_floor[1])

    keep_band = (pts[:, 2] >= band_lo) & (pts[:, 2] <= band_hi)
    pts = pts[keep_band]
    sem = sem[keep_band]

    if colors is not None:
        colors = colors[keep_band]

    if len(pts) == 0:
        raise ValueError("No points remaining after floor-height band filtering")

    xy = pts[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center_xy = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-3)

    half_extent = 0.5 * float(max(span)) + float(xy_padding_m)

    if meters_per_pixel is None:
        meters_per_pixel = float((2.0 * half_extent) / max(grid_size - 1, 1))

    W = grid_size
    H = grid_size

    x0 = center_xy[0] - half_extent
    y0 = center_xy[1] - half_extent

    px = np.clip(((xy[:, 0] - x0) / meters_per_pixel).astype(np.int32), 0, W - 1)
    py = np.clip(((xy[:, 1] - y0) / meters_per_pixel).astype(np.int32), 0, H - 1)

    occ = np.zeros((H, W), np.float32)
    structure = np.zeros((H, W), np.float32)
    furniture = np.zeros((H, W), np.float32)

    if colors is not None:
        rgb = np.zeros((H, W, 3), np.float32)
        count = np.zeros((H, W, 1), np.float32)
    else:
        rgb = None
        count = None

    offsets: list[tuple[int, int]] = []
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            if dx * dx + dy * dy <= splat_radius * splat_radius:
                offsets.append((dy, dx))

    for i in range(len(px)):
        x = int(px[i])
        y = int(py[i])
        sid = int(sem[i])

        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                continue

            # Occupancy should reflect useful geometry, not just raw floor density.
            occ[yy, xx] += 0.30

            if sid in STRUCTURE_IDS:
                structure[yy, xx] += 4.0
                occ[yy, xx] += 0.80
            elif sid in FURNITURE_IDS:
                furniture[yy, xx] += 0.35
                occ[yy, xx] += 0.15

            if rgb is not None:
                rgb[yy, xx] += colors[i].astype(np.float32)
                count[yy, xx] += 1.0

    occ = np.log1p(occ)
    structure = np.log1p(structure)
    furniture = np.log1p(furniture)

    if rgb is not None:
        rgb = rgb / np.maximum(count, 1.0)
        rgb = np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32)

    occ = gaussian_filter(occ, sigma=1.4)
    structure = gaussian_filter(structure, sigma=1.5)
    furniture = gaussian_filter(furniture, sigma=1.15)

    occ = norm01(occ)
    structure = norm01(structure)
    furniture = norm01(furniture)

    structure_bin = structure > 0.07
    furniture_bin = furniture > 0.11

    structure_bin = binary_closing(structure_bin, iterations=3)
    structure_bin = binary_opening(structure_bin, iterations=1)
    structure_bin = binary_fill_holes(structure_bin)
    structure_bin = binary_dilation(structure_bin, iterations=1)
    structure_bin = remove_small_components(structure_bin, min_size=24)

    furniture_bin = binary_closing(furniture_bin, iterations=1)
    furniture_bin = binary_opening(furniture_bin, iterations=1)
    furniture_bin = remove_small_components(furniture_bin, min_size=20)

    structure = np.maximum(structure, structure_bin.astype(np.float32))
    furniture = np.maximum(furniture, furniture_bin.astype(np.float32))

    occ = np.maximum(occ, 0.80 * structure + 0.20 * furniture)
    occ = np.clip(occ, 0.0, 1.0)

    origin_xy = np.array([x0, y0], dtype=np.float32)

    return {
        "origin_xy": origin_xy,
        "center_xy": center_xy.astype(np.float32),
        "meters_per_pixel": np.array([meters_per_pixel], dtype=np.float32),
        "floor_z": np.array([floor_z], dtype=np.float32),
        "height_band_above_floor": np.array(
            [height_band_above_floor[0], height_band_above_floor[1]],
            dtype=np.float32,
        ),
        "occupancy": occ.astype(np.float32),
        "structure": norm01(structure).astype(np.float32),
        "furniture": norm01(furniture).astype(np.float32),
        "rgb": rgb,
    }


def stack_observation(bev: Dict[str, np.ndarray]) -> np.ndarray:
    occ = bev["occupancy"]
    structure = bev["structure"]
    furniture = bev["furniture"]
    rgb = bev["rgb"]

    if rgb is None:
        rgb = np.zeros((*occ.shape, 3), np.float32)

    return np.dstack([occ, structure, furniture, rgb]).astype(np.float32)


def target_from_bev(
    bev: Dict[str, np.ndarray],
    wall_thresh: float = 0.06,
    furn_thresh: float = 0.11,
) -> np.ndarray:
    structure = bev["structure"] > wall_thresh
    furniture = bev["furniture"] > furn_thresh

    structure = binary_closing(structure, iterations=3)
    structure = binary_opening(structure, iterations=1)
    structure = binary_fill_holes(structure)
    structure = binary_dilation(structure, iterations=1)
    structure = remove_small_components(structure, min_size=24)

    furniture = binary_closing(furniture, iterations=1)
    furniture = binary_opening(furniture, iterations=1)
    furniture = remove_small_components(furniture, min_size=20)

    return np.stack(
        [structure.astype(np.float32), furniture.astype(np.float32)],
        axis=0,
    )


def save_bev_preview(observation: np.ndarray, target: np.ndarray, out_png: str | Path):
    occ = observation[..., 0]
    structure = target[0]
    furniture = target[1]

    canvas = np.zeros((*occ.shape, 3), np.uint8)
    canvas[..., 0] = np.clip(furniture * 255, 0, 255).astype(np.uint8)
    canvas[..., 1] = np.clip(structure * 255, 0, 255).astype(np.uint8)
    canvas[..., 2] = np.clip(occ * 255, 0, 255).astype(np.uint8)

    Image.fromarray(canvas).save(out_png)


def masks_to_svg(structure_mask: np.ndarray, out_svg: str | Path, min_area: float = 50.0):
    contours = measure.find_contours(structure_mask.astype(np.float32), 0.5)
    H, W = structure_mask.shape
    paths: List[str] = []

    for contour in contours:
        pts = [(float(c[1]), float(c[0])) for c in contour]
        if len(pts) < 3:
            continue

        try:
            poly = Polygon(pts).buffer(0)
        except Exception:
            continue

        if poly.is_empty:
            continue

        geoms = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        for g in geoms:
            if g.area < min_area:
                continue
            coords = list(g.exterior.coords)
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in coords) + " Z"
            paths.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="2"/>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">' + "".join(paths) + "</svg>"
    )
    Path(out_svg).write_text(svg, encoding="utf-8")


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, out_ply: str | Path):
    points = np.asarray(points, np.float32)
    colors = np.asarray(colors, np.uint8)

    with open(out_ply, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")