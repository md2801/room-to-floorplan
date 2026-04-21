"""
1. Converts depth images into 3D point clouds
2. Separates points into things like structure and furniture using semantic labels
3. Projects those points into a 2D top-down map
4 .Prepares input tensors and target masks for a learning system
5. Saves outputs (preview images, SVG floorplan contours and point clouds

This file performs geometry and preprocessing for our pipeline. It takes indoor scene data like depth maps and RGB
images, semantic labels and camera information and converts them into 3D world points so we can project them into a 
top-down bird's eye view grid which we can then use as inputs for traning visualization and floorplan extraction.
Essentially this file turns raw data into something a floorplan model can use. Raw RGB frames like those that are
included in the panorama and perspective zips arent useful for our use case directly, so we need to conduct some
preprocessing first by converting camera view -> 3D scene -> top-down map
"""

from __future__ import annotations

# import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # , Iterable

# import cv2
import numpy as np
from PIL import Image  # ImageDraw

# smoothing + morphological ops
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter  

# converting masks to polygons
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon

# contour extraction
from skimage import measure

from .structured3d import FURNITURE_IDS, STRUCTURE_IDS  # decode_normals


def perspective_depth_to_world(depth_mm: np.ndarray, rgb: np.ndarray, semantic: np.ndarray, pose: np.ndarray, intr: np.ndarray, stride: int = 4):
    """
    Samples pixels from depth map every "stride" pixels, converts valid depth pixels from image coordinates (x, y, depth) into 3D camera coordinates (X, Y,bZ), then transforms them into   world coordinates using pose matrix. It also keeps the matching RGB color and semantic label for each point.
    
    depth_mm: depth map in millimeters
    rgb: corresponding RGB image
    semantic: per-pixel semantic labels
    pose: camera pose matrix
    intr: camera intrinsics matrix
    stride: subsampling factor
    """
    h, w = depth_mm.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]  # getting image coordinates on a grid

    # downsampling the depth map and converting to metres
    d = depth_mm[0:h:stride, 0:w:stride].astype(np.float32) / 1000.0

    valid = d > 1e-6  # removing 0 or invalid depth values
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    d = d[valid]

    # using camera intrinsics to perform back-projection
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    Z = d

    cam = np.stack([X, Y, Z], axis=1)

    # Now we have 3D points in the camera frame that we rotate and translate into the world frame as such:
    world = (pose[:3, :3] @ cam.T).T + pose[:3, 3]
    colors = rgb[0:h:stride, 0:w:stride][valid]
    sem = semantic[0:h:stride, 0:w:stride][valid]

    return world, colors, sem


def panorama_depth_to_world(depth_mm: np.ndarray, rgb: np.ndarray, semantic: np.ndarray, camera_xyz_mm: np.ndarray, stride: int = 4):
    h, w = depth_mm.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]

    d = depth_mm[0:h:stride, 0:w:stride].astype(np.float32) / 1000.0
    valid = d > 1e-6

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    d = d[valid]

    # equirectangular mapping
    theta = (xs / w - 0.5) * 2.0 * np.pi  # left-right
    phi = (0.5 - ys / h) * np.pi  # up-down

    # Structured3D panorama faces -y by default; z is up.
    dir_x = np.sin(theta) * np.cos(phi)
    dir_y = -np.cos(theta) * np.cos(phi)
    dir_z = np.sin(phi)

    dirs = np.stack([dir_x, dir_y, dir_z], axis=1)
    world = dirs * d[:, None] + (camera_xyz_mm.astype(np.float32) / 1000.0)[None, :]
    colors = rgb[0:h:stride, 0:w:stride][valid]
    sem = semantic[0:h:stride, 0:w:stride][valid]

    return world, colors, sem


def make_bev(points_xyz: np.ndarray, semantic: np.ndarray, colors: Optional[np.ndarray] = None, grid_size: int = 512, meters_per_pixel: Optional[float] = None, z_clip: Tuple[float, float] = (-0.2, 2.5)) -> Dict[str, np.ndarray]:
    if len(points_xyz) == 0:
        raise ValueError("No points given")

    pts = points_xyz.copy()
    keep = (pts[:, 2] >= z_clip[0]) & (pts[:, 2] <= z_clip[1])
    pts = pts[keep]
    sem = semantic[keep]

    if colors is not None:
        colors = colors[keep]
        if colors.ndim == 2 and colors.shape[1] >= 4:
            colors = colors[:, :3]

    mins = pts[:, :2].min(axis=0)
    maxs = pts[:, :2].max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)

    if meters_per_pixel is None:
        meters_per_pixel = float(max(span) / (grid_size - 1))

    W = grid_size
    H = grid_size

    px = np.clip(((pts[:, 0] - mins[0]) / meters_per_pixel).astype(int), 0, W - 1)
    py = np.clip(((pts[:, 1] - mins[1]) / meters_per_pixel).astype(int), 0, H - 1)

    occ = np.zeros((H, W), np.float32)
    structure = np.zeros((H, W), np.float32)
    furniture = np.zeros((H, W), np.float32)
    if colors is not None:
        rgb = np.zeros((H, W, 3), np.float32)
        count = np.zeros((H, W, 1), np.float32)
    else:
        rgb = None

    for i in range(len(px)):
        x, y = px[i], py[i]
        occ[y, x] += 1.0
        sid = int(sem[i])
        if sid in STRUCTURE_IDS:
            structure[y, x] += 1.0
        elif sid in FURNITURE_IDS:
            furniture[y, x] += 1.0
        if rgb is not None:
            rgb[y, x] += colors[i].astype(np.float32)
            count[y, x] += 1.0

    occ = np.log1p(occ)
    structure = np.log1p(structure)
    furniture = np.log1p(furniture)
    if rgb is not None:
        rgb = rgb / np.maximum(count, 1.0)
        rgb = (rgb / 255.0).astype(np.float32)

    occ = norm01(gaussian_filter(occ, 1.0))
    structure = norm01(gaussian_filter(structure, 1.0))
    furniture = norm01(gaussian_filter(furniture, 1.0))

    return {
        "origin_xy": mins,
        "meters_per_pixel": np.array([meters_per_pixel], dtype=np.float32),
        "occupancy": occ,
        "structure": structure,
        "furniture": furniture,
        "rgb": rgb,
    }


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x)

    return (x - mn) / (mx - mn)


def stack_observation(bev: Dict[str, np.ndarray]) -> np.ndarray:
    occ = bev["occupancy"]
    structure = bev["structure"]
    furniture = bev["furniture"]
    rgb = bev["rgb"]
    if rgb is None:
        rgb = np.zeros((*occ.shape, 3), np.float32)

    return np.dstack([occ, structure, furniture, rgb]).astype(np.float32)


def target_from_bev(bev: Dict[str, np.ndarray], wall_thresh=0.10, furn_thresh=0.08) -> np.ndarray:
    structure = bev["structure"] > wall_thresh
    furniture = bev["furniture"] > furn_thresh
    structure = binary_closing(binary_opening(structure, iterations=1), iterations=2)
    furniture = binary_closing(binary_opening(furniture, iterations=1), iterations=2)

    return np.stack([structure.astype(np.float32), furniture.astype(np.float32)], axis=0)


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

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">' + "".join(paths) + "</svg>"
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