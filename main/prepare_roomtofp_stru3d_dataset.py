from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from render_structured3d_roomtofp_floorplan import build_roomtofp_scene_outputs
from room_to_floorplan_helpers.roomtofp_scene_utils import (
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

STRUCTURED3D_ROOM_TYPE_ALIASES = {
    "living room": "living room",
    "living_room": "living room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "balcony": "balcony",
    "corridor": "corridor",
    "hallway": "corridor",
    "dining room": "dining room",
    "dining_room": "dining room",
    "study": "study",
    "office": "office",
    "storage": "storage",
    "undefined": "undefined",
    "other": "other",
}

NON_ROOM_SEMANTIC_TYPES = {
    "outwall",
    "door",
    "window",
    "wall",
    "floor",
    "ceiling",
    "object",
    "furniture",
    "beam",
    "column",
    "stairs",
}

OBSERVATION_CHANNEL_ORDER = [
    "occupancy",
    "wall",
    "floor",
    "opening",
    "furniture",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "observed_corner_prior",
    "distance_to_observed_wall",
]


# ============================================================
# IO helpers
# ============================================================

def extract_zips(zip_paths: list[str], extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)

    for zp in zip_paths:
        zpath = Path(zp)
        print(f"--Extracting-- {zpath} -> {extract_root}")

        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_root)

    return extract_root


def find_scene_dirs(root: Path) -> list[Path]:
    """
    Finds Structured3D scene folders.

    Supports both:
        1. Passing a parent folder containing many scene_* folders.
        2. Passing a specific scene_* folder directly.
    """
    root = Path(root)

    scene_dirs = []

    if root.is_dir() and root.name.startswith("scene_") and (root / "annotation_3d.json").exists():
        scene_dirs.append(root)

    scene_dirs.extend(
        [
            p
            for p in root.rglob("scene_*")
            if p.is_dir() and (p / "annotation_3d.json").exists()
        ]
    )

    return sorted(set(scene_dirs))


def load_annotation(scene_dir: Path) -> Dict[str, Any]:
    annotation_path = scene_dir / "annotation_3d.json"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Numeric / geometry helpers
# ============================================================

def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.size == 0:
        return arr.astype(np.float32)

    mn = float(arr.min())
    mx = float(arr.max())

    if mx - mn <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return ((arr - mn) / (mx - mn)).astype(np.float32)


def polygon_area_pixels(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_area_xy(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def remove_duplicate_points(poly: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Remove consecutive duplicate points and repeated closing point.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) == 0:
        return poly.astype(np.float32)

    cleaned = [poly[0]]

    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) > eps:
            cleaned.append(p)

    cleaned = np.asarray(cleaned, dtype=np.float32)

    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
        cleaned = cleaned[:-1]

    return cleaned.astype(np.float32)


def _largest_polygon_from_geometry(geom):
    """
    Extract the largest Polygon from a Shapely geometry.

    Polygon repair can produce MultiPolygon or GeometryCollection.
    For room targets, the largest valid polygon is usually the real room footprint.
    """
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda g: float(g.area))

    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda g: float(g.area))

    return None


def _ensure_clockwise_image_coords(poly: np.ndarray) -> np.ndarray:
    """
    Stable clockwise-like winding in image coordinates.

    This is only for sequence consistency. It does not change polygon shape.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) < 3:
        return poly.astype(np.float32)

    x = poly[:, 0]
    y = poly[:, 1]
    signed = float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    if signed < 0:
        poly = poly[::-1].copy()

    return poly.astype(np.float32)


def _start_polygon_at_top_left(poly: np.ndarray) -> np.ndarray:
    """
    Rotate sequence so training targets start consistently.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) == 0:
        return poly.astype(np.float32)

    start = int(np.lexsort((poly[:, 0], poly[:, 1]))[0])
    return np.roll(poly, -start, axis=0).astype(np.float32)


def _remove_tiny_edges(poly: np.ndarray, min_edge: float) -> np.ndarray:
    """
    Remove tiny consecutive edges.

    Works in either metre or pixel units depending on input.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) <= 3:
        return poly.astype(np.float32)

    cleaned = [poly[0]]

    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= float(min_edge):
            cleaned.append(p)

    cleaned = np.asarray(cleaned, dtype=np.float32)

    if len(cleaned) > 3 and np.linalg.norm(cleaned[0] - cleaned[-1]) < float(min_edge):
        cleaned = cleaned[:-1]

    return cleaned.astype(np.float32)


def clean_room_polygon_xy_for_v9(
    xy: np.ndarray,
    simplify_eps_m: float = 0.025,
    min_area_m2: float = 0.20,
    min_edge_m: float = 0.03,
) -> np.ndarray:
    """
    Clean a room polygon in real-world XY metre coordinates.

    Important:
        This function preserves actual boundary order.
        It does NOT sort vertices by centroid angle.

    Why:
        Centroid-angle sorting destroys concave rooms and creates diagonal cuts.
    """
    pts = np.asarray(xy, dtype=np.float32)

    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    pts = remove_duplicate_points(pts, eps=1e-5)

    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    try:
        geom = Polygon(pts)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    if not geom.is_valid:
        geom = geom.buffer(0)

    geom = _largest_polygon_from_geometry(geom)

    if geom is None or geom.is_empty or float(geom.area) < float(min_area_m2):
        return np.zeros((0, 2), dtype=np.float32)

    if simplify_eps_m > 0:
        geom = geom.simplify(float(simplify_eps_m), preserve_topology=True)

        if not geom.is_valid:
            geom = geom.buffer(0)

        geom = _largest_polygon_from_geometry(geom)

        if geom is None or geom.is_empty or float(geom.area) < float(min_area_m2):
            return np.zeros((0, 2), dtype=np.float32)

    cleaned = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
    cleaned = remove_duplicate_points(cleaned, eps=1e-5)
    cleaned = _remove_tiny_edges(cleaned, min_edge=min_edge_m)

    if len(cleaned) < 3 or polygon_area_xy(cleaned) < float(min_area_m2):
        return np.zeros((0, 2), dtype=np.float32)

    # Stable sequence only. Shape is unchanged.
    cleaned = _ensure_clockwise_image_coords(cleaned)
    cleaned = _start_polygon_at_top_left(cleaned)

    return cleaned.astype(np.float32)


def clean_room_polygon_pixels_for_v9(
    px: np.ndarray,
    simplify_eps_px: float = 1.0,
    min_area_px: float = 16.0,
    min_edge_px: float = 2.0,
) -> np.ndarray:
    """
    Clean a room polygon in pixel coordinates.

    This is used after converting real-world XY to BEV pixels.
    """
    pts = np.asarray(px, dtype=np.float32)

    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    pts = remove_duplicate_points(pts, eps=1e-4)

    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    try:
        geom = Polygon(pts)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    if not geom.is_valid:
        geom = geom.buffer(0)

    geom = _largest_polygon_from_geometry(geom)

    if geom is None or geom.is_empty or float(geom.area) < float(min_area_px):
        return np.zeros((0, 2), dtype=np.float32)

    if simplify_eps_px > 0:
        geom = geom.simplify(float(simplify_eps_px), preserve_topology=True)

        if not geom.is_valid:
            geom = geom.buffer(0)

        geom = _largest_polygon_from_geometry(geom)

        if geom is None or geom.is_empty or float(geom.area) < float(min_area_px):
            return np.zeros((0, 2), dtype=np.float32)

    cleaned = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
    cleaned = remove_duplicate_points(cleaned, eps=1e-4)
    cleaned = _remove_tiny_edges(cleaned, min_edge=min_edge_px)

    if len(cleaned) < 3 or polygon_area_pixels(cleaned) < float(min_area_px):
        return np.zeros((0, 2), dtype=np.float32)

    cleaned = _ensure_clockwise_image_coords(cleaned)
    cleaned = _start_polygon_at_top_left(cleaned)

    return cleaned.astype(np.float32)


def normalize_points(points: np.ndarray, grid_size: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()

    pts[..., 0] /= max(1, grid_size - 1)
    pts[..., 1] /= max(1, grid_size - 1)

    return np.clip(pts, 0.0, 1.0).astype(np.float32)


def xy_to_pixels_v9(xy_m: np.ndarray, bev: Dict[str, np.ndarray], grid_size: int) -> np.ndarray:
    """
    Convert real-world XY metre coordinates to BEV pixel coordinates.

    This avoids relying on the old xy_to_poly_pixels helper.
    """
    xy_m = np.asarray(xy_m, dtype=np.float32)

    origin_xy = np.asarray(bev["origin_xy"], dtype=np.float32)
    meters_per_pixel = float(np.asarray(bev["meters_per_pixel"]).reshape(-1)[0])

    px = (xy_m[:, 0] - origin_xy[0]) / meters_per_pixel
    py = (xy_m[:, 1] - origin_xy[1]) / meters_per_pixel

    pts = np.stack([px, py], axis=1).astype(np.float32)

    pts[:, 0] = np.clip(pts[:, 0], 0, grid_size - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, grid_size - 1)

    return pts.astype(np.float32)


def resample_polygon_boundary(poly: np.ndarray, num_points: int) -> np.ndarray:
    """
    Uniformly sample points along a closed polygon boundary.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) < 2:
        return np.zeros((num_points, 2), dtype=np.float32)

    closed = np.concatenate([poly, poly[:1]], axis=0)
    segments = closed[1:] - closed[:-1]
    lengths = np.linalg.norm(segments, axis=1)

    perimeter = float(lengths.sum())

    if perimeter <= 1e-6:
        return np.repeat(poly[:1], num_points, axis=0).astype(np.float32)

    sample_dists = np.linspace(0.0, perimeter, num_points, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    sampled = []
    seg_idx = 0

    for d in sample_dists:
        while seg_idx < len(lengths) - 1 and cumulative[seg_idx + 1] < d:
            seg_idx += 1

        seg_start = closed[seg_idx]
        seg_vec = segments[seg_idx]
        seg_len = lengths[seg_idx]

        if seg_len <= 1e-6:
            sampled.append(seg_start)
        else:
            t = (d - cumulative[seg_idx]) / seg_len
            sampled.append(seg_start + t * seg_vec)

    return np.asarray(sampled, dtype=np.float32)


def make_distance_transform(edge_or_wall_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(edge_or_wall_mask)

    if mask.ndim == 3:
        mask = mask[0]

    mask = (mask > 0).astype(np.uint8)

    inverse = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inverse, cv2.DIST_L2, 5).astype(np.float32)

    if float(dist.max()) > 1e-6:
        dist = dist / float(dist.max())

    return dist.astype(np.float32)


def make_gaussian_corner_heatmap_from_points(
    points: np.ndarray,
    grid_size: int,
    sigma: float = 2.5,
) -> np.ndarray:
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    if len(points) == 0:
        return heatmap

    radius = int(max(3, sigma * 3))

    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))

        if cx < 0 or cx >= grid_size or cy < 0 or cy >= grid_size:
            continue

        x0 = max(0, cx - radius)
        x1 = min(grid_size, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(grid_size, cy + radius + 1)

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


def make_corner_prior_from_observed_wall(
    observed_wall: np.ndarray,
    sigma: float = 2.5,
    max_corners: int = 512,
) -> np.ndarray:
    """
    Build an input corner prior from observed wall evidence only.

    This is not GT.
    """
    wall = normalize01(observed_wall)
    h, w = wall.shape

    if float(wall.max()) <= 1e-8:
        return np.zeros((h, w), dtype=np.float32)

    wall_u8 = np.clip(wall * 255.0, 0, 255).astype(np.uint8)

    pts = cv2.goodFeaturesToTrack(
        wall_u8,
        maxCorners=int(max_corners),
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )

    heatmap = np.zeros((h, w), dtype=np.float32)

    if pts is None:
        return heatmap

    radius = int(max(3, sigma * 3))

    for p in pts.reshape(-1, 2):
        cx = int(round(float(p[0])))
        cy = int(round(float(p[1])))

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue

        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


def ensure_hwc_observation(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim != 3:
        raise ValueError(f"Expected observation to be 3D, got shape {obs.shape}")

    if obs.shape[0] in (8, 10) and obs.shape[-1] not in (8, 10):
        obs = np.transpose(obs, (1, 2, 0))

    return obs.astype(np.float32)


def build_v9_observation(
    obs8: np.ndarray,
    observed_corner_prior: np.ndarray,
    distance_to_observed_wall: np.ndarray,
) -> np.ndarray:
    """
    Extend 8-channel observed BEV into 10-channel V9 observed BEV.

    No GT channels are allowed here.
    """
    obs8 = ensure_hwc_observation(obs8)

    if obs8.shape[-1] < 8:
        raise ValueError(f"Expected at least 8 observation channels, got {obs8.shape}")

    if obs8.shape[-1] > 8:
        obs8 = obs8[..., :8]

    ch = np.asarray(observed_corner_prior, dtype=np.float32)
    dt = np.asarray(distance_to_observed_wall, dtype=np.float32)

    if ch.ndim == 3:
        ch = ch[0]

    if dt.ndim == 3:
        dt = dt[0]

    h, w = obs8.shape[:2]

    if ch.shape != (h, w):
        ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if dt.shape != (h, w):
        dt = cv2.resize(dt, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    obs10 = np.concatenate(
        [
            obs8.astype(np.float32),
            np.clip(ch, 0.0, 1.0)[..., None].astype(np.float32),
            np.clip(dt, 0.0, 1.0)[..., None].astype(np.float32),
        ],
        axis=-1,
    )

    return obs10.astype(np.float32)


# ============================================================
# Structured3D annotation graph parsing
# ============================================================

def _get_matrix(annotation: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if key not in annotation:
        return None

    arr = np.asarray(annotation[key])

    if arr.size == 0:
        return None

    return arr


def _line_to_junction_ids(line_junction_matrix: np.ndarray, line_id: int) -> list[int]:
    row = np.asarray(line_junction_matrix[line_id])

    ids = np.where(row > 0)[0].astype(int).tolist()

    if len(ids) >= 2:
        return ids[:2]

    return []


def _plane_to_line_ids(plane_line_matrix: np.ndarray, plane_id: int) -> list[int]:
    row = np.asarray(plane_line_matrix[plane_id])

    return np.where(row > 0)[0].astype(int).tolist()


def _semantic_plane_ids(semantic: Dict[str, Any]) -> list[int]:
    for key in ["planeID", "planeIds", "plane_ids", "plane_id"]:
        if key in semantic:
            value = semantic[key]

            if isinstance(value, list):
                return [int(v) for v in value]

            return [int(value)]

    return []


def _semantic_type(semantic: Dict[str, Any]) -> str:
    raw = str(semantic.get("type", semantic.get("name", "undefined"))).strip().lower()
    raw = raw.replace("-", " ").replace("_", " ")

    return STRUCTURED3D_ROOM_TYPE_ALIASES.get(raw, raw)


def _is_room_semantic(room_type: str, keep_balcony: bool) -> bool:
    if room_type in NON_ROOM_SEMANTIC_TYPES:
        return False

    if room_type == "balcony" and not keep_balcony:
        return False

    return True


def _is_horizontal_plane(plane: Dict[str, Any]) -> bool:
    """
    Heuristic for floor/ceiling planes.

    Structured3D stores plane normals. A floor/ceiling plane should have a dominant z normal.
    If normal is missing, we return True as a fallback so polygonization can still try.
    """
    normal = plane.get("normal", None)

    if normal is None:
        return True

    normal = np.asarray(normal, dtype=np.float32).reshape(-1)

    if len(normal) < 3:
        return True

    norm = float(np.linalg.norm(normal))

    if norm <= 1e-8:
        return True

    normal = normal / norm

    return abs(float(normal[2])) >= 0.80


def _junction_xy_m(annotation: Dict[str, Any]) -> np.ndarray:
    junctions = annotation.get("junctions", [])

    pts = []

    for j in junctions:
        c = j.get("coordinate", None)

        if c is None or len(c) < 2:
            pts.append([0.0, 0.0])
            continue

        pts.append([float(c[0]) / 1000.0, float(c[1]) / 1000.0])

    return np.asarray(pts, dtype=np.float32)


def _polygonize_plane(
    plane_id: int,
    annotation: Dict[str, Any],
    junction_xy: np.ndarray,
    line_junction_matrix: np.ndarray,
    plane_line_matrix: np.ndarray,
    min_area_m2: float,
) -> list[Polygon]:
    """
    Build polygons from a plane's boundary line graph.

    This is the important V9 fix:
        use line topology, not unordered points.
    """
    line_ids = _plane_to_line_ids(plane_line_matrix, plane_id)

    lines = []

    for lid in line_ids:
        if lid < 0 or lid >= len(line_junction_matrix):
            continue

        jids = _line_to_junction_ids(line_junction_matrix, lid)

        if len(jids) != 2:
            continue

        j0, j1 = jids

        if j0 >= len(junction_xy) or j1 >= len(junction_xy):
            continue

        p0 = tuple(junction_xy[j0].tolist())
        p1 = tuple(junction_xy[j1].tolist())

        if np.linalg.norm(junction_xy[j0] - junction_xy[j1]) <= 1e-5:
            continue

        lines.append(LineString([p0, p1]))

    if len(lines) < 3:
        return []

    try:
        merged = unary_union(lines)
        polys = list(polygonize(merged))
    except Exception:
        return []

    out = []

    for poly in polys:
        if poly.is_empty:
            continue

        if float(poly.area) < float(min_area_m2):
            continue

        if not poly.is_valid:
            poly = poly.buffer(0)

        poly = _largest_polygon_from_geometry(poly)

        if poly is None or poly.is_empty:
            continue

        if float(poly.area) < float(min_area_m2):
            continue

        out.append(poly)

    return out


def parse_room_polygons_v9(
    annotation: Dict[str, Any],
    min_area_m2: float = 0.20,
    keep_balcony: bool = True,
) -> list[dict]:
    """
    Parse clean room footprints from Structured3D annotation graph.

    This replaces the old parse_room_polygons path for V9 targets.

    Method:
        semantics -> planeID -> planeLineMatrix -> lineJunctionMatrix -> junctions
        then polygonize boundary lines.

    Why this is better:
        It preserves the actual room boundary topology instead of sorting unordered points.
    """
    junction_xy = _junction_xy_m(annotation)
    line_junction_matrix = _get_matrix(annotation, "lineJunctionMatrix")
    plane_line_matrix = _get_matrix(annotation, "planeLineMatrix")

    planes = annotation.get("planes", [])
    semantics = annotation.get("semantics", [])

    if line_junction_matrix is None or plane_line_matrix is None:
        raise ValueError("annotation_3d.json is missing lineJunctionMatrix or planeLineMatrix.")

    rooms: list[dict] = []

    for sem in semantics:
        room_type = _semantic_type(sem)

        if not _is_room_semantic(room_type, keep_balcony=keep_balcony):
            continue

        plane_ids = _semantic_plane_ids(sem)

        if not plane_ids:
            continue

        candidate_plane_ids = []

        for pid in plane_ids:
            if pid < 0 or pid >= len(planes):
                continue

            plane = planes[pid]

            if _is_horizontal_plane(plane):
                candidate_plane_ids.append(pid)

        # Fallback: if normals/types are unreliable, try all semantic planes.
        if not candidate_plane_ids:
            candidate_plane_ids = [pid for pid in plane_ids if 0 <= pid < len(planes)]

        plane_polys = []

        for pid in candidate_plane_ids:
            plane_polys.extend(
                _polygonize_plane(
                    plane_id=pid,
                    annotation=annotation,
                    junction_xy=junction_xy,
                    line_junction_matrix=line_junction_matrix,
                    plane_line_matrix=plane_line_matrix,
                    min_area_m2=min_area_m2,
                )
            )

        if not plane_polys:
            continue

        try:
            union_geom = unary_union(plane_polys)
        except Exception:
            union_geom = max(plane_polys, key=lambda g: float(g.area))

        poly = _largest_polygon_from_geometry(union_geom)

        if poly is None or poly.is_empty or float(poly.area) < float(min_area_m2):
            continue

        xy = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)

        xy = clean_room_polygon_xy_for_v9(
            xy,
            simplify_eps_m=0.025,
            min_area_m2=min_area_m2,
            min_edge_m=0.03,
        )

        if len(xy) < 3:
            continue

        area_m2 = polygon_area_xy(xy)

        if area_m2 < float(min_area_m2):
            continue

        if room_type not in ROOM_TYPE_TO_ID:
            room_type = "other"

        rooms.append(
            {
                "type": room_type,
                "xy": xy.astype(np.float32),
                "area_m2": float(area_m2),
            }
        )

    rooms = sorted(rooms, key=lambda r: float(r["area_m2"]), reverse=True)

    return rooms


# ============================================================
# Target building
# ============================================================

def pad_polygons(
    rooms: list[dict],
    bev: Dict[str, np.ndarray],
    grid_size: int,
    max_rooms: int,
    max_corners: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sparse auxiliary corner targets.

    Main V9 geometry target is boundary_64 / boundary_128.
    """
    corners = np.zeros((max_rooms, max_corners, 2), dtype=np.float32)
    corner_valid = np.zeros((max_rooms, max_corners), dtype=np.float32)

    room_valid = np.zeros((max_rooms,), dtype=np.float32)
    room_class = np.zeros((max_rooms,), dtype=np.int64)
    room_area = np.zeros((max_rooms,), dtype=np.float32)

    rooms_sorted = sorted(rooms, key=lambda r: float(r["area_m2"]), reverse=True)

    out_ri = 0

    for room in rooms_sorted:
        if out_ri >= max_rooms:
            break

        pts_px = xy_to_pixels_v9(room["xy"], bev, grid_size)

        pts_px = clean_room_polygon_pixels_for_v9(
            pts_px,
            simplify_eps_px=1.0,
            min_area_px=16.0,
            min_edge_px=2.0,
        )

        if len(pts_px) < 3:
            continue

        if len(pts_px) > max_corners:
            pts_sparse = resample_polygon_boundary(pts_px, num_points=max_corners)
        else:
            pts_sparse = pts_px

        pts_norm = normalize_points(pts_sparse, grid_size)

        n = min(len(pts_norm), max_corners)

        corners[out_ri, :n] = pts_norm[:n]
        corner_valid[out_ri, :n] = 1.0

        room_valid[out_ri] = 1.0
        room_class[out_ri] = ROOM_TYPE_TO_ID.get(room["type"], ROOM_TYPE_TO_ID["other"])
        room_area[out_ri] = float(room["area_m2"])

        out_ri += 1

    return corners, corner_valid, room_valid, room_class, room_area


def rasterize_room_instance_targets(
    rooms: list[dict],
    bev: Dict[str, np.ndarray],
    grid_size: int,
    max_rooms: int,
    corner_sigma: float = 2.5,
    boundary_thickness_px: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterise clean room polygons into masks and dense boundary targets.

    Returns:
        room_masks: [max_rooms, H, W]
        boundary_mask: [1, H, W]
        corner_heatmap: [1, H, W]
        boundary_64: [max_rooms, 64, 2]
        boundary_128: [max_rooms, 128, 2]
        valid_mask: [max_rooms]
    """
    room_masks = np.zeros((max_rooms, grid_size, grid_size), dtype=np.float32)
    boundary_mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    boundary_64 = np.zeros((max_rooms, 64, 2), dtype=np.float32)
    boundary_128 = np.zeros((max_rooms, 128, 2), dtype=np.float32)
    target_valid = np.zeros((max_rooms,), dtype=np.float32)

    all_corner_points = []

    rooms_sorted = sorted(rooms, key=lambda r: float(r["area_m2"]), reverse=True)

    out_ri = 0

    for room in rooms_sorted:
        if out_ri >= max_rooms:
            break

        pts_px = xy_to_pixels_v9(room["xy"], bev, grid_size)

        pts_px = clean_room_polygon_pixels_for_v9(
            pts_px,
            simplify_eps_px=1.0,
            min_area_px=16.0,
            min_edge_px=2.0,
        )

        if len(pts_px) < 3:
            continue

        pts_clipped = pts_px.copy()
        pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, grid_size - 1)
        pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, grid_size - 1)

        pts_int = np.round(pts_clipped).astype(np.int32)

        cv2.fillPoly(room_masks[out_ri], [pts_int], 1.0)

        cv2.polylines(
            boundary_mask,
            [pts_int],
            isClosed=True,
            color=1.0,
            thickness=int(boundary_thickness_px),
            lineType=cv2.LINE_AA,
        )

        all_corner_points.extend(pts_clipped.tolist())

        b64 = resample_polygon_boundary(pts_clipped, num_points=64)
        b128 = resample_polygon_boundary(pts_clipped, num_points=128)

        boundary_64[out_ri] = normalize_points(b64, grid_size)
        boundary_128[out_ri] = normalize_points(b128, grid_size)
        target_valid[out_ri] = 1.0

        out_ri += 1

    boundary_mask = np.clip(boundary_mask, 0.0, 1.0).astype(np.float32)

    if all_corner_points:
        corner_heatmap = make_gaussian_corner_heatmap_from_points(
            np.asarray(all_corner_points, dtype=np.float32),
            grid_size=grid_size,
            sigma=corner_sigma,
        )
    else:
        corner_heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    return (
        room_masks.astype(np.float32),
        boundary_mask[None, ...].astype(np.float32),
        corner_heatmap[None, ...].astype(np.float32),
        boundary_64.astype(np.float32),
        boundary_128.astype(np.float32),
        target_valid.astype(np.float32),
    )


def load_preview_outputs(
    preview_out: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Load auxiliary wall/opening/furniture masks from renderer output.
    """
    if (preview_out / "wall_mask.npy").exists():
        wall_mask = np.load(preview_out / "wall_mask.npy").astype(np.float32)
    elif (preview_out / "walls_mask.png").exists():
        wall_mask = (
            np.array(Image.open(preview_out / "walls_mask.png").convert("L")) > 0
        ).astype(np.float32)
    else:
        raise FileNotFoundError(f"Missing wall mask in preview folder: {preview_out}")

    if (preview_out / "opening_mask.npy").exists():
        opening_mask = np.load(preview_out / "opening_mask.npy").astype(np.float32)
    elif (preview_out / "opening_mask.png").exists():
        opening_mask = (
            np.array(Image.open(preview_out / "opening_mask.png").convert("L")) > 0
        ).astype(np.float32)
    else:
        opening_mask = np.zeros_like(wall_mask, dtype=np.float32)

    if (preview_out / "furniture_mask.npy").exists():
        furniture_mask = np.load(preview_out / "furniture_mask.npy").astype(np.float32)
    elif (preview_out / "furniture_mask.png").exists():
        furniture_mask = (
            np.array(Image.open(preview_out / "furniture_mask.png").convert("L")) > 0
        ).astype(np.float32)
    else:
        furniture_mask = np.zeros_like(wall_mask, dtype=np.float32)

    metadata = (
        json.loads((preview_out / "metadata.json").read_text(encoding="utf-8"))
        if (preview_out / "metadata.json").exists()
        else {}
    )

    components = metadata.get("furniture_components", [])

    return (
        furniture_mask[None, ...].astype(np.float32),
        wall_mask[None, ...].astype(np.float32),
        opening_mask[None, ...].astype(np.float32),
        components,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare clean RoomToFP V9 polygon training data from Structured3D scenes."
    )

    ap.add_argument(
        "--structured3d-root",
        nargs="+",
        default=None,
        help="One or more unzipped Structured3D roots or parent folders containing scene_* folders.",
    )
    ap.add_argument(
        "--panorama-zips",
        nargs="*",
        default=[],
        help="Optional Structured3D panorama subset zips to extract first.",
    )
    ap.add_argument("--extract-dir", default="data/structured3d_panorama_extracted")
    ap.add_argument("--output", default="data/roomtofp_polygon_v9_clean")

    ap.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes.")
    ap.add_argument("--max-rooms", type=int, default=20)
    ap.add_argument(
        "--max-corners",
        type=int,
        default=16,
        help="Sparse auxiliary corner target only. V9 primarily uses boundary_64/boundary_128.",
    )

    ap.add_argument("--variants", default="full")
    ap.add_argument("--rgb-name", default="rgb_rawlight.png")

    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--z-min", type=float, default=-0.5)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--height-lo", type=float, default=0.15)
    ap.add_argument("--height-hi", type=float, default=1.60)
    ap.add_argument("--xy-padding-m", type=float, default=0.75)
    ap.add_argument("--splat-radius", type=int, default=2)
    ap.add_argument("--wall-thickness", type=int, default=5)

    ap.add_argument("--corner-sigma", type=float, default=2.5)
    ap.add_argument("--boundary-thickness-px", type=int, default=2)

    ap.add_argument("--min-room-area-m2", type=float, default=0.20)
    ap.add_argument("--exclude-balcony", action="store_true")
    ap.add_argument("--keep-undefined-label", action="store_true")

    ap.add_argument("--min-opening-length-m", type=float, default=0.35)
    ap.add_argument("--max-opening-wall-distance-m", type=float, default=0.20)

    ap.add_argument("--min-furniture-area-px", type=int, default=120)
    ap.add_argument("--min-furniture-width-px", type=int, default=8)
    ap.add_argument("--min-furniture-height-px", type=int, default=8)
    ap.add_argument("--max-furniture-aspect-ratio", type=float, default=12.0)
    ap.add_argument("--furniture-room-erosion-px", type=int, default=1)
    ap.add_argument(
        "--include-furniture",
        action="store_true",
        help="Generate furniture masks/components. Off by default for floorplan-first V9.",
    )
    ap.add_argument(
        "--furniture-source",
        choices=["auto", "annotation", "semantic", "both"],
        default="auto",
    )

    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-previews", action="store_true")

    args = ap.parse_args()

    if not args.include_furniture:
        args.furniture_source = "off"

    here = Path(__file__).resolve().parent

    if args.panorama_zips:
        roots = [extract_zips(args.panorama_zips, (here / args.extract_dir).resolve())]
    elif args.structured3d_root:
        roots = [Path(p).resolve() for p in args.structured3d_root]
    else:
        raise ValueError("Provide either --structured3d-root or --panorama-zips.")

    out_arg = Path(args.output)
    out = out_arg.resolve() if out_arg.is_absolute() else (here / out_arg).resolve()

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

    for idx, scene_dir in enumerate(
        tqdm(scene_dirs, desc="prepare clean RoomToFP V9 data")
    ):
        sample_name = f"{scene_dir.name}.npz"
        sample_path = samples_dir / sample_name

        if args.skip_existing and sample_path.exists():
            records.append(
                {
                    "scene": scene_dir.name,
                    "sample": str(sample_path),
                    "skipped_existing": True,
                }
            )
            continue

        try:
            annotation = load_annotation(scene_dir)

            rooms = parse_room_polygons_v9(
                annotation,
                min_area_m2=args.min_room_area_m2,
                keep_balcony=not args.exclude_balcony,
            )

            if not args.keep_undefined_label:
                for room in rooms:
                    if room["type"] == "undefined":
                        room["type"] = "other"

            if not rooms:
                print(f"--Warning-- No clean room polygons parsed for {scene_dir.name}")
                continue

            pts, colors, sem, _sources = collect_scene_points(
                scene_dir,
                variants=variants,
                rgb_name=args.rgb_name,
                stride=args.stride,
            )

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

            obs8 = stack_roomtofp_observation(bev).astype(np.float32)
            obs8 = ensure_hwc_observation(obs8)

            if not args.include_furniture and obs8.shape[-1] >= 5:
                obs8[..., 4] = 0.0

            preview_out = previews_dir / scene_dir.name

            metadata = build_roomtofp_scene_outputs(
                scene_dir,
                preview_out,
                variants,
                args,
            )

            furniture_mask, wall_mask, opening_mask, furniture_components = load_preview_outputs(preview_out)

            corners, corner_valid, room_valid, room_class, room_area = pad_polygons(
                rooms=rooms,
                bev=bev,
                grid_size=args.grid_size,
                max_rooms=args.max_rooms,
                max_corners=args.max_corners,
            )

            (
                room_masks,
                boundary_mask,
                gt_corner_heatmap,
                boundary_64,
                boundary_128,
                target_valid,
            ) = rasterize_room_instance_targets(
                rooms=rooms,
                bev=bev,
                grid_size=args.grid_size,
                max_rooms=args.max_rooms,
                corner_sigma=args.corner_sigma,
                boundary_thickness_px=args.boundary_thickness_px,
            )

            # Use one validity source for all room targets.
            # If a room is invalid after rasterization, it must not be trained.
            room_valid = np.minimum(room_valid, target_valid).astype(np.float32)

            valid_count = int(room_valid.sum())

            if valid_count == 0:
                print(f"--Warning-- No valid rasterized room targets for {scene_dir.name}")
                continue

            observed_wall = obs8[..., 1].astype(np.float32)

            input_corner_prior = make_corner_prior_from_observed_wall(
                observed_wall,
                sigma=args.corner_sigma,
            )
            input_distance_to_wall = make_distance_transform(observed_wall)

            distance_to_wall_target = make_distance_transform(wall_mask)

            obs10 = build_v9_observation(
                obs8=obs8,
                observed_corner_prior=input_corner_prior,
                distance_to_observed_wall=input_distance_to_wall,
            )

            bev_chw = np.transpose(obs10, (2, 0, 1)).astype(np.float32)

            room_mask_merged = np.clip(np.sum(room_masks, axis=0), 0.0, 1.0).astype(np.float32)

            has_opening_supervision = np.asarray(1, dtype=np.int64)

            np.savez_compressed(
                sample_path,

                # Standard V9 input.
                bev=bev_chw.astype(np.float32),
                observation=obs10.astype(np.float32),

                # Sparse auxiliary targets.
                polygon_xy=corners.astype(np.float32),
                corner_valid=corner_valid.astype(np.float32),
                room_valid=room_valid.astype(np.float32),
                room_class=room_class.astype(np.int64),
                room_area=room_area.astype(np.float32),

                # Dense V9 polygon targets.
                boundary_64=boundary_64.astype(np.float32),
                boundary_128=boundary_128.astype(np.float32),

                # Dense masks.
                furniture_mask=furniture_mask.astype(np.float32),
                wall_mask=wall_mask.astype(np.float32),
                opening_mask=opening_mask.astype(np.float32),
                room_masks=room_masks.astype(np.float32),
                room_mask=room_mask_merged.astype(np.float32),
                room_mask_merged=room_mask_merged.astype(np.float32),
                boundary_mask=boundary_mask.astype(np.float32),
                edge_mask=boundary_mask.astype(np.float32),

                # GT geometry targets.
                corner_heatmap=gt_corner_heatmap.astype(np.float32),
                distance_to_wall_target=distance_to_wall_target[None, ...].astype(np.float32),

                # Input-derived priors.
                input_corner_prior=input_corner_prior[None, ...].astype(np.float32),
                distance_to_wall=input_distance_to_wall[None, ...].astype(np.float32),

                # Metadata.
                has_opening_supervision=has_opening_supervision,
                source=np.asarray("structured3d"),
                num_rooms=np.asarray(valid_count, dtype=np.int64),
                grid_size=np.asarray(int(args.grid_size), dtype=np.int64),
            )

            records.append(
                {
                    "scene": scene_dir.name,
                    "sample": str(sample_path),
                    "sample_name": sample_name,
                    "num_rooms": valid_count,
                    "num_rooms_raw_parsed": int(len(rooms)),
                    "num_openings": int(metadata.get("num_openings", 0)),
                    "num_furniture_components": int(
                        metadata.get(
                            "num_furniture_components",
                            len(furniture_components),
                        )
                    ),
                    "include_furniture": bool(args.include_furniture),
                    "furniture_source": metadata.get("furniture_source", args.furniture_source),
                    "observation_channels": int(obs10.shape[-1]),
                    "bev_channels": int(bev_chw.shape[0]),
                    "has_opening_supervision": True,
                    "source": "structured3d",
                    "parser": "v9_annotation_graph_polygonize",
                }
            )

            if not args.keep_previews and idx >= 10 and preview_out.exists():
                shutil.rmtree(preview_out, ignore_errors=True)

        except Exception as e:
            print(f"--Warning-- Skipped {scene_dir}: {type(e).__name__}: {e}")

    usable_records = [
        r
        for r in records
        if r.get("sample") and Path(r.get("sample", "")).exists()
    ]

    sample_names = [Path(r["sample"]).name for r in usable_records]

    rng = random.Random(int(args.split_seed))
    rng.shuffle(sample_names)

    total_ratio = max(
        1e-8,
        float(args.train_ratio + args.val_ratio + args.test_ratio),
    )

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
        (out / f"{split_name}.txt").write_text(
            "\n".join(names) + ("\n" if names else ""),
            encoding="utf-8",
        )

    (out / "splits.json").write_text(
        json.dumps(splits, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "dataset": "Structured3D RoomToFP V9",
        "version": "roomtofp_v9_clean_graph_polygon_targets",
        "room_types": ROOM_TYPES,
        "parser": "annotation_graph_polygonize",
        "parser_note": (
            "Room masks are built from semantics -> planeID -> planeLineMatrix -> "
            "lineJunctionMatrix -> junction graph polygonization. "
            "This avoids centroid-angle sorting and malformed target polygons."
        ),
        "observation_channel_order": OBSERVATION_CHANNEL_ORDER,
        "bev_layout": "channel_first",
        "observation_layout": "channel_last",
        "observation_channels": 10,
        "bev_channels": 10,
        "include_furniture": bool(args.include_furniture),
        "split_seed": int(args.split_seed),
        "grid_size": int(args.grid_size),
        "max_rooms": int(args.max_rooms),
        "max_corners": int(args.max_corners),
        "max_corners_note": (
            "Sparse auxiliary corner target only. "
            "V9 primary polygon supervision should use boundary_64 and boundary_128."
        ),
        "corner_sigma": float(args.corner_sigma),
        "boundary_thickness_px": int(args.boundary_thickness_px),
        "channel_notes": {
            "0": "occupancy",
            "1": "observed_wall",
            "2": "observed_floor_or_room_interior",
            "3": "observed_opening",
            "4": "observed_furniture_or_zero",
            "5": "rgb_r",
            "6": "rgb_g",
            "7": "rgb_b",
            "8": "observed_corner_prior_not_gt",
            "9": "distance_to_observed_wall_not_gt",
        },
        "target_notes": {
            "corner_heatmap": "GT target only, not inserted into bev input.",
            "distance_to_wall_target": "GT target only, not inserted into bev input.",
            "wall_mask": "GT wall target from renderer.",
            "opening_mask": "GT opening target from renderer.",
            "room_masks": "GT per-room masks from graph-polygonized room footprints.",
            "room_mask": "GT merged room mask.",
        },
        "split_counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
        "samples": records,
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset preprocessing complete: {len(records)} samples written to {samples_dir}")
    print(
        f"Splits: train={len(splits['train'])} "
        f"val={len(splits['val'])} "
        f"test={len(splits['test'])}"
    )

    print("Channels:")
    for i, name in enumerate(OBSERVATION_CHANNEL_ORDER):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()