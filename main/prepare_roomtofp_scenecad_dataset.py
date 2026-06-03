from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to [0, 1].
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.size == 0:
        return arr.astype(np.float32)

    mn = float(arr.min())
    mx = float(arr.max())

    if mx - mn <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return ((arr - mn) / (mx - mn)).astype(np.float32)


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """
    Remove connected components smaller than min_area_px.
    """
    mask = (mask > 0).astype(np.uint8)

    if min_area_px <= 1:
        return mask.astype(bool)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area >= int(min_area_px):
            out[labels == i] = 1

    return out.astype(bool)


def polygon_area(poly: np.ndarray) -> float:
    """
    Shoelace polygon area.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def remove_duplicate_points(poly: np.ndarray, eps: float = 1e-4) -> np.ndarray:
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
    Extract largest Polygon from a Shapely geometry.
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
    Rotate polygon sequence so it starts at the top-left-most point.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) == 0:
        return poly.astype(np.float32)

    start = int(np.lexsort((poly[:, 0], poly[:, 1]))[0])

    return np.roll(poly, -start, axis=0).astype(np.float32)


def _remove_tiny_edges(poly: np.ndarray, min_edge_px: float = 2.0) -> np.ndarray:
    """
    Remove tiny consecutive polygon edges.
    """
    poly = np.asarray(poly, dtype=np.float32)

    if len(poly) <= 3:
        return poly.astype(np.float32)

    cleaned = [poly[0]]

    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= float(min_edge_px):
            cleaned.append(p)

    cleaned = np.asarray(cleaned, dtype=np.float32)

    if len(cleaned) > 3 and np.linalg.norm(cleaned[0] - cleaned[-1]) < float(min_edge_px):
        cleaned = cleaned[:-1]

    return cleaned.astype(np.float32)


def clean_polygon_for_v9(
    points: np.ndarray,
    simplify_eps_px: float = 1.0,
    min_area_px: float = 16.0,
    min_edge_px: float = 2.0,
) -> np.ndarray:
    """
    Clean polygon points for V9 targets.

    Important:
        This does not sort points around a centroid.
        It preserves polygon boundary order and repairs invalid geometry.
    """
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    pts = remove_duplicate_points(pts)

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
    cleaned = remove_duplicate_points(cleaned)
    cleaned = _remove_tiny_edges(cleaned, min_edge_px=min_edge_px)

    if len(cleaned) < 3 or polygon_area(cleaned) < float(min_area_px):
        return np.zeros((0, 2), dtype=np.float32)

    cleaned = _ensure_clockwise_image_coords(cleaned)
    cleaned = _start_polygon_at_top_left(cleaned)

    return cleaned.astype(np.float32)


def normalize_points(points: np.ndarray, size: int) -> np.ndarray:
    """
    Convert pixel coordinates to normalized [0, 1] coordinates.
    """
    pts = np.asarray(points, dtype=np.float32).copy()

    pts[..., 0] /= max(1, size - 1)
    pts[..., 1] /= max(1, size - 1)

    return np.clip(pts, 0.0, 1.0).astype(np.float32)


def resample_polygon_boundary(poly: np.ndarray, num_points: int = 64) -> np.ndarray:
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


def pad_polygon(poly: np.ndarray, max_corners: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad or truncate sparse polygon corners.

    This is auxiliary only.
    Dense boundary_64 / boundary_128 should be the main V9 geometry target.
    """
    padded = np.zeros((max_corners, 2), dtype=np.float32)
    valid = np.zeros((max_corners,), dtype=np.float32)

    n = min(len(poly), max_corners)

    if n > 0:
        padded[:n] = poly[:n]
        valid[:n] = 1.0

    return padded, valid


def rasterize_polygon(poly: np.ndarray, size: int) -> np.ndarray:
    """
    Rasterize filled polygon mask.
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    if len(poly) >= 3:
        pts = np.round(poly).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)
        cv2.fillPoly(mask, [pts], 1)

    return mask.astype(np.float32)


def rasterize_edges(poly: np.ndarray, size: int, thickness: int = 2) -> np.ndarray:
    """
    Rasterize polygon boundary.
    """
    edge = np.zeros((size, size), dtype=np.uint8)

    if len(poly) >= 2:
        pts = np.round(poly).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)

        cv2.polylines(
            edge,
            [pts],
            isClosed=True,
            color=1,
            thickness=int(thickness),
            lineType=cv2.LINE_AA,
        )

    return edge.astype(np.float32)


def make_corner_heatmap_from_polygons(
    polygons: List[np.ndarray],
    size: int,
    sigma: float = 2.5,
) -> np.ndarray:
    """
    Create GT corner heatmap from polygon vertices.
    """
    heatmap = np.zeros((size, size), dtype=np.float32)

    radius = int(max(3, sigma * 3))

    for poly in polygons:
        for x, y in poly:
            cx = int(round(float(x)))
            cy = int(round(float(y)))

            if cx < 0 or cx >= size or cy < 0 or cy >= size:
                continue

            x0 = max(0, cx - radius)
            x1 = min(size, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(size, cy + radius + 1)

            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
            heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


def make_distance_transform(edge_mask: np.ndarray) -> np.ndarray:
    """
    Normalized distance-to-wall map.

    0 near wall, 1 far from wall.
    """
    mask = np.asarray(edge_mask)

    if mask.ndim == 3:
        mask = mask[0]

    mask = (mask > 0).astype(np.uint8)
    inverse = (mask == 0).astype(np.uint8)

    dist = cv2.distanceTransform(inverse, cv2.DIST_L2, 5).astype(np.float32)

    if float(dist.max()) > 1e-6:
        dist = dist / float(dist.max())

    return dist.astype(np.float32)


def compute_bbox_from_poly(poly: np.ndarray) -> np.ndarray:
    """
    Return bbox [x, y, w, h] in pixels.
    """
    if len(poly) == 0:
        return np.zeros((4,), dtype=np.float32)

    x1 = float(poly[:, 0].min())
    y1 = float(poly[:, 1].min())
    x2 = float(poly[:, 0].max())
    y2 = float(poly[:, 1].max())

    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


# ============================================================
# SceneCAD observation / pseudo-room extraction
# ============================================================

def make_observed_wall_prior_from_density(density: np.ndarray) -> np.ndarray:
    """
    Build observed wall/boundary prior from SceneCAD density image only.

    This is input-derived and safe to place in BEV channel 1.
    """
    d = normalize01(density)

    if float(d.max()) <= 1e-8:
        return np.zeros_like(d, dtype=np.float32)

    d_blur = cv2.GaussianBlur(d, (0, 0), 0.8)
    d_u8 = np.clip(d_blur * 255.0, 0, 255).astype(np.uint8)

    # Canny edges catch thin wall strokes.
    canny = cv2.Canny(d_u8, 20, 80).astype(np.float32) / 255.0

    # High-density pixels catch dark/strong wall regions.
    positive = d_blur[d_blur > 0]

    if len(positive) > 0:
        thresh = float(np.percentile(positive, 62))
    else:
        thresh = 0.5

    dense = (d_blur >= thresh).astype(np.float32)

    wall = np.maximum(canny, dense)
    wall = cv2.morphologyEx(
        wall.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(np.float32)

    wall = cv2.GaussianBlur(wall, (0, 0), 0.5)

    return np.clip(normalize01(wall), 0.0, 1.0).astype(np.float32)


def make_corner_prior_from_observed_wall(
    observed_wall: np.ndarray,
    sigma: float = 2.5,
    max_corners: int = 512,
) -> np.ndarray:
    """
    Build weak input corner prior from observed wall evidence.

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


def make_wall_binary_for_room_split(
    density: np.ndarray,
    observed_wall_prior: np.ndarray,
    layout_mask: np.ndarray,
    wall_threshold: float = 0.20,
    wall_dilate_px: int = 2,
) -> np.ndarray:
    """
    Convert observed wall prior into a binary wall separator mask.

    This is used only for pseudo-room extraction.
    """
    d = normalize01(density)
    wall = normalize01(observed_wall_prior)

    positive = d[d > 0]

    if len(positive) > 0:
        d_thresh = float(np.percentile(positive, 58))
    else:
        d_thresh = 0.5

    density_bin = d >= d_thresh
    wall_bin = wall >= float(wall_threshold)

    combined = np.logical_or(density_bin, wall_bin)
    combined = np.logical_and(combined, layout_mask > 0)

    combined = cv2.morphologyEx(
        combined.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    if wall_dilate_px > 0:
        k = 2 * int(wall_dilate_px) + 1
        combined = cv2.dilate(
            combined,
            np.ones((k, k), np.uint8),
            iterations=1,
        )

    combined = remove_small_components(combined, min_area_px=12)

    return combined.astype(np.float32)


def contour_to_clean_polygon(mask: np.ndarray, simplify_eps_px: float = 2.0) -> np.ndarray:
    """
    Convert a connected mask component into a cleaned polygon.
    """
    m = (mask > 0).astype(np.uint8)

    contours, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((0, 2), dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 16:
        return np.zeros((0, 2), dtype=np.float32)

    approx = cv2.approxPolyDP(
        contour,
        epsilon=float(simplify_eps_px),
        closed=True,
    ).reshape(-1, 2).astype(np.float32)

    approx = clean_polygon_for_v9(
        approx,
        simplify_eps_px=1.0,
        min_area_px=16.0,
        min_edge_px=2.0,
    )

    return approx.astype(np.float32)


def extract_pseudo_room_polygons_from_density(
    density: np.ndarray,
    observed_wall_prior: np.ndarray,
    layout_mask: np.ndarray,
    max_rooms: int = 20,
    min_room_area_px: int = 350,
    wall_threshold: float = 0.20,
    wall_dilate_px: int = 2,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract pseudo-room polygons from SceneCAD density image.

    This is the key SceneCAD fix.

    Old behavior:
        COCO outer polygon was incorrectly used as room_mask.

    New behavior:
        COCO polygon is layout_mask.
        Density-derived wall separators split the layout interior into pseudo rooms.

    Returns:
        polygons: list of cleaned pseudo-room polygons in pixel coords.
        pseudo_wall_mask: binary wall separator mask from density.
        free_space_mask: split free-space mask before connected components.
    """
    layout = (layout_mask > 0).astype(np.uint8)

    # Keep the usable interior slightly away from outer boundary.
    layout_inner = cv2.erode(
        layout,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)

    pseudo_wall = make_wall_binary_for_room_split(
        density=density,
        observed_wall_prior=observed_wall_prior,
        layout_mask=layout_inner,
        wall_threshold=wall_threshold,
        wall_dilate_px=wall_dilate_px,
    )

    free = np.logical_and(layout_inner, pseudo_wall <= 0)

    free = cv2.morphologyEx(
        free.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    free = cv2.morphologyEx(
        free.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    free = remove_small_components(free, min_area_px=int(min_room_area_px))

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        free.astype(np.uint8),
        connectivity=8,
    )

    components = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < int(min_room_area_px):
            continue

        comp = labels == i
        poly = contour_to_clean_polygon(comp.astype(np.uint8), simplify_eps_px=2.0)

        if len(poly) < 3:
            continue

        components.append(
            {
                "area": float(polygon_area(poly)),
                "mask": comp.astype(np.float32),
                "polygon": poly.astype(np.float32),
            }
        )

    components = sorted(components, key=lambda x: x["area"], reverse=True)
    components = components[: int(max_rooms)]

    polygons = [c["polygon"] for c in components]

    return polygons, pseudo_wall.astype(np.float32), free.astype(np.float32)


# ============================================================
# COCO / SceneCAD loading
# ============================================================

def maybe_unzip(input_path: Path, work_dir: Path) -> Path:
    """
    If input_path is a zip, extract it and return the SceneCAD root.
    If input_path is already a folder, return it directly.
    """
    if input_path.is_dir():
        return input_path

    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Input must be a folder or .zip file. Got: {input_path}")

    extract_dir = work_dir / input_path.stem

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    ensure_dir(extract_dir)

    print(f"Extracting {input_path} -> {extract_dir}")

    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)

    candidates = [
        extract_dir,
        extract_dir / "scenecad",
    ]

    for c in candidates:
        if (c / "annotations").exists():
            return c

    for child in extract_dir.iterdir():
        if child.is_dir() and (child / "annotations").exists():
            return child

    raise FileNotFoundError(
        f"Could not find SceneCAD root with annotations/ inside extracted zip: {extract_dir}"
    )


def load_coco_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_lookup(coco: Dict) -> Dict[int, Dict]:
    return {int(img["id"]): img for img in coco["images"]}


def build_annotations_by_image(coco: Dict) -> Dict[int, List[Dict]]:
    anns_by_img: Dict[int, List[Dict]] = {}

    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        anns_by_img.setdefault(image_id, []).append(ann)

    return anns_by_img


def parse_coco_polygon_from_annotation(ann: Dict) -> Optional[np.ndarray]:
    """
    Parse COCO polygon segmentation.

    For SceneCAD V9, this is treated as layout / outer footprint,
    not as a room instance.
    """
    seg = ann.get("segmentation", None)

    if seg is None:
        return None

    if not isinstance(seg, list) or len(seg) == 0:
        return None

    coords = seg[0]

    if len(coords) < 6:
        return None

    poly = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    poly = remove_duplicate_points(poly)

    if len(poly) < 3:
        return None

    if polygon_area(poly) <= 1.0:
        return None

    return poly.astype(np.float32)


def make_layout_mask_from_coco_polygons(
    polygons: List[np.ndarray],
    scale_x: float,
    scale_y: float,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Build layout mask and outer boundary target from COCO segmentation polygons.

    Important:
        This is not used as room_mask.
        It is saved separately as layout_mask / layout_boundary.
    """
    layout_mask = np.zeros((output_size, output_size), dtype=np.float32)
    layout_boundary = np.zeros((output_size, output_size), dtype=np.float32)

    scaled_polys = []

    for poly in polygons:
        p = poly.copy()
        p[:, 0] *= scale_x
        p[:, 1] *= scale_y

        p[:, 0] = np.clip(p[:, 0], 0, output_size - 1)
        p[:, 1] = np.clip(p[:, 1], 0, output_size - 1)

        p = clean_polygon_for_v9(
            p,
            simplify_eps_px=1.0,
            min_area_px=16.0,
            min_edge_px=2.0,
        )

        if len(p) < 3:
            continue

        scaled_polys.append(p)

        layout_mask = np.maximum(layout_mask, rasterize_polygon(p, output_size))
        layout_boundary = np.maximum(layout_boundary, rasterize_edges(p, output_size, thickness=2))

    layout_mask = np.clip(layout_mask, 0.0, 1.0).astype(np.float32)
    layout_boundary = np.clip(layout_boundary, 0.0, 1.0).astype(np.float32)

    return layout_mask, layout_boundary, scaled_polys


# ============================================================
# Target assembly
# ============================================================

def build_targets_from_polygons(
    polygons: List[np.ndarray],
    output_size: int,
    max_rooms: int,
    max_corners: int,
    edge_thickness: int,
    corner_sigma: float,
) -> Dict[str, np.ndarray]:
    """
    Build standard V9 target fields from pseudo-room polygons.
    """
    polygons = polygons[: int(max_rooms)]

    room_masks = []
    edge_masks = []
    padded_polys = []
    valid_corner_masks = []
    boundary_64_list = []
    boundary_128_list = []
    bboxes = []
    bboxes_norm = []
    areas = []

    for poly in polygons:
        if len(poly) < 3:
            continue

        room_mask = rasterize_polygon(poly, output_size)
        edge_mask = rasterize_edges(poly, output_size, thickness=edge_thickness)

        padded_px, valid = pad_polygon(poly, max_corners=max_corners)
        padded_norm = normalize_points(padded_px, output_size)

        b64 = normalize_points(resample_polygon_boundary(poly, num_points=64), output_size)
        b128 = normalize_points(resample_polygon_boundary(poly, num_points=128), output_size)

        bbox_px = compute_bbox_from_poly(poly)
        bbox_norm = bbox_px.copy()
        bbox_norm[[0, 2]] /= max(1, output_size - 1)
        bbox_norm[[1, 3]] /= max(1, output_size - 1)

        area = polygon_area(poly)

        room_masks.append(room_mask)
        edge_masks.append(edge_mask)
        padded_polys.append(padded_norm)
        valid_corner_masks.append(valid)
        boundary_64_list.append(b64)
        boundary_128_list.append(b128)
        bboxes.append(bbox_px)
        bboxes_norm.append(bbox_norm)
        areas.append(area)

    if len(room_masks) == 0:
        return {}

    room_masks_np = np.stack(room_masks, axis=0).astype(np.float32)
    edge_masks_np = np.stack(edge_masks, axis=0).astype(np.float32)

    room_mask_merged = np.clip(np.sum(room_masks_np, axis=0), 0, 1).astype(np.float32)
    edge_mask_merged = np.clip(np.sum(edge_masks_np, axis=0), 0, 1).astype(np.float32)

    corner_heatmap = make_corner_heatmap_from_polygons(
        polygons=polygons,
        size=output_size,
        sigma=corner_sigma,
    )

    distance_to_wall_target = make_distance_transform(edge_mask_merged.astype(np.uint8))

    return {
        "polygon_xy": np.stack(padded_polys, axis=0).astype(np.float32),
        "corner_valid": np.stack(valid_corner_masks, axis=0).astype(np.float32),
        "room_valid": np.ones((len(room_masks),), dtype=np.float32),
        "room_class": np.zeros((len(room_masks),), dtype=np.int64),
        "boundary_64": np.stack(boundary_64_list, axis=0).astype(np.float32),
        "boundary_128": np.stack(boundary_128_list, axis=0).astype(np.float32),
        "room_masks": room_masks_np.astype(np.float32),
        "room_mask": room_mask_merged.astype(np.float32),
        "wall_mask": edge_mask_merged.astype(np.float32),
        "edge_mask": edge_mask_merged.astype(np.float32),
        "corner_heatmap": corner_heatmap.astype(np.float32),
        "distance_to_wall_target": distance_to_wall_target.astype(np.float32),
        "bboxes": np.stack(bboxes_norm, axis=0).astype(np.float32),
        "bboxes_px": np.stack(bboxes, axis=0).astype(np.float32),
        "areas": np.asarray(areas, dtype=np.float32),
        "num_rooms": np.asarray(len(room_masks), dtype=np.int64),
    }


# ============================================================
# Conversion
# ============================================================

def convert_split(
    scenecad_root: Path,
    output_root: Path,
    split: str,
    output_size: int = 512,
    max_rooms: int = 20,
    max_corners: int = 16,
    edge_thickness: int = 2,
    corner_sigma: float = 2.5,
    min_room_area_px: int = 350,
    min_pseudo_rooms: int = 2,
    wall_threshold: float = 0.20,
    wall_dilate_px: int = 2,
    keep_layout_only: bool = False,
    save_debug_images: bool = True,
    max_debug_images: int = 30,
) -> Dict:
    """
    Convert one split.

    SceneCAD fix:
        COCO polygons are saved as layout targets.
        Room masks are pseudo-extracted from observed density / wall evidence.
    """
    ann_path = scenecad_root / "annotations" / f"{split}.json"
    image_dir = scenecad_root / split

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    coco = load_coco_json(ann_path)
    image_lookup = build_image_lookup(coco)
    anns_by_img = build_annotations_by_image(coco)

    split_out = output_root / split
    ensure_dir(split_out)

    debug_out = output_root / "_debug" / split
    if save_debug_images:
        ensure_dir(debug_out)

    converted = 0
    skipped = 0
    layout_only_count = 0

    sample_index = []

    for image_id, img_info in sorted(image_lookup.items(), key=lambda x: x[0]):
        file_name = img_info["file_name"]
        img_path = image_dir / file_name

        if not img_path.exists():
            print(f"Missing image: {img_path}")
            skipped += 1
            continue

        anns = anns_by_img.get(image_id, [])

        if len(anns) == 0:
            skipped += 1
            continue

        coco_polygons = []

        for ann in anns:
            poly = parse_coco_polygon_from_annotation(ann)

            if poly is not None:
                coco_polygons.append(poly)

        if len(coco_polygons) == 0:
            skipped += 1
            continue

        img = Image.open(img_path).convert("L")
        original_w, original_h = img.size

        img_np = np.asarray(img).astype(np.float32) / 255.0

        img_resized = cv2.resize(
            img_np,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        scale_x = output_size / float(original_w)
        scale_y = output_size / float(original_h)

        layout_mask, layout_boundary, layout_polygons = make_layout_mask_from_coco_polygons(
            polygons=coco_polygons,
            scale_x=scale_x,
            scale_y=scale_y,
            output_size=output_size,
        )

        if float(layout_mask.max()) <= 0:
            skipped += 1
            continue

        observed_wall_prior = make_observed_wall_prior_from_density(img_resized)

        pseudo_room_polygons, pseudo_wall_mask, pseudo_free_space = extract_pseudo_room_polygons_from_density(
            density=img_resized,
            observed_wall_prior=observed_wall_prior,
            layout_mask=layout_mask,
            max_rooms=max_rooms,
            min_room_area_px=min_room_area_px,
            wall_threshold=wall_threshold,
            wall_dilate_px=wall_dilate_px,
        )

        has_room_supervision = len(pseudo_room_polygons) >= int(min_pseudo_rooms)

        if not has_room_supervision:
            if not keep_layout_only:
                skipped += 1
                continue

            # Layout-only fallback:
            # use outer layout as one target, but mark it so the V9 loader can down-weight
            # or exclude it from room-instance training.
            pseudo_room_polygons = layout_polygons[:1]
            layout_only_count += 1

        targets = build_targets_from_polygons(
            polygons=pseudo_room_polygons,
            output_size=output_size,
            max_rooms=max_rooms,
            max_corners=max_corners,
            edge_thickness=edge_thickness,
            corner_sigma=corner_sigma,
        )

        if not targets:
            skipped += 1
            continue

        input_corner_prior = make_corner_prior_from_observed_wall(
            observed_wall_prior,
            sigma=corner_sigma,
        )

        input_distance_to_wall = make_distance_transform(observed_wall_prior)

        opening_mask = np.zeros((output_size, output_size), dtype=np.float32)
        furniture_prior = np.zeros((output_size, output_size), dtype=np.float32)

        # Standard V9 input.
        #
        # No GT masks are placed into BEV.
        bev = np.stack(
            [
                img_resized.astype(np.float32),                 # 0 density / occupancy observation
                observed_wall_prior.astype(np.float32),         # 1 observed wall prior from density
                np.zeros_like(img_resized, dtype=np.float32),   # 2 floor/interior unavailable for SceneCAD
                opening_mask.astype(np.float32),                # 3 no opening supervision
                furniture_prior.astype(np.float32),             # 4 no furniture supervision
                img_resized.astype(np.float32),                 # 5 pseudo-R
                img_resized.astype(np.float32),                 # 6 pseudo-G
                img_resized.astype(np.float32),                 # 7 pseudo-B
                input_corner_prior.astype(np.float32),          # 8 input-derived corner prior
                input_distance_to_wall.astype(np.float32),      # 9 input-derived distance map
            ],
            axis=0,
        ).astype(np.float32)

        out_name = f"{split}_{image_id:06d}.npz"
        out_path = split_out / out_name

        num_rooms = int(np.asarray(targets["num_rooms"]).reshape(-1)[0])

        np.savez_compressed(
            out_path,

            # Standard V9 input.
            bev=bev.astype(np.float32),
            observation=np.transpose(bev, (1, 2, 0)).astype(np.float32),
            density=img_resized.astype(np.float32),

            # Pseudo room-instance targets.
            polygon_xy=targets["polygon_xy"].astype(np.float32),
            corner_valid=targets["corner_valid"].astype(np.float32),
            room_polygons=targets["polygon_xy"].astype(np.float32),
            room_polygon_valid=targets["corner_valid"].astype(np.float32),
            room_valid=targets["room_valid"].astype(np.float32),
            room_class=targets["room_class"].astype(np.int64),

            boundary_64=targets["boundary_64"].astype(np.float32),
            boundary_128=targets["boundary_128"].astype(np.float32),

            room_masks=targets["room_masks"].astype(np.float32),
            room_mask=targets["room_mask"].astype(np.float32),

            # Pseudo targets derived from density, not real SceneCAD semantic labels.
            wall_mask=targets["wall_mask"].astype(np.float32),
            edge_mask=targets["edge_mask"].astype(np.float32),
            opening_mask=opening_mask.astype(np.float32),

            corner_heatmap=targets["corner_heatmap"].astype(np.float32),
            distance_to_wall_target=targets["distance_to_wall_target"].astype(np.float32),

            # Input priors saved separately.
            input_wall_prior=observed_wall_prior.astype(np.float32),
            input_corner_prior=input_corner_prior.astype(np.float32),
            distance_to_wall=input_distance_to_wall.astype(np.float32),

            # SceneCAD-specific layout targets.
            layout_mask=layout_mask.astype(np.float32),
            layout_boundary=layout_boundary.astype(np.float32),
            pseudo_wall_mask=pseudo_wall_mask.astype(np.float32),
            pseudo_free_space=pseudo_free_space.astype(np.float32),

            # Metadata.
            bboxes=targets["bboxes"].astype(np.float32),
            bboxes_px=targets["bboxes_px"].astype(np.float32),
            areas=targets["areas"].astype(np.float32),
            num_rooms=np.asarray(num_rooms, dtype=np.int64),
            has_room_supervision=np.asarray(1 if has_room_supervision else 0, dtype=np.int64),
            is_layout_only=np.asarray(0 if has_room_supervision else 1, dtype=np.int64),
            has_opening_supervision=np.asarray(0, dtype=np.int64),
            source=np.asarray("scenecad"),
            supervision_type=np.asarray(
                "pseudo_rooms_from_density" if has_room_supervision else "layout_only"
            ),
            image_id=np.asarray(image_id, dtype=np.int64),
            original_size=np.asarray([original_h, original_w], dtype=np.int64),
            output_size=np.asarray(output_size, dtype=np.int64),
            grid_size=np.asarray(output_size, dtype=np.int64),
        )

        sample_index.append(
            {
                "file": str(out_path.relative_to(output_root)),
                "image_id": int(image_id),
                "source_image": file_name,
                "num_rooms": int(num_rooms),
                "has_room_supervision": bool(has_room_supervision),
                "is_layout_only": bool(not has_room_supervision),
                "has_opening_supervision": False,
                "original_size": [int(original_h), int(original_w)],
                "output_size": int(output_size),
            }
        )

        if save_debug_images and converted < max_debug_images:
            debug_img = make_debug_visual(
                density=img_resized,
                observed_wall_prior=observed_wall_prior,
                layout_mask=layout_mask,
                pseudo_wall_mask=pseudo_wall_mask,
                room_mask=targets["room_mask"],
                edge_mask=targets["edge_mask"],
                corner_heatmap=targets["corner_heatmap"],
                room_polygons=pseudo_room_polygons,
            )
            debug_path = debug_out / f"{split}_{image_id:06d}_debug.png"
            cv2.imwrite(str(debug_path), debug_img)

        converted += 1

        if converted % 100 == 0:
            print(f"{split}: converted {converted} samples...")

    return {
        "split": split,
        "converted": converted,
        "skipped": skipped,
        "layout_only_count": layout_only_count,
        "total_images": len(image_lookup),
        "samples": sample_index,
    }


def make_debug_visual(
    density: np.ndarray,
    observed_wall_prior: np.ndarray,
    layout_mask: np.ndarray,
    pseudo_wall_mask: np.ndarray,
    room_mask: np.ndarray,
    edge_mask: np.ndarray,
    corner_heatmap: np.ndarray,
    room_polygons: List[np.ndarray],
) -> np.ndarray:
    """
    Debug image with three panels:

    1. Observed density + observed wall prior
    2. COCO layout mask / layout boundary
    3. Pseudo room targets
    """
    base = np.clip(normalize01(density) * 255.0, 0, 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Panel 1: input-derived wall prior.
    p1 = base_rgb.copy()
    red = np.zeros_like(p1)
    red[:, :, 2] = np.clip(observed_wall_prior * 255, 0, 255).astype(np.uint8)
    p1 = cv2.addWeighted(p1, 1.0, red, 0.8, 0)

    # Panel 2: layout mask.
    p2 = base_rgb.copy()
    green = np.zeros_like(p2)
    green[:, :, 1] = np.clip(layout_mask * 180, 0, 180).astype(np.uint8)
    p2 = cv2.addWeighted(p2, 1.0, green, 0.55, 0)

    # Panel 3: pseudo room target.
    p3 = np.zeros_like(p1)
    room_green = np.zeros_like(p3)
    room_green[:, :, 1] = np.clip(room_mask * 210, 0, 210).astype(np.uint8)
    p3 = cv2.addWeighted(p3, 1.0, room_green, 0.8, 0)

    wall_red = np.zeros_like(p3)
    wall_red[:, :, 2] = np.clip(edge_mask * 255, 0, 255).astype(np.uint8)
    p3 = cv2.addWeighted(p3, 1.0, wall_red, 0.9, 0)

    blue = np.zeros_like(p3)
    blue[:, :, 0] = np.clip(corner_heatmap * 255, 0, 255).astype(np.uint8)
    p3 = cv2.addWeighted(p3, 1.0, blue, 0.75, 0)

    for poly in room_polygons:
        if len(poly) >= 3:
            pts = np.round(poly).astype(np.int32)
            cv2.polylines(p3, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

    return np.concatenate([p1, p2, p3], axis=1)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare processed SceneCAD dataset for RoomToFP V9 using pseudo-room extraction."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to scenecad folder or scenecad.zip.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder for processed .npz files.",
    )

    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--max-rooms", type=int, default=20)
    parser.add_argument("--max-corners", type=int, default=16)
    parser.add_argument("--edge-thickness", type=int, default=2)
    parser.add_argument("--corner-sigma", type=float, default=2.5)

    parser.add_argument(
        "--min-room-area-px",
        type=int,
        default=350,
        help="Minimum connected free-space component area to become a pseudo room.",
    )

    parser.add_argument(
        "--min-pseudo-rooms",
        type=int,
        default=2,
        help="Minimum pseudo rooms required to keep a sample as room-supervised.",
    )

    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=0.20,
        help="Threshold for observed wall prior during pseudo-room splitting.",
    )

    parser.add_argument(
        "--wall-dilate-px",
        type=int,
        default=2,
        help="Dilate wall separators before connected-component room extraction.",
    )

    parser.add_argument(
        "--keep-layout-only",
        action="store_true",
        help=(
            "Keep samples where pseudo-room extraction fails by saving the COCO layout "
            "as one layout-only target. Recommended off for room-instance training."
        ),
    )

    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--max-debug-images", type=int, default=30)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)

    ensure_dir(output_root)

    temp_root = output_root / "_tmp_extract"
    ensure_dir(temp_root)

    scenecad_root = maybe_unzip(input_path, temp_root)

    print(f"SceneCAD root: {scenecad_root}")
    print(f"Output root: {output_root}")

    all_summaries = {}

    for split in ["train", "val"]:
        print(f"\nConverting split: {split}")

        summary = convert_split(
            scenecad_root=scenecad_root,
            output_root=output_root,
            split=split,
            output_size=args.output_size,
            max_rooms=args.max_rooms,
            max_corners=args.max_corners,
            edge_thickness=args.edge_thickness,
            corner_sigma=args.corner_sigma,
            min_room_area_px=args.min_room_area_px,
            min_pseudo_rooms=args.min_pseudo_rooms,
            wall_threshold=args.wall_threshold,
            wall_dilate_px=args.wall_dilate_px,
            keep_layout_only=args.keep_layout_only,
            save_debug_images=not args.no_debug,
            max_debug_images=args.max_debug_images,
        )

        all_summaries[split] = summary

        print(
            f"{split}: converted={summary['converted']}, "
            f"skipped={summary['skipped']}, "
            f"layout_only={summary['layout_only_count']}"
        )

    metadata = {
        "dataset": "SceneCAD processed RoomFormer format",
        "version": "roomtofp_v9_scenecad_pseudo_rooms_from_density",
        "output_size": args.output_size,
        "max_rooms": args.max_rooms,
        "max_corners": args.max_corners,
        "bev_layout": "channel_first",
        "observation_layout": "channel_last",
        "observation_channels": 10,
        "bev_channels": 10,
        "important_note": (
            "COCO segmentation is treated as layout_mask / outer footprint, not as room_mask. "
            "room_masks are pseudo-extracted from density-derived wall separators."
        ),
        "bev_channels_description": {
            "0": "density_or_occupancy_observation",
            "1": "observed_wall_prior_from_density",
            "2": "floor_or_room_interior_prior_zero_for_scenecad",
            "3": "opening_placeholder_zero",
            "4": "furniture_placeholder_zero",
            "5": "pseudo_rgb_r_density",
            "6": "pseudo_rgb_g_density",
            "7": "pseudo_rgb_b_density",
            "8": "observed_corner_prior_not_gt",
            "9": "distance_to_observed_wall_prior_not_gt",
        },
        "target_notes": {
            "room_masks": "Pseudo room masks from density/free-space connected components.",
            "room_mask": "Merged pseudo room mask.",
            "layout_mask": "Original COCO segmentation footprint.",
            "layout_boundary": "Original COCO layout boundary.",
            "wall_mask": "Pseudo room boundary target, not true SceneCAD wall annotation.",
            "opening_mask": "All zeros; processed SceneCAD has no opening labels.",
            "has_opening_supervision": "0 for all samples.",
        },
        "pseudo_room_settings": {
            "min_room_area_px": args.min_room_area_px,
            "min_pseudo_rooms": args.min_pseudo_rooms,
            "wall_threshold": args.wall_threshold,
            "wall_dilate_px": args.wall_dilate_px,
            "keep_layout_only": bool(args.keep_layout_only),
        },
        "splits": {
            split: {
                "converted": all_summaries[split]["converted"],
                "skipped": all_summaries[split]["skipped"],
                "layout_only_count": all_summaries[split]["layout_only_count"],
                "total_images": all_summaries[split]["total_images"],
            }
            for split in all_summaries
        },
    }

    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_root / "index_train.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries["train"]["samples"], f, indent=2)

    with open(output_root / "index_val.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries["val"]["samples"], f, indent=2)

    print("\nSceneCAD preparation complete.")
    print(f"Metadata saved to: {output_root / 'metadata.json'}")
    print(f"Debug images saved to: {output_root / '_debug'}")


if __name__ == "__main__":
    main()