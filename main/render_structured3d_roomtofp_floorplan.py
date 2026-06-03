from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from room_to_floorplan_helpers.floorplan_svg import clean_mask, save_mask_png
from room_to_floorplan_helpers.geometry import save_bev_preview
from room_to_floorplan_helpers.roomtofp_scene_utils import (
    SceneGrid,
    collect_scene_points,
    load_annotation_xy,
    make_scene_bev,
    stack_roomtofp_observation,
    world_xy_to_pixels,
)
from room_to_floorplan_helpers.structured3d import FURNITURE_IDS

ROOM_COLORS = {
    "living room": "#f6d365",
    "kitchen": "#fda085",
    "bedroom": "#a1c4fd",
    "bathroom": "#84fab0",
    "balcony": "#cfd9df",
    "corridor": "#d4fc79",
    "dining room": "#fbc2eb",
    "study": "#c2e9fb",
    "office": "#e2d1c3",
    "storage": "#e5e5e5",
    "other": "#dddddd",
    "undefined": "#dddddd",
}

FURNITURE_LABELS = {
    5: "chair",
    6: "sofa",
    10: "table",
    11: "desk",
    12: "bed",
    14: "wardrobe",
    15: "cabinet",
    16: "shelf",
    17: "dresser",
    18: "fridge",
    19: "sink",
    24: "bathtub",
    25: "toilet",
    28: "tv",
    29: "microwave",
    30: "oven",
    31: "dishwasher",
    32: "washing machine",
    33: "dryer",
    34: "other furniture",
}


# ----------------------------- annotation parsing -----------------------------
def load_annotation(scene_dir: Path) -> dict:
    path = scene_dir / "annotation_3d.json"
    if not path.exists():
        raise FileNotFoundError(f"annotation_3d.json not found in {scene_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def convert_lines_to_vertices(lines: list[list[int]]) -> list[list[int]]:
    lines_arr = np.asarray(lines, dtype=np.int64)
    polygons: list[list[int]] = []
    polygon: Optional[list[int]] = None
    while len(lines_arr) != 0:
        if polygon is None:
            polygon = lines_arr[0].tolist()
            lines_arr = np.delete(lines_arr, 0, axis=0)
        line_idx, junc_idx = np.where(lines_arr == polygon[-1])
        if len(line_idx) == 0:
            break
        # If the current junction is in column 0, then the other endpoint is in column 1.
        # If the current junction is in column 1, then the other endpoint is in column 0.
        next_vertex = int(lines_arr[line_idx[0], 1 - junc_idx[0]])
        lines_arr = np.delete(lines_arr, line_idx[0], axis=0)
        if next_vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(next_vertex)
    return polygons


def _polygon_orientation_clean(xy: np.ndarray) -> np.ndarray:
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return xy
    coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
    return coords


def parse_room_polygons(
    annotation: dict,
    min_area_m2: float = 0.20,
    keep_balcony: bool = True,
) -> list[dict]:
    junctions = np.asarray([j["coordinate"] for j in annotation.get("junctions", [])], dtype=np.float32) / 1000.0
    plane_line = np.asarray(annotation.get("planeLineMatrix", []), dtype=np.uint8)
    line_junc = np.asarray(annotation.get("lineJunctionMatrix", []), dtype=np.uint8)
    if len(junctions) == 0 or plane_line.size == 0 or line_junc.size == 0:
        return []

    rooms: list[dict] = []
    skip_types = {"outwall", "door", "window"}
    if not keep_balcony:
        skip_types.add("balcony")

    for sem in annotation.get("semantics", []):
        room_type = str(sem.get("type", "undefined")).lower().strip() or "undefined"
        if room_type in skip_types:
            continue
        for plane_id in sem.get("planeID", []):
            plane_id = int(plane_id)
            if plane_id >= len(annotation.get("planes", [])):
                continue
            if str(annotation["planes"][plane_id].get("type", "")).lower() != "floor":
                continue
            line_ids = np.where(plane_line[plane_id] > 0)[0].tolist()
            pairs = [np.where(line_junc[lid] > 0)[0].tolist() for lid in line_ids]
            pairs = [p for p in pairs if len(p) == 2]
            for poly_junc_ids in convert_lines_to_vertices(pairs):
                xy = junctions[np.asarray(poly_junc_ids, dtype=np.int64), :2]
                if len(xy) < 3:
                    continue
                xy = _polygon_orientation_clean(xy)
                poly = Polygon(xy)
                if poly.is_valid and poly.area >= min_area_m2:
                    rooms.append({"type": room_type, "xy": xy, "area_m2": float(poly.area)})

    # Drop exact duplicate polygons that may be repeated in rare annotations.
    unique: list[dict] = []
    seen: set[tuple] = set()
    for room in sorted(rooms, key=lambda r: r["area_m2"], reverse=True):
        q = tuple(np.round(room["xy"].reshape(-1), 3).tolist())
        if q in seen:
            continue
        seen.add(q)
        unique.append(room)
    return unique


def cleanup_room_labels(rooms: list[dict], keep_undefined: bool = False) -> list[dict]:
    """
    Normalise room labels for cleaner final visual output.

    Structured3D sometimes uses "undefined" for valid connector/utility spaces.
    For final floorplan output we keep the geometry, but rename these spaces to a
    clearer "other" label unless the user explicitly wants raw labels.
    """
    if keep_undefined:
        return rooms
    out: list[dict] = []
    for room in rooms:
        r = dict(room)
        if str(r.get("type", "")).lower().strip() in {"", "undefined"}:
            r["type"] = "other"
        out.append(r)
    return out

def _room_boundary_union(rooms: list[dict]) -> Optional[object]:
    lines = []
    for room in rooms:
        pts = np.asarray(room["xy"], dtype=np.float32)
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            if np.linalg.norm(p1 - p0) > 0.05:
                lines.append(LineString([tuple(p0), tuple(p1)]))
    if not lines:
        return None
    return unary_union(lines)


def _segment_key(seg: np.ndarray, quant: float = 0.05) -> tuple:
    """
    identifies shared boundaries since some segments may appear repeatedly with small value differences
    - Takes a line segment with two endpoints.
    - Snaps each endpoint to a 5 cm grid.
    - Converts endpoints into tuples.
    - Sorts the endpoints so direction doesnt matter.
    - Returns a stable key for comparing/deduplicating segments.
    """
    a = tuple(np.round(seg[0] / quant).astype(int).tolist())
    b = tuple(np.round(seg[1] / quant).astype(int).tolist())
    return tuple(sorted([a, b]))


def parse_opening_lines(
    annotation: dict,
    wanted: set[str],
    rooms: Optional[list[dict]] = None,
    min_length_m: float = 0.35,
    max_wall_distance_m: float = 0.20,
    max_per_plane: int = 1,
) -> list[dict]:
    """
    Parse door/window annotation planes into one clean top-down segment per plane.

    Structured3D door/window planes are usually vertical rectangles. The previous code
    drew every edge from every plane, which created dozens of duplicate marks. This
    function keeps only the longest valid XY segment per opening plane, removes
    duplicates, and optionally keeps only segments close to room boundaries.
    """
    junctions = np.asarray([j["coordinate"] for j in annotation.get("junctions", [])], dtype=np.float32) / 1000.0
    plane_line = np.asarray(annotation.get("planeLineMatrix", []), dtype=np.uint8)
    line_junc = np.asarray(annotation.get("lineJunctionMatrix", []), dtype=np.uint8)
    if len(junctions) == 0 or plane_line.size == 0 or line_junc.size == 0:
        return []

    boundary = _room_boundary_union(rooms or [])
    out: list[dict] = []
    seen: set[tuple] = set()

    for sem in annotation.get("semantics", []):
        sem_type = str(sem.get("type", "")).lower().strip()
        if sem_type not in wanted:
            continue

        for plane_id in sem.get("planeID", []):
            plane_id = int(plane_id)
            if plane_id < 0 or plane_id >= plane_line.shape[0]:
                continue

            candidates: list[tuple[float, np.ndarray, float]] = []
            for lid in np.where(plane_line[plane_id] > 0)[0].tolist():
                jids = np.where(line_junc[lid] > 0)[0].tolist()
                if len(jids) != 2:
                    continue
                p0 = junctions[jids[0], :2]
                p1 = junctions[jids[1], :2]
                seg = np.asarray([p0, p1], dtype=np.float32)
                length = float(np.linalg.norm(p1 - p0))
                if length < min_length_m:
                    continue
                line = LineString([tuple(p0), tuple(p1)])
                dist = float(line.distance(boundary)) if boundary is not None else 0.0
                if boundary is not None and dist > max_wall_distance_m:
                    continue
                candidates.append((length, seg, dist))

            # Longest edge is usually the top-down opening span; vertical rectangle side
            # edges collapse to near-zero XY length and were already removed.
            candidates.sort(key=lambda x: (x[2], -x[0]))
            for length, seg, dist in candidates[:max_per_plane]:
                key = (sem_type, _segment_key(seg))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"type": sem_type, "xy": seg, "length_m": length, "wall_distance_m": dist})

    return out


# ----------------------------- BEV/grid helpers ------------------------------
def grid_from_bev(bev: dict) -> SceneGrid:
    return SceneGrid(
        origin_xy=np.asarray(bev["origin_xy"], dtype=np.float32),
        center_xy=np.asarray(bev["center_xy"], dtype=np.float32),
        meters_per_pixel=np.asarray(bev["meters_per_pixel"], dtype=np.float32),
        floor_z=np.asarray(bev["floor_z"], dtype=np.float32),
        height_band_above_floor=np.asarray(bev["height_band_above_floor"], dtype=np.float32),
    )


def xy_to_poly_pixels(xy: np.ndarray, grid: SceneGrid, grid_size: int) -> np.ndarray:
    px, py = world_xy_to_pixels(np.asarray(xy, dtype=np.float32), grid, grid_size)
    return np.stack([px, py], axis=1).astype(np.int32)


def make_annotation_masks(rooms: list[dict], openings: list[dict], bev: dict, grid_size: int, wall_thickness: int):
    grid = grid_from_bev(bev)
    room_fill = np.zeros((grid_size, grid_size), np.uint8)
    walls = np.zeros((grid_size, grid_size), np.uint8)
    opening_mask = np.zeros((grid_size, grid_size), np.uint8)
    for room in rooms:
        pts = xy_to_poly_pixels(room["xy"], grid, grid_size)
        cv2.fillPoly(room_fill, [pts], 1)
        cv2.polylines(walls, [pts], True, 1, thickness=wall_thickness, lineType=cv2.LINE_AA)
    for opening in openings:
        pts = xy_to_poly_pixels(opening["xy"], grid, grid_size)
        color = 1 if opening["type"] == "door" else 2
        cv2.line(opening_mask, tuple(pts[0]), tuple(pts[1]), color, thickness=max(2, wall_thickness), lineType=cv2.LINE_AA)
    return room_fill.astype(bool), walls.astype(bool), opening_mask


def make_room_type_map(rooms: list[dict], bev: dict, grid_size: int) -> np.ndarray:
    """
    Rasterise room labels into an integer map for semantic sanity checks.
    """
    room_type_map = np.full((grid_size, grid_size), -1, dtype=np.int32)
    grid = grid_from_bev(bev)
    # Fill smaller rooms last so they are not swallowed by larger overlapping polygons.
    for idx, room in enumerate(sorted(rooms, key=lambda r: float(r.get("area_m2", 0.0)), reverse=True)):
        pts = xy_to_poly_pixels(room["xy"], grid, grid_size)
        cv2.fillPoly(room_type_map, [pts], int(idx))
    return room_type_map


def _erode_room_mask(room_mask: Optional[np.ndarray], erosion_px: int) -> Optional[np.ndarray]:
    if room_mask is None:
        return None
    mask = room_mask.astype(np.uint8)
    if erosion_px <= 0:
        return mask.astype(bool)
    k = 2 * int(erosion_px) + 1
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    # Avoid accidentally deleting very narrow rooms entirely.
    if int(eroded.sum()) < max(1, int(mask.sum() * 0.50)):
        return mask.astype(bool)
    return eroded.astype(bool)


def _component_room_type(comp_mask: np.ndarray, room_type_map: Optional[np.ndarray], rooms: Optional[list[dict]]) -> str:
    if room_type_map is None or rooms is None or not rooms:
        return "unknown"
    vals = room_type_map[comp_mask]
    vals = vals[vals >= 0]
    if vals.size == 0:
        return "outside"
    idx = int(np.bincount(vals).argmax())
    ordered = sorted(rooms, key=lambda r: float(r.get("area_m2", 0.0)), reverse=True)
    if idx >= len(ordered):
        return "unknown"
    return str(ordered[idx].get("type", "unknown")).lower()


FURNITURE_FILTER_PROFILES = {
    # Class-aware thresholds are intentionally a middle ground between v2 and v3:
    # keep useful medium furniture, but keep bathroom/kitchen fixtures conservative.
    "chair": {"min_area": 60, "min_w": 6, "min_h": 6, "max_aspect": 7.5, "inside_ratio": 0.55},
    "tv": {"min_area": 55, "min_w": 6, "min_h": 5, "max_aspect": 10.0, "inside_ratio": 0.50},
    "microwave": {"min_area": 70, "min_w": 6, "min_h": 6, "max_aspect": 8.0, "inside_ratio": 0.60},
    "table": {"min_area": 100, "min_w": 8, "min_h": 8, "max_aspect": 9.0, "inside_ratio": 0.55},
    "desk": {"min_area": 95, "min_w": 8, "min_h": 8, "max_aspect": 10.0, "inside_ratio": 0.55},
    "shelf": {"min_area": 85, "min_w": 6, "min_h": 6, "max_aspect": 14.0, "inside_ratio": 0.50},
    "cabinet": {"min_area": 90, "min_w": 7, "min_h": 7, "max_aspect": 12.0, "inside_ratio": 0.55},
    "wardrobe": {"min_area": 110, "min_w": 8, "min_h": 8, "max_aspect": 12.0, "inside_ratio": 0.55},
    "dresser": {"min_area": 95, "min_w": 7, "min_h": 7, "max_aspect": 10.0, "inside_ratio": 0.55},
    "fridge": {"min_area": 90, "min_w": 7, "min_h": 7, "max_aspect": 9.0, "inside_ratio": 0.60},
    "sofa": {"min_area": 120, "min_w": 10, "min_h": 9, "max_aspect": 10.0, "inside_ratio": 0.50},
    "bed": {"min_area": 140, "min_w": 11, "min_h": 11, "max_aspect": 7.0, "inside_ratio": 0.60},
    "sink": {"min_area": 80, "min_w": 7, "min_h": 7, "max_aspect": 8.0, "inside_ratio": 0.65},
    "toilet": {"min_area": 120, "min_w": 9, "min_h": 9, "max_aspect": 5.0, "inside_ratio": 0.75},
    "bathtub": {"min_area": 150, "min_w": 12, "min_h": 9, "max_aspect": 6.0, "inside_ratio": 0.75},
    "oven": {"min_area": 80, "min_w": 7, "min_h": 7, "max_aspect": 8.0, "inside_ratio": 0.65},
    "dishwasher": {"min_area": 80, "min_w": 7, "min_h": 7, "max_aspect": 8.0, "inside_ratio": 0.65},
    "washing machine": {"min_area": 90, "min_w": 8, "min_h": 8, "max_aspect": 8.0, "inside_ratio": 0.65},
    "dryer": {"min_area": 90, "min_w": 8, "min_h": 8, "max_aspect": 8.0, "inside_ratio": 0.65},
    "other furniture": {"min_area": 130, "min_w": 9, "min_h": 9, "max_aspect": 9.0, "inside_ratio": 0.60},
}


def _profile_for_furniture(label: str, fallback_area: int, fallback_w: int, fallback_h: int, fallback_aspect: float) -> dict:
    profile = dict(
        FURNITURE_FILTER_PROFILES.get(
            label,
            {
                "min_area": int(fallback_area),
                "min_w": int(fallback_w),
                "min_h": int(fallback_h),
                "max_aspect": float(fallback_aspect),
                "inside_ratio": 0.60,
            },
        )
    )
    profile["min_area"] = max(int(profile["min_area"]), min(int(fallback_area), 120))
    profile["min_w"] = max(int(profile["min_w"]), min(int(fallback_w), 8))
    profile["min_h"] = max(int(profile["min_h"]), min(int(fallback_h), 8))
    profile["max_aspect"] = min(float(profile["max_aspect"]), max(float(fallback_aspect), float(profile["max_aspect"])))

    return profile


def _semantic_room_allowed(label: str, room_type: str, known_room_types: set[str]) -> bool:
    """
    Suppress obvious panorama projection noise using weak room-type priors.

    The rule is deliberately class-aware rather than globally strict. Generic furniture
    is allowed in most indoor rooms; bathroom fixtures and kitchen appliances are kept
    only in semantically plausible spaces when the scene has those room labels.
    """
    if room_type in {"unknown", "outside"}:
        return room_type != "outside"

    room_type = room_type.lower()
    bathroom_labels = {"toilet", "bathtub"}
    sink_labels = {"sink"}
    kitchen_only_labels = {"fridge", "microwave", "oven", "dishwasher"}
    laundry_labels = {"washing machine", "dryer"}

    if label in bathroom_labels:
        return room_type in {"bathroom", "toilet", "undefined"}

    if label in sink_labels:
        if "bathroom" in known_room_types or "kitchen" in known_room_types:
            return room_type in {"bathroom", "kitchen", "undefined"}
        return room_type not in {"balcony", "outside"}

    if label in kitchen_only_labels:
        if "kitchen" in known_room_types:
            return room_type in {"kitchen", "undefined"}
        return room_type not in {"balcony", "outside"}

    if label in laundry_labels:
        return room_type in {"bathroom", "kitchen", "storage", "undefined"}

    return room_type not in {"balcony", "outside"}


def _box_iou_xywh(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def furniture_components_from_semantic(
    points_xyz: np.ndarray,
    semantic: np.ndarray,
    bev: dict,
    grid_size: int,
    z_clip: tuple[float, float],
    floor_band: tuple[float, float],
    room_mask: Optional[np.ndarray] = None,
    rooms: Optional[list[dict]] = None,
    min_area_px: int = 120,
    min_width_px: int = 8,
    min_height_px: int = 8,
    max_aspect_ratio: float = 12.0,
    room_erosion_px: int = 1,
    max_components_per_class: int = 40,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build a cleaner object-level furniture mask from projected semantic points.

    This is still semantic projection, not learned detection. The class-aware filtering below is designed to keep useful medium furniture
    while removing panorama/depth noise such as 1-pixel strips, tiny fragments,
    bathroom fixtures projected into living spaces, and furniture outside the
    actual room polygons.
    """
    grid = grid_from_bev(bev)
    pts = np.asarray(points_xyz, dtype=np.float32)
    sem = np.asarray(semantic)
    floor_z = float(bev["floor_z"][0])
    keep_base = (
        (pts[:, 2] >= z_clip[0])
        & (pts[:, 2] <= z_clip[1])
        & (pts[:, 2] >= floor_z + floor_band[0])
        & (pts[:, 2] <= floor_z + floor_band[1])
    )

    valid_room_mask = _erode_room_mask(room_mask, room_erosion_px)
    room_type_map = make_room_type_map(rooms or [], bev, grid_size) if rooms else None
    known_room_types = {str(r.get("type", "unknown")).lower() for r in (rooms or [])}

    full_mask = np.zeros((grid_size, grid_size), np.uint8)
    components: list[dict] = []

    for sid in sorted(int(x) for x in FURNITURE_IDS):
        keep = keep_base & (sem == sid)
        if not np.any(keep):
            continue

        mask = np.zeros((grid_size, grid_size), np.float32)
        px, py = world_xy_to_pixels(pts[keep, :2], grid, grid_size)
        for x, y in zip(px, py):
            cv2.circle(mask, (int(x), int(y)), 3, 1.0, thickness=-1)

        label_name = FURNITURE_LABELS.get(sid, f"furniture_{sid}")
        profile = _profile_for_furniture(label_name, min_area_px, min_width_px, min_height_px, max_aspect_ratio)
        eff_area = int(profile["min_area"])
        eff_w = int(profile["min_w"])
        eff_h = int(profile["min_h"])
        eff_aspect = float(profile["max_aspect"])
        eff_inside_ratio = float(profile.get("inside_ratio", 0.60))

        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        if valid_room_mask is not None:
            # Clip before cleaning as well as after cleaning. This prevents boundary/wall
            # projection noise from being joined to valid furniture by morphology.
            mask = mask * valid_room_mask.astype(np.float32)

        cleaned = clean_mask(
            mask,
            threshold=0.065,
            close_kernel=5,
            open_kernel=3,
            min_area_px=max(20, eff_area // 3),
            fill_holes=True,
        )

        if valid_room_mask is not None:
            cleaned = cleaned & valid_room_mask

        cleaned_u8 = cleaned.astype(np.uint8)
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned_u8, connectivity=8)
        class_components = []

        for lab in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[lab]]
            if area < eff_area:
                continue
            if w < eff_w or h < eff_h:
                continue
            aspect = float(max(w, h) / max(1, min(w, h)))
            if aspect > eff_aspect:
                continue

            comp_mask = labels == lab
            if valid_room_mask is not None:
                inside_ratio = float(np.count_nonzero(comp_mask & valid_room_mask) / max(1, np.count_nonzero(comp_mask)))
                if inside_ratio < eff_inside_ratio:
                    continue

            room_type = _component_room_type(comp_mask, room_type_map, rooms)
            if not _semantic_room_allowed(label_name, room_type, known_room_types):
                continue

            ys, xs = np.where(comp_mask)
            if len(xs) < eff_area:
                continue
            rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            box = cv2.boxPoints(rect).astype(np.int32)
            comp = {
                "semantic_id": sid,
                "label": label_name,
                "room_type": room_type,
                "area_px": int(area),
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "aspect_ratio": float(aspect),
                "oriented_box": box.tolist(),
                "filter_min_area_px": int(eff_area),
            }
            class_components.append((area, comp, comp_mask))

        class_components.sort(key=lambda t: t[0], reverse=True)

        # Simple same-class NMS/containment suppression to remove duplicate fragments.
        kept_for_class: list[tuple[int, dict, np.ndarray]] = []
        for area, comp, comp_mask in class_components:
            duplicate = False
            for kept_area, kept_comp, _kept_mask in kept_for_class:
                iou = _box_iou_xywh(comp["bbox_xywh"], kept_comp["bbox_xywh"])
                smaller = min(area, kept_area)
                if iou > 0.35 and smaller / max(1, max(area, kept_area)) < 0.55:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept_for_class.append((area, comp, comp_mask))
            if len(kept_for_class) >= max_components_per_class:
                break

        for _area, comp, comp_mask in kept_for_class:
            components.append(comp)
            full_mask[comp_mask] = 1

    components.sort(key=lambda c: c["area_px"], reverse=True)
    return full_mask.astype(bool), components


# ---------------------- annotation-first furniture extraction ----------------------
def _annotation_plane_xy(annotation: dict, plane_id: int) -> np.ndarray:
    """
    Return unique XY coordinates, in metres, for all junctions belonging to one annotation plane
    """
    junctions = np.asarray([j["coordinate"] for j in annotation.get("junctions", [])], dtype=np.float32) / 1000.0
    plane_line = np.asarray(annotation.get("planeLineMatrix", []), dtype=np.uint8)
    line_junc = np.asarray(annotation.get("lineJunctionMatrix", []), dtype=np.uint8)
    if len(junctions) == 0 or plane_line.size == 0 or line_junc.size == 0 or plane_id >= plane_line.shape[0]:
        return np.empty((0, 2), dtype=np.float32)
    ids = []
    for lid in np.where(plane_line[int(plane_id)] > 0)[0].tolist():
        ids.extend(np.where(line_junc[lid] > 0)[0].tolist())
    ids = sorted(set(int(i) for i in ids if 0 <= int(i) < len(junctions)))
    if not ids:
        return np.empty((0, 2), dtype=np.float32)
    return junctions[np.asarray(ids, dtype=np.int64), :2].astype(np.float32)


def _normalise_furniture_semantic_type(raw: str) -> Optional[str]:
    t = str(raw or "").lower().strip().replace("_", " ").replace("-", " ")
    aliases = {
        "bookcase": "shelf",
        "bookshelf": "shelf",
        "wardrobe cabinet": "wardrobe",
        "closet": "wardrobe",
        "couch": "sofa",
        "refrigerator": "fridge",
        "television": "tv",
        "wash basin": "sink",
        "basin": "sink",
    }
    t = aliases.get(t, t)
    known = set(FURNITURE_LABELS.values())
    if t in known:
        return t
    if "chair" in t:
        return "chair"
    if "sofa" in t or "couch" in t:
        return "sofa"
    if "table" in t:
        return "table"
    if "desk" in t:
        return "desk"
    if "bed" in t:
        return "bed"
    if "shelf" in t or "bookcase" in t:
        return "shelf"
    if "cabinet" in t:
        return "cabinet"
    if "wardrobe" in t or "closet" in t:
        return "wardrobe"
    if "fridge" in t or "refrigerator" in t:
        return "fridge"
    if "toilet" in t:
        return "toilet"
    if "bath" in t:
        return "bathtub"
    if "sink" in t or "basin" in t:
        return "sink"
    return None


def _component_from_mask(mask: np.ndarray, label_name: str, source: str, room_type_map: Optional[np.ndarray], rooms: Optional[list[dict]]) -> Optional[dict]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    room_type = _component_room_type(mask.astype(bool), room_type_map, rooms)
    return {
        "semantic_id": -1,
        "label": label_name,
        "source": source,
        "area_px": int(np.count_nonzero(mask)),
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "oriented_box": box.tolist(),
        "room_type": room_type,
        "confidence": 1.0 if source == "annotation" else 0.5,
    }


def furniture_components_from_annotation(
    annotation: dict,
    bev: dict,
    grid_size: int,
    room_mask: Optional[np.ndarray] = None,
    rooms: Optional[list[dict]] = None,
    min_area_px: int = 80,
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract furniture footprints from annotation_3d.json when object semantics exist.

    Some Structured3D releases expose furniture/object semantics as planes in
    annotation_3d.json. When available, these footprints are cleaner than panorama
    semantic projection, so this function is used first and panorama projection is
    kept as a fallback. If a scene only contains room/outwall/door/window semantics,
    this returns an empty result.
    """
    room_type_map = make_room_type_map(rooms or [], bev, grid_size) if rooms else None
    known_room_types = {str(r.get("type", "unknown")).lower() for r in (rooms or [])}
    full_mask = np.zeros((grid_size, grid_size), np.uint8)
    components: list[dict] = []
    grid = grid_from_bev(bev)

    for sem in annotation.get("semantics", []):
        label = _normalise_furniture_semantic_type(sem.get("type", ""))
        if label is None:
            continue
        if not sem.get("planeID"):
            continue

        sem_mask = np.zeros((grid_size, grid_size), np.uint8)
        for plane_id in sem.get("planeID", []):
            xy = _annotation_plane_xy(annotation, int(plane_id))
            if len(xy) < 2:
                continue
            px, py = world_xy_to_pixels(xy, grid, grid_size)
            pts = np.stack([px, py], axis=1).astype(np.int32)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts)
                area = abs(float(cv2.contourArea(hull)))
                if area >= 4.0:
                    cv2.fillPoly(sem_mask, [hull], 1)
                else:
                    cv2.polylines(sem_mask, [hull], False, 1, thickness=6)
            else:
                cv2.line(sem_mask, tuple(pts[0]), tuple(pts[-1]), 1, thickness=6)

        if room_mask is not None:
            sem_mask = (sem_mask.astype(bool) & room_mask.astype(bool)).astype(np.uint8)
        sem_mask = clean_mask(sem_mask.astype(np.float32), threshold=0.5, close_kernel=5, open_kernel=3, min_area_px=max(20, int(min_area_px * 0.5)), fill_holes=True).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(sem_mask, connectivity=8)
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            comp_mask = labels == cid
            room_type = _component_room_type(comp_mask, room_type_map, rooms)
            if not _semantic_room_allowed(label, room_type, known_room_types):
                continue
            comp = _component_from_mask(comp_mask.astype(np.uint8), label, "annotation", room_type_map, rooms)
            if comp is not None:
                components.append(comp)
                full_mask[comp_mask] = 1

    # Remove duplicate boxes of the same class.
    components.sort(key=lambda c: (c.get("confidence", 0.0), c["area_px"]), reverse=True)
    kept: list[dict] = []
    for comp in components:
        if any(comp["label"] == k["label"] and _box_iou_xywh(comp["bbox_xywh"], k["bbox_xywh"]) > 0.55 for k in kept):
            continue
        kept.append(comp)
    if len(kept) != len(components):
        full_mask[:] = 0
        for comp in kept:
            box = np.asarray(comp["oriented_box"], dtype=np.int32)
            cv2.fillPoly(full_mask, [box], 1)
    return full_mask.astype(bool), kept


# ------------------------------- visual outputs ------------------------------
def _xml_text(s: str) -> str:
    return s.replace("&", "and").replace("<", "").replace(">", "")


def _svg_rect(parts: list[str], x: int, y: int, w: int, h: int, fill: str = "#8b4513", stroke: str = "#5a2d0c", opacity: float = 0.42, rx: int = 2) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.5"/>')


def _append_svg_furniture_symbol(parts: list[str], comp: dict) -> None:
    """Draw simple floorplan-style furniture symbols instead of only generic boxes."""
    x, y, w, h = [int(v) for v in comp["bbox_xywh"]]
    label = str(comp.get("label", "furniture"))
    if w <= 0 or h <= 0:
        return
    if label == "bed":
        _svg_rect(parts, x, y, w, h, "#9b6b43", "#5a2d0c", 0.35, 3)
        _svg_rect(parts, x + 2, y + 2, max(3, w // 3), max(3, h // 3), "#ffffff", "#5a2d0c", 0.65, 2)
    elif label == "sofa":
        _svg_rect(parts, x, y, w, h, "#8b4513", "#5a2d0c", 0.36, 4)
        parts.append(f'<line x1="{x+2}" y1="{y+max(2,h//4)}" x2="{x+w-2}" y2="{y+max(2,h//4)}" stroke="#5a2d0c" stroke-width="1"/>')
    elif label in {"table", "desk"}:
        _svg_rect(parts, x, y, w, h, "#b07d45", "#5a2d0c", 0.34, 3)
    elif label == "chair":
        _svg_rect(parts, x, y, w, h, "#c08a5a", "#5a2d0c", 0.32, 2)
    elif label in {"shelf", "cabinet", "wardrobe", "dresser", "fridge"}:
        _svg_rect(parts, x, y, w, h, "#7a6a58", "#3d3328", 0.30, 2)
        parts.append(f'<line x1="{x+w/2:.1f}" y1="{y+1}" x2="{x+w/2:.1f}" y2="{y+h-1}" stroke="#3d3328" stroke-width="1"/>')
    elif label in {"toilet", "sink"}:
        parts.append(f'<ellipse cx="{x+w/2:.1f}" cy="{y+h/2:.1f}" rx="{max(2,w/2):.1f}" ry="{max(2,h/2):.1f}" fill="#75c9c8" fill-opacity="0.35" stroke="#357b7a" stroke-width="1.5"/>')
    elif label == "bathtub":
        _svg_rect(parts, x, y, w, h, "#75c9c8", "#357b7a", 0.30, 6)
    else:
        box = np.asarray(comp["oriented_box"], dtype=np.int32)
        pts_s = " ".join(f"{int(px)},{int(py)}" for px, py in box)
        parts.append(f'<polygon points="{pts_s}" fill="#8b4513" fill-opacity="0.34" stroke="#5a2d0c" stroke-width="1.5"/>')
    if w * h > 450:
        parts.append(f'<text x="{x + w/2:.1f}" y="{y + h/2:.1f}" font-size="8" text-anchor="middle" fill="#5a2d0c">{_xml_text(label)}</text>')


def _draw_preview_furniture_symbol(draw: ImageDraw.ImageDraw, comp: dict) -> None:
    x, y, w, h = [int(v) for v in comp["bbox_xywh"]]
    label = str(comp.get("label", "furniture"))
    if w <= 0 or h <= 0:
        return
    base = (139, 69, 19, 95)
    outline = (90, 45, 12, 230)
    if label in {"sink", "toilet", "bathtub"}:
        base = (117, 201, 200, 95)
        outline = (53, 123, 122, 230)
    elif label in {"shelf", "cabinet", "wardrobe", "dresser", "fridge"}:
        base = (110, 96, 80, 85)
        outline = (61, 51, 40, 230)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=3, fill=base, outline=outline, width=1)
    if label == "bed":
        draw.rectangle([x + 2, y + 2, x + max(4, w // 3), y + max(4, h // 3)], fill=(255, 255, 255, 120), outline=outline)
    elif label in {"shelf", "cabinet", "wardrobe", "dresser", "fridge"}:
        draw.line([x + w / 2, y + 1, x + w / 2, y + h - 1], fill=outline, width=1)
    elif label == "sofa":
        draw.line([x + 2, y + max(2, h // 4), x + w - 2, y + max(2, h // 4)], fill=outline, width=1)
    if w * h > 500:
        draw.text((float(x + w / 2), float(y + h / 2)), label, fill=outline, anchor="mm")

def save_svg(out_path: Path, rooms: list[dict], openings: list[dict], furniture_components: list[dict], bev: dict, grid_size: int) -> None:
    grid = grid_from_bev(bev)
    mpp = float(bev["meters_per_pixel"][0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{grid_size*mpp:.3f}m" height="{grid_size*mpp:.3f}m" viewBox="0 0 {grid_size} {grid_size}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for room in rooms:
        pts = xy_to_poly_pixels(room["xy"], grid, grid_size)
        pts_s = " ".join(f"{x},{y}" for x, y in pts)
        color = ROOM_COLORS.get(room["type"], ROOM_COLORS["undefined"])
        parts.append(f'<polygon points="{pts_s}" fill="{color}" fill-opacity="0.35" stroke="#111111" stroke-width="3"/>')
        cx, cy = pts.mean(axis=0)
        label = _xml_text(room["type"])
        parts.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="10" text-anchor="middle" fill="#333333">{label}</text>')

    for comp in furniture_components:
        _append_svg_furniture_symbol(parts, comp)

    for opening in openings:
        pts = xy_to_poly_pixels(opening["xy"], grid, grid_size)
        stroke = "#1f77b4" if opening["type"] == "door" else "#17becf"
        parts.append(f'<line x1="{pts[0,0]}" y1="{pts[0,1]}" x2="{pts[1,0]}" y2="{pts[1,1]}" stroke="{stroke}" stroke-width="4" stroke-linecap="round"/>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def save_preview_png(out_path: Path, rooms: list[dict], openings: list[dict], furniture_components: list[dict], bev: dict, grid_size: int) -> None:
    grid = grid_from_bev(bev)
    img = Image.new("RGB", (grid_size, grid_size), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    for room in rooms:
        pts = xy_to_poly_pixels(room["xy"], grid, grid_size)
        color = ROOM_COLORS.get(room["type"], ROOM_COLORS["undefined"])
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        draw.polygon([tuple(p) for p in pts], fill=(*rgb, 80), outline=(0, 0, 0, 255))
        cx, cy = pts.mean(axis=0)
        draw.text((float(cx), float(cy)), room["type"], fill=(40, 40, 40, 255), anchor="mm")

    draw = ImageDraw.Draw(img, "RGBA")
    for comp in furniture_components:
        _draw_preview_furniture_symbol(draw, comp)

    for opening in openings:
        pts = xy_to_poly_pixels(opening["xy"], grid, grid_size)
        col = (31, 119, 180, 255) if opening["type"] == "door" else (23, 190, 207, 255)
        draw.line([tuple(pts[0]), tuple(pts[1])], fill=col, width=4)
    img.convert("RGB").save(out_path)


def _save_roomtofp_bev_preview(obs: np.ndarray, wall_mask: np.ndarray, furniture_mask: np.ndarray, out_png: Path) -> None:
    # save_bev_preview expects the old target layout [wall, furniture], but works fine
    # for visualization of the new observation too.
    save_bev_preview(obs, np.stack([wall_mask.astype(np.float32), furniture_mask.astype(np.float32)], axis=0), out_png)


def build_roomtofp_scene_outputs(scene_dir: Path, output_dir: Path, variants: list[str], args) -> dict:
    # Inspired by RoomFormer - use all panorama RGB-D views to build a scene-level BEV,
    # then use polygon annotations for clean vector floorplan supervision.
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = load_annotation(scene_dir)
    keep_balcony = not getattr(args, "exclude_balcony", False)
    rooms = parse_room_polygons(annotation, min_area_m2=args.min_room_area_m2, keep_balcony=keep_balcony)
    rooms = cleanup_room_labels(rooms, keep_undefined=getattr(args, "keep_undefined_label", False))
    if not rooms:
        raise ValueError("No room floor polygons were parsed from annotation_3d.json")

    openings = parse_opening_lines(
        annotation,
        {"door", "window"},
        rooms=rooms,
        min_length_m=args.min_opening_length_m,
        max_wall_distance_m=args.max_opening_wall_distance_m,
        max_per_plane=1,
    )

    annotation_xy = load_annotation_xy(scene_dir)
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
        annotation_xy_m=annotation_xy,
    )
    obs = stack_roomtofp_observation(bev)
    furniture_source = getattr(args, "furniture_source", "off")
    if furniture_source == "off" and obs.shape[-1] >= 5:
        obs[..., 4] = 0.0
    room_fill, wall_mask, opening_mask = make_annotation_masks(rooms, openings, bev, args.grid_size, args.wall_thickness)
    # Improvement v3: keep renderer masks binary for fair comparison with BCE/Dice-trained model outputs.
    wall_mask = np.clip(wall_mask.astype(np.float32), 0.0, 1.0).astype(bool)
    opening_mask = np.clip((opening_mask > 0).astype(np.float32), 0.0, 1.0).astype(bool)
    furniture_mask = np.zeros_like(room_fill, dtype=bool)
    furniture_components: list[dict] = []
    if furniture_source in {"auto", "annotation", "both"}:
        ann_mask, ann_components = furniture_components_from_annotation(
            annotation,
            bev,
            args.grid_size,
            room_mask=room_fill,
            rooms=rooms,
            min_area_px=max(40, int(args.min_furniture_area_px * 0.6)),
        )
        if furniture_source == "annotation" or (furniture_source == "auto" and len(ann_components) > 0):
            furniture_mask, furniture_components = ann_mask, ann_components
        elif furniture_source == "both" and len(ann_components) > 0:
            furniture_mask = ann_mask
            furniture_components = ann_components

    if furniture_source in {"auto", "semantic", "both"} and (furniture_source != "auto" or len(furniture_components) == 0):
        sem_mask, sem_components = furniture_components_from_semantic(
            pts,
            sem,
            bev,
            args.grid_size,
            (args.z_min, args.z_max),
            (args.height_lo, args.height_hi),
            room_mask=room_fill,
            rooms=rooms,
            min_area_px=args.min_furniture_area_px,
            min_width_px=args.min_furniture_width_px,
            min_height_px=args.min_furniture_height_px,
            max_aspect_ratio=args.max_furniture_aspect_ratio,
            room_erosion_px=args.furniture_room_erosion_px,
        )
        if furniture_source == "both" and furniture_components:
            existing = furniture_components[:]
            for comp in sem_components:
                if any(comp["label"] == e["label"] and _box_iou_xywh(comp["bbox_xywh"], e["bbox_xywh"]) > 0.40 for e in existing):
                    continue
                comp["source"] = "semantic"
                comp["confidence"] = float(comp.get("confidence", 0.55))
                furniture_components.append(comp)
            furniture_mask = np.logical_or(furniture_mask, sem_mask)
        else:
            furniture_mask, furniture_components = sem_mask, sem_components

    furniture_components.sort(key=lambda c: (float(c.get("confidence", 0.5)), int(c.get("area_px", 0))), reverse=True)

    np.save(output_dir / "roomtofp_observation.npy", obs)
    np.save(output_dir / "room_polygons.npy", np.array([r["xy"] for r in rooms], dtype=object), allow_pickle=True)
    np.save(output_dir / "furniture_components.npy", np.array(furniture_components, dtype=object), allow_pickle=True)
    np.save(output_dir / "furniture_mask.npy", furniture_mask.astype(np.float32))
    np.save(output_dir / "wall_mask.npy", wall_mask.astype(np.float32))
    np.save(output_dir / "opening_mask.npy", opening_mask.astype(np.float32))

    _save_roomtofp_bev_preview(obs, wall_mask, furniture_mask, output_dir / "bev_preview.png")
    save_mask_png(wall_mask, output_dir / "walls_mask.png")
    save_mask_png(room_fill, output_dir / "room_fill_mask.png")
    save_mask_png(opening_mask > 0, output_dir / "opening_mask.png")
    save_mask_png(furniture_mask, output_dir / "furniture_mask.png")
    save_preview_png(output_dir / "floorplan_preview.png", rooms, openings, furniture_components, bev, args.grid_size)
    save_svg(output_dir / "floorplan.svg", rooms, openings, furniture_components, bev, args.grid_size)

    metadata = {
        "scene_dir": str(scene_dir),
        "rooms": [{"type": r["type"], "num_vertices": int(len(r["xy"])), "area_m2": float(r.get("area_m2", 0.0))} for r in rooms],
        "num_openings": len(openings),
        "openings": [
            {
                "type": o["type"],
                "length_m": float(o.get("length_m", 0.0)),
                "wall_distance_m": float(o.get("wall_distance_m", 0.0)),
            }
            for o in openings
        ],
        "num_furniture_components": len(furniture_components),
        "furniture_components": furniture_components,
        "furniture_pixels": int(np.count_nonzero(furniture_mask)),
        "furniture_source": furniture_source,
        "grid_size": args.grid_size,
        "observation_channels": int(obs.shape[-1]),
        "observation_channel_order": ["occupancy", "wall", "floor", "opening", "furniture", "rgb_r", "rgb_g", "rgb_b"],
        "meters_per_pixel": float(bev["meters_per_pixel"][0]),
        "outputs": {"floorplan_svg": str(output_dir / "floorplan.svg"), "preview_png": str(output_dir / "floorplan_preview.png")},
        "sources": sources,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description="Immediate Structured3D scene -> RoomToFP polygon floorplan renderer. Furniture is optional and off by default.")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variants", default="full", help="Structured3D panorama variant(s) to use, e.g. full or full,empty")
    ap.add_argument("--rgb-name", default="rgb_rawlight.png")
    ap.add_argument("--grid-size", type=int, default=768)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--z-min", type=float, default=-0.5)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--height-lo", type=float, default=0.15)
    ap.add_argument("--height-hi", type=float, default=1.60)
    ap.add_argument("--xy-padding-m", type=float, default=0.75)
    ap.add_argument("--splat-radius", type=int, default=2)
    ap.add_argument("--wall-thickness", type=int, default=5)
    ap.add_argument("--min-room-area-m2", type=float, default=0.20)
    ap.add_argument("--exclude-balcony", action="store_true")
    ap.add_argument("--keep-undefined-label", action="store_true", help="Keep raw Structured3D 'undefined' room labels instead of remapping them to 'other'.")
    ap.add_argument("--min-opening-length-m", type=float, default=0.35)
    ap.add_argument("--max-opening-wall-distance-m", type=float, default=0.20)
    ap.add_argument("--min-furniture-area-px", type=int, default=120)
    ap.add_argument("--min-furniture-width-px", type=int, default=8)
    ap.add_argument("--min-furniture-height-px", type=int, default=8)
    ap.add_argument("--max-furniture-aspect-ratio", type=float, default=12.0)
    ap.add_argument("--furniture-room-erosion-px", type=int, default=1)
    ap.add_argument("--furniture-source", choices=["off", "auto", "annotation", "semantic", "both"], default="off", help="Furniture is off by default. Use auto/annotation/semantic/both only when you want furniture components.")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    metadata = build_roomtofp_scene_outputs(Path(args.scene_dir), Path(args.output_dir), variants, args)
    print("=" * 88)
    print("Complete")
    print(
        f"Rooms: {len(metadata['rooms'])} | openings: {metadata['num_openings']} | "
        f"furniture components: {metadata['num_furniture_components']} | furniture pixels: {metadata['furniture_pixels']}"
    )
    print(f"SVG: {metadata['outputs']['floorplan_svg']}")
    print(f"Preview: {metadata['outputs']['preview_png']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
