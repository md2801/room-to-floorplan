from __future__ import annotations

"""
RoomToFP v7.4 clean-vector inference.

v7.4 improvements (includes all v7.3 improvements):
- Uses best_ema.pt cleanly as the recommended checkpoint.
- Adds automatic multi-candidate layout selection instead of relying on one brittle
  room/corner threshold pair.
- Scores each candidate layout using scene evidence:
    1) query room-mask support,
    2) dense wall-mask support,
    3) opening evidence support,
    4) room-overlap penalty,
    5) missed room-evidence/gap penalty,
    6) polygon complexity penalty.
- Keeps polygon-first output, but uses query masks as a controlled fallback only
  when the mask is clearly better than the polygon.
- Adds wall-mask-assisted Manhattan snapping so shared wall axes come from both
  predicted polygons and dense wall evidence.
- Adds conservative living-room gap filling using room-mask evidence only. This
  helps central living spaces absorb small evidence-supported holes without
  blindly filling every empty area.
- Keeps BEV-assisted opening extraction and snapped vector openings from v7.
- v7.4 adds IoU/recall/precision-based layout scoring so selected layouts
  cover dense room evidence better, instead of only sitting on high-confidence pixels.
- v7.4 adds suspicious-layout fallback for under-predicted scenes where only
  1-2 rooms are selected despite strong dense room evidence.
- v7.4 penalises long diagonal/non-Manhattan edges to avoid broken polygon
  ordering artifacts.
- v7.4 stops automatically rectangularising uncertain other/undefined rooms
  and filters weak uncertain rooms.
- v7.4 merges nearby duplicate openings and removes implausibly long openings.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from room_to_floorplan_helpers.floorplan_svg import save_mask_png
from room_to_floorplan_helpers.roomtofp_polygon_model import RoomToFPPolygonNet
from room_to_floorplan_helpers.roomtofp_scene_utils import (
    collect_scene_points,
    load_annotation_xy,
    make_scene_bev,
    stack_roomtofp_observation,
)

ROOM_TYPES = [
    "undefined", "living room", "kitchen", "bedroom", "bathroom", "balcony",
    "corridor", "dining room", "study", "office", "storage", "other",
]

ROOM_COLORS = {
    "living room": "#f6d365", "kitchen": "#fda085", "bedroom": "#a1c4fd",
    "bathroom": "#84fab0", "balcony": "#cfd9df", "corridor": "#d4fc79",
    "dining room": "#fbc2eb", "study": "#c2e9fb", "office": "#e2d1c3",
    "storage": "#e5e5e5", "other": "#dddddd", "undefined": "#dddddd",
}


def _positive_int(value, fallback: int) -> int:
    try:
        value = int(value)
        return value if value > 0 else int(fallback)
    except Exception:
        return int(fallback)


def _infer_model_dims_from_state_dict(state_dict: dict) -> dict:
    dims = {}
    if "room_queries.weight" in state_dict:
        dims["max_rooms"] = int(state_dict["room_queries.weight"].shape[0])
        dims["embed_dim"] = int(state_dict["room_queries.weight"].shape[1])
    if "corner_queries.weight" in state_dict:
        dims["max_corners"] = int(state_dict["corner_queries.weight"].shape[0])
    if "class_head.1.bias" in state_dict:
        dims["room_classes"] = int(state_dict["class_head.1.bias"].numel())
    if "backbone.stem.0.weight" in state_dict:
        dims["in_channels"] = int(state_dict["backbone.stem.0.weight"].shape[1])
    elif "encoder.stem.0.weight" in state_dict:
        dims["in_channels"] = int(state_dict["encoder.stem.0.weight"].shape[1])
    return dims


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _xml_text(s: str) -> str:
    return str(s).replace("&", "and").replace("<", "").replace(">", "")


def _polygon_area_px(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts)))


def _sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= 2:
        return pts.astype(np.int32)
    c = pts.mean(axis=0)
    a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(a)].astype(np.int32)


def _simplify_polygon_points(points: np.ndarray, eps_px: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    if len(pts) <= 3 or eps_px <= 0:
        return pts
    approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon=float(eps_px), closed=True).reshape(-1, 2)
    return approx.astype(np.int32) if len(approx) >= 3 else pts


def _snap_axis_aligned(points: np.ndarray, threshold_px: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    if len(pts) < 3 or threshold_px <= 0:
        return pts.astype(np.int32)
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        dx = abs(float(pts[j, 0] - pts[i, 0]))
        dy = abs(float(pts[j, 1] - pts[i, 1]))
        if dx <= threshold_px and dy > threshold_px:
            x = 0.5 * (pts[i, 0] + pts[j, 0])
            pts[i, 0] = x
            pts[j, 0] = x
        elif dy <= threshold_px and dx > threshold_px:
            y = 0.5 * (pts[i, 1] + pts[j, 1])
            pts[i, 1] = y
            pts[j, 1] = y
    return np.clip(np.round(pts), 0, None).astype(np.int32)


def _remove_consecutive_duplicate_points(points: np.ndarray, min_dist_px: float = 2.0) -> np.ndarray:
    """
    Preserve polygon order while removing only repeated/near-repeated consecutive vertices.

    Do NOT use np.unique for polygons because it sorts points and destroys polygon order.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)

    cleaned = []
    for p in pts:
        if not cleaned:
            cleaned.append(p)
            continue
        if np.linalg.norm(p - cleaned[-1]) >= float(min_dist_px):
            cleaned.append(p)

    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) < float(min_dist_px):
        cleaned.pop()

    if len(cleaned) < 3:
        return pts.astype(np.int32)

    return np.asarray(cleaned, dtype=np.float32).round().astype(np.int32)


def _remove_short_edges(points: np.ndarray, min_edge_px: float = 8.0, max_iters: int = 3) -> np.ndarray:
    """
    Remove tiny edges that create small notches/spikes.

    This keeps polygon order and removes vertices responsible for very short edges.
    """
    pts = np.asarray(points, dtype=np.float32)

    for _ in range(int(max_iters)):
        if len(pts) <= 3:
            break

        keep = np.ones(len(pts), dtype=bool)
        changed = False

        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            edge_len = float(np.linalg.norm(pts[j] - pts[i]))

            if edge_len < float(min_edge_px):
                # Remove the next vertex, which usually collapses the tiny edge.
                keep[j] = False
                changed = True

        new_pts = pts[keep]
        if not changed or len(new_pts) < 3:
            break

        pts = new_pts

    return np.round(pts).astype(np.int32)


def _remove_nearly_collinear_points(points: np.ndarray, area_eps_px: float = 6.0, axis_eps_px: float = 3.0) -> np.ndarray:
    """
    Remove points that lie almost on a straight horizontal/vertical/diagonal segment.

    This reduces jagged stair-step polygon edges after snapping.
    """
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) <= 3:
        return np.round(pts).astype(np.int32)

    keep = []

    for i in range(len(pts)):
        prev_p = pts[(i - 1) % len(pts)]
        p = pts[i]
        next_p = pts[(i + 1) % len(pts)]

        v1 = p - prev_p
        v2 = next_p - p

        len1 = float(np.linalg.norm(v1))
        len2 = float(np.linalg.norm(v2))

        if len1 < 1e-6 or len2 < 1e-6:
            continue

        # Twice triangle area from cross product.
        cross = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))

        # Same vertical-ish or horizontal-ish line.
        same_x_line = abs(float(prev_p[0] - p[0])) <= axis_eps_px and abs(float(p[0] - next_p[0])) <= axis_eps_px
        same_y_line = abs(float(prev_p[1] - p[1])) <= axis_eps_px and abs(float(p[1] - next_p[1])) <= axis_eps_px

        # If nearly straight, drop middle point.
        if cross <= area_eps_px * max(len1, len2) or same_x_line or same_y_line:
            continue

        keep.append(p)

    if len(keep) < 3:
        return np.round(pts).astype(np.int32)

    return np.asarray(keep, dtype=np.float32).round().astype(np.int32)


def _canonical_winding_no_sort(points: np.ndarray) -> np.ndarray:
    """
    Keep the current polygon order but make winding direction consistent.

    This does NOT sort points by angle.
    """
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 3:
        return pts.astype(np.int32)

    x = pts[:, 0]
    y = pts[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    # Keep one consistent direction. The exact direction is not critical for drawing,
    # but consistency helps later processing.
    if signed_area < 0:
        pts = pts[::-1]

    return np.round(pts).astype(np.int32)


def _axis_aligned_rect_from_points(points: np.ndarray, grid_size: int, pad_px: int = 0) -> np.ndarray:
    """
    Convert a roughly rectangular room to a clean axis-aligned rectangle.

    Useful for bedrooms, bathrooms, kitchens, balconies, storage, etc.
    """
    pts = np.asarray(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    x0 = max(0, x - int(pad_px))
    y0 = max(0, y - int(pad_px))
    x1 = min(grid_size - 1, x + w + int(pad_px))
    y1 = min(grid_size - 1, y + h + int(pad_px))

    return np.asarray(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.int32,
    )


def _should_rectangularize_room(room_type: str, points: np.ndarray) -> bool:
    """
    Rectangularise simple room types only.

    Do not force living rooms/corridors into rectangles because they are often L-shaped.
    """
    room_type = str(room_type).lower().strip()

    rectangular_types = {
        "bedroom",
        "bathroom",
        "kitchen",
        "balcony",
        "storage",
        "store room",
        "study",
        "office",
    }

    if room_type not in rectangular_types:
        return False

    pts = np.asarray(points, dtype=np.int32)

    if len(pts) <= 4:
        return True

    # If the room has many vertices but is not living/corridor, it is probably a noisy rectangle.
    return len(pts) <= 8




def _should_keep_uncertain_room(room: dict, grid_size: int) -> bool:
    """
    v7.4 improvement:
    Remove weak uncertain rooms.

    'Other' and 'undefined' predictions are often false positives. Earlier code
    rectangularised them, making weak false positives look clean. v7.4 keeps
    uncertain rooms only when they are confident and large enough.
    """
    rtype = str(room.get("type", "")).lower().strip()

    if rtype not in {"other", "undefined"}:
        return True

    score = float(room.get("score", 0.0))
    area = float(room.get("area_px", 0.0))
    area_ratio = area / float(max(grid_size * grid_size, 1))

    return score >= 0.55 and area_ratio >= 0.015


def _cluster_1d_values(values: list[float], threshold: float) -> list[float]:
    """
    Cluster nearby x or y wall coordinates into shared global wall axes.

    Example:
        [100, 103, 107, 240, 244] with threshold 10
    becomes approximately:
        [103.3, 242.0]

    This helps neighbouring rooms share the same wall line.
    """
    if not values:
        return []

    vals = sorted(float(v) for v in values)
    clusters: list[list[float]] = [[vals[0]]]

    for v in vals[1:]:
        if abs(v - np.mean(clusters[-1])) <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    return [float(np.mean(c)) for c in clusters]


def _nearest_axis_value(value: float, axes: list[float], threshold: float) -> float:
    """
    Snap a coordinate to the nearest global axis if it is close enough.
    """
    if not axes:
        return float(value)

    axes_arr = np.asarray(axes, dtype=np.float32)
    idx = int(np.argmin(np.abs(axes_arr - float(value))))
    nearest = float(axes_arr[idx])

    if abs(float(value) - nearest) <= float(threshold):
        return nearest

    return float(value)


def _collect_global_wall_axes(
    rooms: list[dict],
    axis_threshold_px: float,
    min_edge_len_px: float = 20.0,
) -> tuple[list[float], list[float]]:
    """
    Collect shared x/y wall axes from all accepted room polygons.

    Near-vertical edges contribute x coordinates.
    Near-horizontal edges contribute y coordinates.
    """
    x_values: list[float] = []
    y_values: list[float] = []

    for room in rooms:
        pts = np.asarray(room["points_px"], dtype=np.float32)
        if len(pts) < 3:
            continue

        n = len(pts)

        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]

            dx = abs(float(p1[0] - p0[0]))
            dy = abs(float(p1[1] - p0[1]))
            length = float(np.linalg.norm(p1 - p0))

            if length < float(min_edge_len_px):
                continue

            # Vertical-ish wall.
            if dx <= axis_threshold_px and dy > axis_threshold_px:
                x_values.append(float(0.5 * (p0[0] + p1[0])))

            # Horizontal-ish wall.
            elif dy <= axis_threshold_px and dx > axis_threshold_px:
                y_values.append(float(0.5 * (p0[1] + p1[1])))

    x_axes = _cluster_1d_values(x_values, threshold=axis_threshold_px)
    y_axes = _cluster_1d_values(y_values, threshold=axis_threshold_px)

    return x_axes, y_axes


def _snap_polygon_to_global_axes(
    points: np.ndarray,
    x_axes: list[float],
    y_axes: list[float],
    snap_threshold_px: float,
) -> np.ndarray:
    """
    Snap polygon vertices to shared global x/y wall axes.
    """
    pts = np.asarray(points, dtype=np.float32).copy()

    for i in range(len(pts)):
        pts[i, 0] = _nearest_axis_value(float(pts[i, 0]), x_axes, snap_threshold_px)
        pts[i, 1] = _nearest_axis_value(float(pts[i, 1]), y_axes, snap_threshold_px)

    return np.round(pts).astype(np.int32)


def _suppress_overlapping_rooms_again(
    rooms: list[dict],
    grid_size: int,
    overlap_thr: float,
) -> list[dict]:
    """
    Re-run overlap suppression after global snapping.

    Global snapping may slightly move room polygons, so we suppress duplicates again.
    """
    accepted: list[dict] = []
    occupied = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # Keep larger/higher-confidence rooms first.
    sorted_rooms = sorted(
        rooms,
        key=lambda r: (
            float(r.get("score", 0.0)),
            float(r.get("area_px", 0.0)),
        ),
        reverse=True,
    )

    for room in sorted_rooms:
        pts = np.asarray(room["points_px"], dtype=np.int32)

        if len(pts) < 3:
            continue

        mask = _room_mask(pts, grid_size)
        area = float(mask.sum())

        if area <= 0:
            continue

        overlap_ratio = float((mask & occupied).sum()) / max(area, 1.0)

        if overlap_ratio > overlap_thr:
            continue

        occupied = np.maximum(occupied, mask)
        room = dict(room)
        room["overlap_ratio"] = overlap_ratio
        accepted.append(room)

    return accepted


def _room_center_and_area(room: dict) -> tuple[np.ndarray, float]:
    pts = np.asarray(room["points_px"], dtype=np.float32)
    center = pts.mean(axis=0)
    area = float(_polygon_area_px(pts))
    return center, area


def _remove_duplicate_nearby_rooms(
    rooms: list[dict],
    center_dist_px: float = 35.0,
    area_ratio_thr: float = 0.55,
) -> list[dict]:
    """
    Remove duplicate room predictions of the same/similar type.

    This helps when the model predicts two bathrooms or two small 'other' rooms
    almost on top of each other.
    """
    kept: list[dict] = []

    for room in sorted(rooms, key=lambda r: float(r.get("score", 0.0)), reverse=True):
        r_type = str(room.get("type", "other")).lower().strip()
        r_center, r_area = _room_center_and_area(room)

        duplicate = False

        for kept_room in kept:
            k_type = str(kept_room.get("type", "other")).lower().strip()

            # Only suppress same/similar small-room classes.
            similar_type = (
                r_type == k_type
                or {r_type, k_type} <= {"other", "undefined", "storage", "study", "office"}
                or {r_type, k_type} <= {"bathroom", "other", "undefined"}
            )

            if not similar_type:
                continue

            k_center, k_area = _room_center_and_area(kept_room)
            dist = float(np.linalg.norm(r_center - k_center))
            area_ratio = min(r_area, k_area) / max(r_area, k_area, 1.0)

            if dist <= center_dist_px and area_ratio >= area_ratio_thr:
                duplicate = True
                break

        if not duplicate:
            kept.append(room)

    return kept


def _global_manhattan_cleanup(
    rooms: list[dict],
    grid_size: int,
    simplify_eps_px: float,
    snap_axis_thr_px: float,
    overlap_thr: float,
    wall_prob: np.ndarray | None = None,
) -> list[dict]:
    """
    Final global layout cleanup.

    This makes predicted polygons more floorplan-like by forcing neighbouring
    rooms to share common horizontal/vertical wall axes.
    """
    if len(rooms) <= 1:
        return rooms

    # Slightly stronger than local snap threshold.
    global_snap_thr = max(float(snap_axis_thr_px), 10.0)

    x_axes, y_axes = _collect_global_wall_axes(
        rooms,
        axis_threshold_px=global_snap_thr,
        min_edge_len_px=18.0,
    )

    # v7.3 improvement:
    # Add dense wall-mask evidence to the global wall axes.
    # Earlier versions only collected axes from predicted room polygons. If a
    # polygon edge was slightly wrong, cleanup could reinforce the wrong line.
    # By adding axes from the wall probability map, snapping is guided by both
    # vector predictions and raster wall evidence.
    if wall_prob is not None:
        mx, my = _collect_mask_wall_axes(
            wall_prob,
            threshold=0.45,
            min_len_px=28,
            axis_merge_px=global_snap_thr,
        )
        x_axes = _cluster_1d_values(x_axes + mx, threshold=global_snap_thr)
        y_axes = _cluster_1d_values(y_axes + my, threshold=global_snap_thr)

    cleaned: list[dict] = []

    for room in rooms:
        pts = np.asarray(room["points_px"], dtype=np.int32)

        if len(pts) < 3:
            continue

        room_type = str(room.get("type", "other")).lower().strip()

        pts = _snap_polygon_to_global_axes(
            points=pts,
            x_axes=x_axes,
            y_axes=y_axes,
            snap_threshold_px=global_snap_thr,
        )

        # Clean again after global snapping.
        pts = _clean_ordered_polygon(
            points=pts,
            grid_size=grid_size,
            simplify_eps_px=max(2.0, simplify_eps_px * 0.75),
            snap_axis_thr_px=snap_axis_thr_px,
            min_edge_px=max(7.0, simplify_eps_px * 1.5),
            rectangularize=True,
            room_type=room_type,
        )

        if len(pts) < 3:
            continue

        area = _polygon_area_px(pts)

        if area <= 0:
            continue

        room = dict(room)
        room["points_px"] = pts.tolist()
        room["area_px"] = float(area)

        # v7.4 improvement:
        # Do not let weak "other"/"undefined" rooms survive just because
        # rectangular cleanup made them visually neat.
        if not _should_keep_uncertain_room(room, grid_size):
            continue

        cleaned.append(room)

    cleaned = _suppress_overlapping_rooms_again(
        cleaned,
        grid_size=grid_size,
        overlap_thr=overlap_thr,
    )

    cleaned = _remove_duplicate_nearby_rooms(cleaned)

    return cleaned


def _orthogonalize_polygon_edges(points: np.ndarray, axis_threshold_px: float = 12.0) -> np.ndarray:
    """
    Convert near-Manhattan polygon edges into cleaner horizontal/vertical edges.

    This preserves polygon order. It does not sort points by angle.
    It works best after simplify + global axis snapping.
    """
    pts = np.asarray(points, dtype=np.float32).copy()

    if len(pts) < 3:
        return pts.astype(np.int32)

    n = len(pts)

    for i in range(n):
        j = (i + 1) % n

        dx = abs(float(pts[j, 0] - pts[i, 0]))
        dy = abs(float(pts[j, 1] - pts[i, 1]))

        if dx < 1e-6 and dy < 1e-6:
            continue

        # If edge is closer to vertical, force same x.
        if dx <= dy and dx <= axis_threshold_px * 2.0:
            x = 0.5 * (pts[i, 0] + pts[j, 0])
            pts[i, 0] = x
            pts[j, 0] = x

        # If edge is closer to horizontal, force same y.
        elif dy < dx and dy <= axis_threshold_px * 2.0:
            y = 0.5 * (pts[i, 1] + pts[j, 1])
            pts[i, 1] = y
            pts[j, 1] = y

    return np.round(pts).astype(np.int32)


def _remove_tiny_living_room_notches(
    points: np.ndarray,
    min_notch_depth_px: float = 18.0,
    min_edge_px: float = 14.0,
) -> np.ndarray:
    """
    Remove tiny zig-zag notches from large living-room polygons.

    This keeps useful L-shapes but removes small accidental steps/spikes.
    """
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) <= 4:
        return np.round(pts).astype(np.int32)

    changed = True
    passes = 0

    while changed and passes < 4 and len(pts) > 4:
        changed = False
        passes += 1
        keep = np.ones(len(pts), dtype=bool)

        for i in range(len(pts)):
            p0 = pts[(i - 1) % len(pts)]
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]

            e1 = np.linalg.norm(p1 - p0)
            e2 = np.linalg.norm(p2 - p1)

            if e1 < min_edge_px or e2 < min_edge_px:
                keep[i] = False
                changed = True
                continue

            # If p1 creates a very small local deviation, remove it.
            base = np.linalg.norm(p2 - p0)
            if base > 1e-6:
                area2 = abs(float(np.cross(p1 - p0, p2 - p0)))
                height = area2 / base
                if height < min_notch_depth_px and min(e1, e2) < min_edge_px * 2.0:
                    keep[i] = False
                    changed = True

        if keep.sum() >= 4:
            pts = pts[keep]
        else:
            break

    return np.round(pts).astype(np.int32)


def _clean_living_room_polygon(
    points: np.ndarray,
    grid_size: int,
    simplify_eps_px: float,
    snap_axis_thr_px: float,
) -> np.ndarray:
    """
    Special cleanup for living/dining/corridor-style large spaces.

    Unlike bedrooms/bathrooms, living rooms should not always be converted to a rectangle.
    They should become clean rectilinear polygons that can preserve L-shapes.
    """
    pts = np.asarray(points, dtype=np.int32)

    if len(pts) < 3:
        return pts

    pts = _remove_consecutive_duplicate_points(pts, min_dist_px=2.0)
    pts = _canonical_winding_no_sort(pts)

    # Stronger simplification for large living spaces.
    pts = _simplify_polygon_points(pts, max(4.0, simplify_eps_px))

    # Snap to local axes.
    pts = _snap_axis_aligned(pts, max(12.0, snap_axis_thr_px))

    # Force nearly Manhattan edges.
    pts = _orthogonalize_polygon_edges(pts, axis_threshold_px=max(12.0, snap_axis_thr_px))

    # Remove little staircase effects/notches.
    pts = _remove_short_edges(pts, min_edge_px=max(12.0, simplify_eps_px * 2.0))
    pts = _remove_nearly_collinear_points(pts, area_eps_px=8.0, axis_eps_px=4.0)
    pts = _remove_tiny_living_room_notches(pts)

    # Final snap.
    pts = _snap_axis_aligned(pts, max(12.0, snap_axis_thr_px))
    pts = _orthogonalize_polygon_edges(pts, axis_threshold_px=max(12.0, snap_axis_thr_px))

    pts = np.clip(pts, 0, grid_size - 1).astype(np.int32)

    if len(pts) < 3:
        return np.asarray(points, dtype=np.int32)

    return pts


def _clean_ordered_polygon(
    points: np.ndarray,
    grid_size: int,
    simplify_eps_px: float,
    snap_axis_thr_px: float,
    min_edge_px: float = 8.0,
    rectangularize: bool = False,
    room_type: str = "",
) -> np.ndarray:
    """
    Main model-only polygon cleanup.

    Preserves polygon order, removes spikes/notches, simplifies, snaps Manhattan edges,
    and optionally rectangularises simple rooms.
    """
    room_type_lower = str(room_type).lower().strip()

    if room_type_lower in {"living room", "dining room", "corridor"}:
        return _clean_living_room_polygon(
            points=points,
            grid_size=grid_size,
            simplify_eps_px=simplify_eps_px,
            snap_axis_thr_px=snap_axis_thr_px,
        )

    pts = np.asarray(points, dtype=np.int32)

    if len(pts) < 3:
        return pts

    pts = _remove_consecutive_duplicate_points(pts, min_dist_px=2.0)

    if len(pts) < 3:
        return pts

    # Preserve existing order; do not centroid-sort.
    pts = _canonical_winding_no_sort(pts)

    # Simplify while preserving contour order.
    pts = _simplify_polygon_points(pts, simplify_eps_px)

    # Snap near-horizontal and near-vertical edges.
    pts = _snap_axis_aligned(pts, snap_axis_thr_px)

    # Remove tiny notches/spikes after snapping.
    pts = _remove_short_edges(pts, min_edge_px=min_edge_px)

    # Remove redundant points lying almost on a straight line.
    pts = _remove_nearly_collinear_points(pts)

    # One final light snap after removing vertices.
    pts = _snap_axis_aligned(pts, snap_axis_thr_px)

    pts = np.clip(pts, 0, grid_size - 1).astype(np.int32)

    if rectangularize and _should_rectangularize_room(room_type, pts):
        pts = _axis_aligned_rect_from_points(pts, grid_size)

    return pts.astype(np.int32)


def _largest_contour_polygon(mask: np.ndarray, simplify_eps_px: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    pts = cv2.approxPolyDP(c, epsilon=max(1.0, float(simplify_eps_px)), closed=True).reshape(-1, 2)
    if len(pts) < 3:
        pts = cv2.convexHull(c).reshape(-1, 2)
    return pts.astype(np.int32) if len(pts) >= 3 else None


def _room_mask(points: np.ndarray, grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


# -----------------------------
# v7.3 layout evidence helpers
# -----------------------------


def _rooms_union_mask(rooms: list[dict], grid_size: int) -> np.ndarray:
    """
    v7.3 improvement:
    Create one binary mask containing all accepted room polygons.

    This is used for layout scoring and gap detection. It lets inference check
    whether a proposed vector layout covers the same area that the model's dense
    room-mask head thinks should contain rooms.
    """
    out = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for room in rooms:
        pts = np.asarray(room.get("points_px", []), dtype=np.int32)
        if len(pts) >= 3:
            out = np.maximum(out, _room_mask(pts, grid_size))

    return out


def _polygon_boundary_mask(rooms: list[dict], grid_size: int, thickness: int = 3) -> np.ndarray:
    """
    v7.3 improvement:
    Rasterise only the polygon boundaries.

    This lets us score whether predicted vector walls agree with the model's
    dense wall probability map.
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for room in rooms:
        pts = np.asarray(room.get("points_px", []), dtype=np.int32)
        if len(pts) >= 3:
            cv2.polylines(mask, [pts], isClosed=True, color=1, thickness=int(thickness))

    return mask


def _safe_mean_on_mask(prob: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> float:
    """
    Return the mean probability inside a binary mask.

    If the mask is empty, return a stable fallback instead of producing NaN.
    """
    m = np.asarray(mask).astype(bool)

    if int(m.sum()) == 0:
        return float(fallback)

    return float(np.asarray(prob, dtype=np.float32)[m].mean())


def _layout_complexity(rooms: list[dict]) -> float:
    """
    v7.3 improvement:
    Penalise unnecessarily complex polygons.

    Ground-truth Structured3D-style floorplans are usually clean rectilinear
    polygons. If two candidate layouts have similar evidence support, the one
    with fewer unnecessary vertices is usually better.
    """
    if not rooms:
        return 0.0

    return float(sum(len(r.get("points_px", [])) for r in rooms))


def _layout_overlap_ratio(rooms: list[dict], grid_size: int) -> float:
    """
    v7.3 improvement:
    Measure how much room polygons overlap each other.

    A good final floorplan should have rooms that touch/share walls, not rooms
    that sit on top of each other.
    """
    if len(rooms) <= 1:
        return 0.0

    accum = np.zeros((grid_size, grid_size), dtype=np.uint8)
    overlap = 0
    total = 0

    for room in rooms:
        pts = np.asarray(room.get("points_px", []), dtype=np.int32)
        if len(pts) < 3:
            continue

        m = _room_mask(pts, grid_size)
        overlap += int((m & accum).sum())
        total += int(m.sum())
        accum = np.maximum(accum, m)

    return float(overlap / max(total, 1))


def _layout_gap_ratio(
    rooms: list[dict],
    room_union_prob: np.ndarray,
    grid_size: int,
    threshold: float = 0.45,
) -> float:
    """
    v7.3 improvement:
    Penalise layouts that ignore strong room-mask evidence.

    This helps when a threshold setting produces clean polygons but misses an
    obvious room or leaves a large empty area that the room-mask head strongly
    believes is part of the floorplan.
    """
    pred_union = _rooms_union_mask(rooms, grid_size).astype(bool)
    evidence = np.asarray(room_union_prob, dtype=np.float32) >= float(threshold)

    if int(evidence.sum()) == 0:
        return 0.0

    missed = evidence & (~pred_union)
    return float(missed.sum() / max(int(evidence.sum()), 1))


def _collect_mask_wall_axes(
    wall_prob: np.ndarray,
    threshold: float = 0.45,
    min_len_px: int = 28,
    axis_merge_px: float = 10.0,
) -> tuple[list[float], list[float]]:
    """
    v7.3 improvement:
    Extract likely vertical and horizontal wall axes from the dense wall mask.

    Earlier cleanup only snapped rooms to axes discovered from the room polygons
    themselves. This function lets the dense wall prediction contribute extra
    x/y axes, which improves alignment when the polygon corners are slightly off.
    """
    mask = (np.asarray(wall_prob, dtype=np.float32) >= float(threshold)).astype(np.uint8)

    k = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs: list[float] = []
    ys: list[float] = []

    for c in contours:
        if cv2.contourArea(c) < 20:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Tall and thin component -> likely vertical wall.
        if h >= int(min_len_px) and w <= max(8, h * 0.20):
            xs.append(float(x + w / 2.0))

        # Wide and thin component -> likely horizontal wall.
        if w >= int(min_len_px) and h <= max(8, w * 0.20):
            ys.append(float(y + h / 2.0))

    return (
        _cluster_1d_values(xs, threshold=axis_merge_px),
        _cluster_1d_values(ys, threshold=axis_merge_px),
    )



def _binary_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """
    v7.4 improvement:
    Compute binary IoU between two masks.

    This makes layout selection care about coverage agreement, not only whether
    selected rooms lie on high-confidence pixels.
    """
    aa = np.asarray(a).astype(bool)
    bb = np.asarray(b).astype(bool)

    inter = float((aa & bb).sum())
    union = float((aa | bb).sum())

    return float((inter + eps) / (union + eps))


def _binary_recall(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    v7.4 improvement:
    Measure how much dense room evidence is covered by the predicted vector layout.

    Low recall usually means a missed room or an under-sized layout.
    """
    pp = np.asarray(pred).astype(bool)
    tt = np.asarray(target).astype(bool)

    inter = float((pp & tt).sum())
    total = float(tt.sum())

    return float((inter + eps) / (total + eps))


def _binary_precision(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    v7.4 improvement:
    Measure how much of the predicted vector layout is supported by dense evidence.

    Low precision usually means a hallucinated or over-expanded room polygon.
    """
    pp = np.asarray(pred).astype(bool)
    tt = np.asarray(target).astype(bool)

    inter = float((pp & tt).sum())
    total = float(pp.sum())

    return float((inter + eps) / (total + eps))


def _polygon_diagonal_penalty(points: np.ndarray, min_len_px: float = 18.0) -> float:
    """
    v7.4 improvement:
    Penalise long diagonal edges.

    Structured3D-style floorplans are mostly Manhattan/rectilinear. A long
    diagonal edge is usually a bad polygon ordering artifact, not a real wall.
    """
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 3:
        return 1.0

    bad = 0
    total = 0

    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]

        dx = abs(float(p1[0] - p0[0]))
        dy = abs(float(p1[1] - p0[1]))
        length = float(np.linalg.norm(p1 - p0))

        if length < float(min_len_px):
            continue

        total += 1

        # A Manhattan edge has one component close to zero. If both components
        # are substantial, it is a diagonal/non-Manhattan edge.
        if min(dx, dy) / max(length, 1e-6) > 0.20:
            bad += 1

    return float(bad / max(total, 1))


def _layout_diagonal_penalty(rooms: list[dict]) -> float:
    """
    v7.4 improvement:
    Penalise layouts with many non-Manhattan room edges.
    """
    if not rooms:
        return 1.0

    vals = []
    for room in rooms:
        pts = np.asarray(room.get("points_px", []), dtype=np.int32)
        vals.append(_polygon_diagonal_penalty(pts))

    return float(np.mean(vals)) if vals else 1.0

def _score_layout(
    rooms: list[dict],
    grid_size: int,
    wall_prob: np.ndarray,
    opening_evidence: np.ndarray,
    room_union_prob: np.ndarray,
    wall_weight: float = 0.8,
    room_mask_weight: float = 1.0,
    opening_weight: float = 0.25,
    overlap_penalty_weight: float = 1.5,
    gap_penalty_weight: float = 0.4,
    complexity_penalty_weight: float = 0.03,
) -> float:
    """
    v7.4 improvement:
    Score a complete candidate floorplan using both local support and coverage.

    v7.3 mostly rewarded candidates whose rooms sat on high-confidence pixels.
    v7.4 also adds room-mask IoU, recall and precision so candidate selection
    prefers layouts that cover the full dense room evidence without over-growing.
    It also penalises suspiciously tiny layouts and diagonal/non-Manhattan edges.
    """
    if not rooms:
        return -1e9

    union = _rooms_union_mask(rooms, grid_size)
    boundary = _polygon_boundary_mask(rooms, grid_size, thickness=3)

    room_support = _safe_mean_on_mask(room_union_prob, union, fallback=0.0)
    wall_support = _safe_mean_on_mask(wall_prob, boundary, fallback=0.0)
    opening_support = _safe_mean_on_mask(opening_evidence, boundary, fallback=0.0)

    room_evidence_bin = np.asarray(room_union_prob, dtype=np.float32) >= 0.45
    wall_evidence_bin = np.asarray(wall_prob, dtype=np.float32) >= 0.45

    room_iou = _binary_iou(union, room_evidence_bin)
    room_recall = _binary_recall(union, room_evidence_bin)
    room_precision = _binary_precision(union, room_evidence_bin)
    boundary_iou = _binary_iou(boundary, wall_evidence_bin)

    overlap_penalty = _layout_overlap_ratio(rooms, grid_size)
    gap_penalty = _layout_gap_ratio(rooms, room_union_prob, grid_size, threshold=0.45)
    complexity_penalty = _layout_complexity(rooms) / max(len(rooms), 1)
    diagonal_penalty = _layout_diagonal_penalty(rooms)

    # v7.4 improvement:
    # Penalise layouts that select only 1-2 rooms even though the dense room-mask
    # head suggests a larger floorplan region. This catches catastrophic
    # under-prediction cases like a scene producing only living room + bathroom.
    evidence_area = float(room_evidence_bin.sum()) / float(grid_size * grid_size)
    too_few_rooms_penalty = 0.0
    if evidence_area > 0.08 and len(rooms) <= 2:
        too_few_rooms_penalty = 1.0

    # Soft reward only. Do not force a fixed room count.
    room_count_reward = min(len(rooms), 10) * 0.025

    score = (
        0.75 * room_mask_weight * room_support
        + 0.85 * room_iou
        + 0.55 * room_recall
        + 0.25 * room_precision
        + wall_weight * wall_support
        + 0.45 * boundary_iou
        + opening_weight * opening_support
        + room_count_reward
        - overlap_penalty_weight * overlap_penalty
        - gap_penalty_weight * gap_penalty
        - complexity_penalty_weight * complexity_penalty
        - 0.60 * diagonal_penalty
        - 0.75 * too_few_rooms_penalty
    )

    return float(score)

def _candidate_settings(mode: str = "balanced") -> list[dict]:
    """
    v7.3 improvement:
    Generate several general-purpose inference candidates.

    These are not tuned for a single scene. They cover a small range from
    recall-heavy to stricter/cleaner. The scoring function then chooses the best
    one using the current scene's evidence.
    """
    mode = str(mode).lower().strip()

    if mode == "strict":
        return [
            {"room_thr": 0.45, "corner_thr": 0.45, "simplify": 3.0, "snap": 10.0, "overlap": 0.55, "prefer_masks": False},
            {"room_thr": 0.50, "corner_thr": 0.45, "simplify": 3.0, "snap": 12.0, "overlap": 0.55, "prefer_masks": False},
            {"room_thr": 0.45, "corner_thr": 0.40, "simplify": 2.5, "snap": 10.0, "overlap": 0.60, "prefer_masks": False},
            {"room_thr": 0.45, "corner_thr": 0.45, "simplify": 3.0, "snap": 10.0, "overlap": 0.55, "prefer_masks": True},
        ]

    if mode == "recall":
        return [
            {"room_thr": 0.25, "corner_thr": 0.35, "simplify": 2.0, "snap": 8.0,  "overlap": 0.75, "prefer_masks": False},
            {"room_thr": 0.30, "corner_thr": 0.40, "simplify": 2.0, "snap": 8.0,  "overlap": 0.70, "prefer_masks": False},
            {"room_thr": 0.35, "corner_thr": 0.40, "simplify": 2.0, "snap": 8.0,  "overlap": 0.70, "prefer_masks": True},
            {"room_thr": 0.35, "corner_thr": 0.45, "simplify": 2.5, "snap": 10.0, "overlap": 0.65, "prefer_masks": True},
        ]

    # balanced default
    return [
        {"room_thr": 0.30, "corner_thr": 0.40, "simplify": 2.0, "snap": 8.0,  "overlap": 0.70, "prefer_masks": False},
        {"room_thr": 0.35, "corner_thr": 0.40, "simplify": 2.0, "snap": 8.0,  "overlap": 0.65, "prefer_masks": False},
        {"room_thr": 0.40, "corner_thr": 0.45, "simplify": 2.5, "snap": 10.0, "overlap": 0.65, "prefer_masks": False},
        {"room_thr": 0.45, "corner_thr": 0.45, "simplify": 3.0, "snap": 10.0, "overlap": 0.60, "prefer_masks": False},

        # Mask-assisted alternatives. These are controlled by the support check
        # inside _extract_predicted_rooms, so they do not blindly replace polygons.
        {"room_thr": 0.35, "corner_thr": 0.45, "simplify": 2.0, "snap": 8.0,  "overlap": 0.65, "prefer_masks": True},
        {"room_thr": 0.40, "corner_thr": 0.45, "simplify": 2.5, "snap": 10.0, "overlap": 0.60, "prefer_masks": True},
    ]



def _layout_is_suspicious(
    rooms: list[dict],
    score: float,
    room_union_prob: np.ndarray,
    grid_size: int,
) -> bool:
    """
    v7.4 improvement:
    Detect obviously weak layouts.

    This catches scenes where the candidate selector returns only 1-2 rooms
    even though the dense room-mask evidence suggests a much larger floorplan.
    """
    if not rooms:
        return True

    evidence = np.asarray(room_union_prob, dtype=np.float32) >= 0.45
    evidence_area_ratio = float(evidence.sum()) / float(grid_size * grid_size)

    if len(rooms) <= 2 and evidence_area_ratio > 0.08:
        return True

    if float(score) < 1.15 and evidence_area_ratio > 0.08:
        return True

    return False

def _select_best_layout(
    pred: dict,
    grid_size: int,
    mask_thr: float,
    min_room_area_px: float,
    wall_prob: np.ndarray,
    opening_evidence: np.ndarray,
    room_union_prob: np.ndarray,
    args,
) -> tuple[list[dict], dict]:
    """
    v7.3 improvement:
    Run multiple inference candidates and choose the best complete layout.

    This is the main generalisation improvement. Instead of hard-coding one
    threshold combination, each scene gets the candidate that best matches model
    evidence while avoiding overlaps, missing gaps and over-complex polygons.
    """
    best_rooms: list[dict] = []
    best_info: dict = {"score": -1e9, "settings": None, "num_rooms": 0}

    tried: list[dict] = []

    for st in _candidate_settings(args.layout_candidates):
        rooms = _extract_predicted_rooms(
            pred=pred,
            grid_size=grid_size,
            room_thr=float(st["room_thr"]),
            corner_thr=float(st["corner_thr"]),
            mask_thr=mask_thr,
            min_room_area_px=min_room_area_px,
            overlap_thr=float(st["overlap"]),
            simplify_eps_px=float(st["simplify"]),
            snap_axis_thr_px=float(st["snap"]),
            prefer_room_masks=bool(st["prefer_masks"]),
            wall_prob=wall_prob,
        )

        score = _score_layout(
            rooms=rooms,
            grid_size=grid_size,
            wall_prob=wall_prob,
            opening_evidence=opening_evidence,
            room_union_prob=room_union_prob,
            wall_weight=args.layout_score_wall,
            room_mask_weight=args.layout_score_room_mask,
            opening_weight=args.layout_score_opening,
            overlap_penalty_weight=args.layout_score_overlap_penalty,
            gap_penalty_weight=args.layout_score_gap_penalty,
            complexity_penalty_weight=args.layout_score_complexity_penalty,
        )

        rec = {
            "score": float(score),
            "settings": dict(st),
            "num_rooms": int(len(rooms)),
        }
        tried.append(rec)

        if score > float(best_info["score"]):
            best_rooms = rooms
            best_info = rec

    # v7.4 improvement:
    # If the selected candidate is suspiciously small or weak, retry with a
    # recall-heavy, mask-assisted fallback. This is designed for scenes where
    # the polygon-valid logits are too conservative and only 1-2 rooms survive.
    if _layout_is_suspicious(
        rooms=best_rooms,
        score=float(best_info["score"]),
        room_union_prob=room_union_prob,
        grid_size=grid_size,
    ):
        for st0 in _candidate_settings("recall"):
            st = dict(st0)
            st["prefer_masks"] = True
            st["room_thr"] = min(float(st["room_thr"]), 0.25)
            st["corner_thr"] = min(float(st["corner_thr"]), 0.35)
            st["overlap"] = max(float(st["overlap"]), 0.75)

            rooms = _extract_predicted_rooms(
                pred=pred,
                grid_size=grid_size,
                room_thr=float(st["room_thr"]),
                corner_thr=float(st["corner_thr"]),
                mask_thr=mask_thr,
                min_room_area_px=max(40.0, float(min_room_area_px) * 0.6),
                overlap_thr=float(st["overlap"]),
                simplify_eps_px=float(st["simplify"]),
                snap_axis_thr_px=float(st["snap"]),
                prefer_room_masks=bool(st["prefer_masks"]),
                wall_prob=wall_prob,
            )

            score = _score_layout(
                rooms=rooms,
                grid_size=grid_size,
                wall_prob=wall_prob,
                opening_evidence=opening_evidence,
                room_union_prob=room_union_prob,
                wall_weight=args.layout_score_wall,
                room_mask_weight=args.layout_score_room_mask,
                opening_weight=args.layout_score_opening,
                overlap_penalty_weight=args.layout_score_overlap_penalty,
                gap_penalty_weight=args.layout_score_gap_penalty,
                complexity_penalty_weight=args.layout_score_complexity_penalty,
            )

            rec = {
                "score": float(score),
                "settings": dict(st),
                "num_rooms": int(len(rooms)),
                "fallback": True,
            }
            tried.append(rec)

            if score > float(best_info["score"]):
                best_rooms = rooms
                best_info = rec

    best_info = dict(best_info)
    best_info["tried"] = tried

    return best_rooms, best_info


def _fill_living_room_gaps(
    rooms: list[dict],
    grid_size: int,
    room_union_prob: np.ndarray,
    max_gap_area_ratio: float = 0.12,
    evidence_thr: float = 0.45,
) -> list[dict]:
    """
    v7.3 improvement:
    Conservative gap filling for large living/corridor/dining spaces.

    This is not a scene-specific trick. It fills only small gaps that:
    - are supported by dense room-mask evidence,
    - are not already covered by any room polygon,
    - touch or sit very near the largest living/corridor/dining room,
    - are not huge missing regions.

    This helps central living rooms look closer to ground-truth layouts while
    avoiding aggressive hallucinated room expansion.
    """
    if not rooms:
        return rooms

    living_indices = [
        i for i, r in enumerate(rooms)
        if str(r.get("type", "")).lower().strip() in {"living room", "corridor", "dining room"}
    ]

    if not living_indices:
        return rooms

    union = _rooms_union_mask(rooms, grid_size).astype(np.uint8)
    evidence = (np.asarray(room_union_prob, dtype=np.float32) >= float(evidence_thr)).astype(np.uint8)

    gap = ((evidence > 0) & (union == 0)).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)

    if num <= 1:
        return rooms

    living_idx = max(
        living_indices,
        key=lambda i: _polygon_area_px(np.asarray(rooms[i]["points_px"], dtype=np.int32)),
    )

    living_mask = _room_mask(np.asarray(rooms[living_idx]["points_px"], dtype=np.int32), grid_size)
    add_mask = living_mask.copy()

    total_layout_area = max(int(union.sum()), 1)
    dilated_living = cv2.dilate(living_mask.astype(np.uint8), np.ones((9, 9), np.uint8))

    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])

        if area <= 0:
            continue

        # Prevent the living room from absorbing a huge incorrect region.
        if area / total_layout_area > float(max_gap_area_ratio):
            continue

        comp = labels == lab

        # Only fill gaps that touch or nearly touch the living-room polygon.
        if not np.any(comp & (dilated_living > 0)):
            continue

        add_mask[comp] = 1

    poly = _largest_contour_polygon(add_mask, simplify_eps_px=3.0)

    if poly is None or len(poly) < 3:
        return rooms

    poly = _clean_living_room_polygon(
        poly,
        grid_size=grid_size,
        simplify_eps_px=3.0,
        snap_axis_thr_px=10.0,
    )

    if len(poly) < 3:
        return rooms

    out = [dict(r) for r in rooms]
    out[living_idx]["points_px"] = poly.tolist()
    out[living_idx]["area_px"] = float(_polygon_area_px(poly))
    out[living_idx]["source"] = str(out[living_idx].get("source", "")) + "+gap_fill"

    return out


def _postprocess_rooms(
    rooms: list[dict],
    grid_size: int,
    min_area_px: float,
    overlap_thr: float,
    simplify_eps_px: float,
    snap_axis_thr_px: float,
    wall_prob: np.ndarray | None = None,
) -> list[dict]:
    """
    Clean predicted room polygons.

    Important v7 inference fix:
    - Preserve predicted polygon order.
    - Do NOT use np.unique + centroid angle sort, because that destroys concave L-shaped rooms.
    - Remove duplicate/short/collinear vertices instead.
    - Rectangularise simple room types where appropriate.
    """
    cleaned: list[dict] = []

    for room in sorted(rooms, key=lambda r: float(r.get("score", 0.0)), reverse=True):
        pts = np.asarray(room["points_px"], dtype=np.int32)

        if len(pts) < 3:
            continue

        room_type = str(room.get("type", "other")).lower().strip()

        pts = _clean_ordered_polygon(
            points=pts,
            grid_size=grid_size,
            simplify_eps_px=simplify_eps_px,
            snap_axis_thr_px=snap_axis_thr_px,
            min_edge_px=max(6.0, simplify_eps_px * 1.5),
            rectangularize=True,
            room_type=room_type,
        )

        if len(pts) < 3:
            continue

        area = _polygon_area_px(pts)

        if area < min_area_px:
            continue

        room = dict(room)
        room["points_px"] = pts.tolist()
        room["area_px"] = float(area)
        cleaned.append(room)

    accepted: list[dict] = []
    occupied = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for room in cleaned:
        pts = np.asarray(room["points_px"], dtype=np.int32)
        mask = _room_mask(pts, grid_size)
        area = float(mask.sum())

        if area <= 0:
            continue

        overlap_ratio = float((mask & occupied).sum()) / max(area, 1.0)

        if overlap_ratio > overlap_thr:
            continue

        occupied = np.maximum(occupied, mask)
        room["overlap_ratio"] = overlap_ratio
        accepted.append(room)

    # Improvement v7 inference: global Manhattan layout cleanup.
    # This aligns neighbouring room walls to shared x/y axes, which makes the output
    # look much closer to annotation-style floorplans.
    accepted = _global_manhattan_cleanup(
        rooms=accepted,
        grid_size=grid_size,
        simplify_eps_px=simplify_eps_px,
        snap_axis_thr_px=snap_axis_thr_px,
        overlap_thr=overlap_thr,
        wall_prob=wall_prob,
    )

    return accepted

def _extract_predicted_rooms(
    pred: dict,
    grid_size: int,
    room_thr: float,
    corner_thr: float,
    mask_thr: float,
    min_room_area_px: float,
    overlap_thr: float,
    simplify_eps_px: float,
    snap_axis_thr_px: float,
    prefer_room_masks: bool,
    wall_prob: np.ndarray | None = None,
) -> list[dict]:
    valid = torch.sigmoid(pred["room_valid_logits"])[0].detach().cpu().numpy()
    corner_valid = torch.sigmoid(pred["corner_valid_logits"])[0].detach().cpu().numpy()
    corners = pred["corner_xy"][0].detach().cpu().numpy()
    classes = torch.argmax(pred["room_class_logits"], dim=-1)[0].detach().cpu().numpy()
    room_masks = torch.sigmoid(pred["room_mask_logits"])[0].detach().cpu().numpy() if "room_mask_logits" in pred else None

    rooms: list[dict] = []
    for ridx, score in enumerate(valid):
        if float(score) < room_thr:
            continue

        keep = corner_valid[ridx] >= corner_thr
        pts = corners[ridx, keep]

        # If too few corners pass the threshold, keep the top-confidence corners instead
        # of blindly taking the first four.
        if len(pts) < 3:
            topk = np.argsort(-corner_valid[ridx])[:4]
            topk = np.sort(topk)  # preserve predicted sequence order
            pts = corners[ridx, topk]

        pts_px = np.clip(np.round(pts * (grid_size - 1)), 0, grid_size - 1).astype(np.int32)
        source = "corner_head"
        if prefer_room_masks and room_masks is not None:
            # v7.3 improvement:
            # Query masks are useful when polygon corners are weak, but using
            # masks unconditionally can create blob-like rooms. Therefore, the
            # mask polygon is only selected when it has clearly stronger support
            # from its own query mask and remains reasonably simple.
            mask_bin = (room_masks[ridx] >= mask_thr).astype(np.uint8)
            mask_poly = _largest_contour_polygon(mask_bin, simplify_eps_px)

            if mask_poly is not None and _polygon_area_px(mask_poly) >= min_room_area_px:
                corner_mask = _room_mask(pts_px, grid_size)
                mask_poly_mask = _room_mask(mask_poly, grid_size)
                query_prob = room_masks[ridx].astype(np.float32)

                corner_support = _safe_mean_on_mask(query_prob, corner_mask, fallback=0.0)
                mask_support = _safe_mean_on_mask(query_prob, mask_poly_mask, fallback=0.0)

                if mask_support > corner_support + 0.08 and len(mask_poly) <= 10:
                    pts_px = mask_poly
                    source = "query_mask_selected"
        if len(np.unique(pts_px, axis=0)) < 3:
            continue
        cls = int(classes[ridx])
        rtype = ROOM_TYPES[cls] if 0 <= cls < len(ROOM_TYPES) else "other"
        rooms.append({"query_index": int(ridx), "type": rtype, "score": float(score), "points_px": pts_px.tolist(), "source": source})

    return _postprocess_rooms(
        rooms,
        grid_size,
        min_room_area_px,
        overlap_thr,
        simplify_eps_px,
        snap_axis_thr_px,
        wall_prob=wall_prob,
    )


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if min_area_px <= 1:
        return mask_u8.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for lab in range(1, num):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= int(min_area_px):
            out[labels == lab] = 1
    return out.astype(bool)


def _clean_binary_mask(prob: np.ndarray, threshold: float, close_kernel: int = 3, open_kernel: int = 1, min_area_px: int = 8) -> np.ndarray:
    mask = (np.asarray(prob, dtype=np.float32) >= float(threshold)).astype(np.uint8)
    if close_kernel and close_kernel > 1:
        k = np.ones((int(close_kernel), int(close_kernel)), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if open_kernel and open_kernel > 1:
        k = np.ones((int(open_kernel), int(open_kernel)), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return _remove_small_components(mask.astype(bool), int(min_area_px))


def _segment_length(seg: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(seg[1], np.float32) - np.asarray(seg[0], np.float32)))


def _segment_key(seg: np.ndarray, quant_px: float = 4.0) -> tuple:
    pts = np.asarray(seg, dtype=np.float32)
    a = tuple(np.round(pts[0] / quant_px).astype(int).tolist())
    b = tuple(np.round(pts[1] / quant_px).astype(int).tolist())
    return tuple(sorted([a, b]))


def _room_wall_segments(rooms: list[dict], min_length_px: float = 8.0, dedup_quant_px: float = 4.0) -> list[dict]:
    segments: list[dict] = []
    seen: set[tuple] = set()
    for room in rooms:
        pts = np.asarray(room["points_px"], dtype=np.float32)
        if len(pts) < 3:
            continue
        for i in range(len(pts)):
            seg = np.asarray([pts[i], pts[(i + 1) % len(pts)]], dtype=np.float32)
            length = _segment_length(seg)
            if length < min_length_px:
                continue
            key = _segment_key(seg, quant_px=dedup_quant_px)
            if key in seen:
                continue
            seen.add(key)
            segments.append({"xy": seg.tolist(), "length_px": float(length), "room_type": str(room.get("type", "unknown")), "query_index": int(room.get("query_index", -1))})
    return segments


def _point_to_segment_projection(point: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, float, float]:
    p = np.asarray(point, dtype=np.float32)
    a = np.asarray(seg[0], dtype=np.float32)
    b = np.asarray(seg[1], dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-8:
        return a.copy(), float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return proj, float(np.linalg.norm(p - proj)), t


def _opening_components_to_segments(opening_mask: np.ndarray, wall_segments: list[dict], min_area_px: float, min_len_px: float, snap_dist_px: float) -> list[dict]:
    mask_u8 = opening_mask.astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if not wall_segments:
        return []
    openings: list[dict] = []
    for lab in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[lab]]
        if area < float(min_area_px):
            continue
        centroid = np.asarray(centroids[lab], dtype=np.float32)
        best = None
        best_dist = float("inf")
        best_t = 0.0
        for wall in wall_segments:
            proj, dist, t = _point_to_segment_projection(centroid, np.asarray(wall["xy"], dtype=np.float32))
            if dist < best_dist:
                best = wall
                best_dist = dist
                best_t = t
        if best is None or best_dist > float(snap_dist_px):
            continue
        wall_seg = np.asarray(best["xy"], dtype=np.float32)
        a, b = wall_seg[0], wall_seg[1]
        vec = b - a
        length = float(np.linalg.norm(vec))
        if length <= 1e-6:
            continue
        direction = vec / length
        comp_len = max(float(w), float(h), float(math.sqrt(max(area, 1))))
        max_len = max(1.0, min(length * 0.75, max(float(min_len_px), comp_len * 1.6)))
        min_len = min(float(min_len_px), max_len)
        open_len = float(np.clip(max(comp_len, min_len), min_len, max_len))
        centre_min = open_len / 2.0
        centre_max = max(centre_min, length - open_len / 2.0)
        centre_dist = float(np.clip(best_t * length, centre_min, centre_max))
        centre = a + direction * centre_dist
        p0 = centre - direction * (open_len / 2.0)
        p1 = centre + direction * (open_len / 2.0)
        openings.append({"xy": np.asarray([p0, p1], dtype=np.float32).tolist(), "length_px": float(open_len), "source_area_px": int(area), "snap_distance_px": float(best_dist), "wall_room_type": str(best.get("room_type", "unknown")), "type": "opening"})
    return openings



def _merge_nearby_openings(openings: list[dict], merge_dist_px: float = 18.0) -> list[dict]:
    """
    v7.4 improvement:
    Merge nearby openings on the same wall direction.

    Opening masks can fragment one real door/window into several blobs. Without
    merging, the final preview gets repeated cyan dashes along the same wall.
    """
    if len(openings) <= 1:
        return openings

    used = [False] * len(openings)
    merged: list[dict] = []

    for i, op in enumerate(openings):
        if used[i]:
            continue

        seg_i = np.asarray(op["xy"], dtype=np.float32)
        centre_i = seg_i.mean(axis=0)
        vec_i = seg_i[1] - seg_i[0]
        horizontal_i = abs(float(vec_i[0])) >= abs(float(vec_i[1]))

        group = [op]
        used[i] = True

        for j in range(i + 1, len(openings)):
            if used[j]:
                continue

            seg_j = np.asarray(openings[j]["xy"], dtype=np.float32)
            centre_j = seg_j.mean(axis=0)
            vec_j = seg_j[1] - seg_j[0]
            horizontal_j = abs(float(vec_j[0])) >= abs(float(vec_j[1]))

            if horizontal_i != horizontal_j:
                continue

            # Require nearby centres. This is deliberately conservative so
            # separate openings on the same long wall do not get merged.
            dist = float(np.linalg.norm(centre_i - centre_j))

            if dist <= float(merge_dist_px):
                group.append(openings[j])
                used[j] = True

        if len(group) == 1:
            merged.append(op)
            continue

        pts = np.concatenate([np.asarray(g["xy"], dtype=np.float32) for g in group], axis=0)

        if horizontal_i:
            y = float(np.mean(pts[:, 1]))
            x0 = float(np.min(pts[:, 0]))
            x1 = float(np.max(pts[:, 0]))
            new_seg = np.asarray([[x0, y], [x1, y]], dtype=np.float32)
        else:
            x = float(np.mean(pts[:, 0]))
            y0 = float(np.min(pts[:, 1]))
            y1 = float(np.max(pts[:, 1]))
            new_seg = np.asarray([[x, y0], [x, y1]], dtype=np.float32)

        new_op = dict(group[0])
        new_op["xy"] = new_seg.tolist()
        new_op["length_px"] = float(np.linalg.norm(new_seg[1] - new_seg[0]))
        new_op["source_area_px"] = int(sum(g.get("source_area_px", 0) for g in group))
        new_op["type"] = "opening_merged"
        merged.append(new_op)

    return merged


def _filter_openings_by_length_and_wall(
    openings: list[dict],
    max_wall_fraction: float = 0.55,
    max_abs_len_px: float = 80.0,
) -> list[dict]:
    """
    v7.4 improvement:
    Remove implausibly long openings.

    Ground-truth openings are usually shorter than the wall segment. This avoids
    huge cyan segments being drawn across room boundaries. max_wall_fraction is
    kept as a future-facing parameter; current opening records do not store the
    parent wall length, so v7.4 applies the absolute guard robustly.
    """
    kept: list[dict] = []

    for op in openings:
        length = float(op.get("length_px", 0.0))
        if length <= 0:
            continue

        if length > float(max_abs_len_px):
            continue

        kept.append(op)

    return kept

def _draw_debug_mask_overlay(img: Image.Image, wall_mask: np.ndarray, opening_mask: np.ndarray, furniture_mask: np.ndarray | None = None) -> Image.Image:
    w, h = img.size
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    ov[wall_mask.astype(bool)] = [0, 0, 0, 65]
    ov[opening_mask.astype(bool)] = [31, 119, 180, 115]
    if furniture_mask is not None:
        ov[furniture_mask.astype(bool)] = [139, 69, 19, 95]
    return Image.alpha_composite(img.convert("RGBA"), Image.fromarray(ov)).convert("RGB")


def _write_vector_preview(path: Path, rooms: list[dict], wall_segments: list[dict], opening_segments: list[dict], grid_size: int, debug_mask_overlays: bool = False, wall_mask: np.ndarray | None = None, opening_mask: np.ndarray | None = None, furniture_mask: np.ndarray | None = None) -> None:
    img = Image.new("RGB", (grid_size, grid_size), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    for room in rooms:
        pts = [tuple(p) for p in room["points_px"]]
        color = _hex_to_rgb(ROOM_COLORS.get(room["type"], ROOM_COLORS["undefined"]))
        draw.polygon(pts, fill=(*color, 70), outline=None)
    for wall in wall_segments:
        seg = np.asarray(wall["xy"], dtype=np.float32)
        draw.line([tuple(seg[0]), tuple(seg[1])], fill=(17, 17, 17, 235), width=3)
    for opening in opening_segments:
        seg = np.asarray(opening["xy"], dtype=np.float32)
        draw.line([tuple(seg[0]), tuple(seg[1])], fill=(0, 188, 212, 255), width=5)
    for room in rooms:
        arr = np.asarray(room["points_px"], dtype=np.float32)
        if len(arr) == 0:
            continue
        cx, cy = arr.mean(axis=0)
        draw.text((float(cx), float(cy)), room["type"], fill=(45, 45, 45, 255), anchor="mm")
    if debug_mask_overlays and wall_mask is not None and opening_mask is not None:
        img = _draw_debug_mask_overlay(img, wall_mask, opening_mask, furniture_mask)
    img.save(path)


def _write_vector_svg(path: Path, rooms: list[dict], wall_segments: list[dict], opening_segments: list[dict], grid_size: int, debug_mask_overlays: bool = False, wall_mask: np.ndarray | None = None, opening_mask: np.ndarray | None = None, furniture_mask: np.ndarray | None = None) -> None:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{grid_size}" height="{grid_size}" viewBox="0 0 {grid_size} {grid_size}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<g id="rooms">')
    for room in rooms:
        pts_s = " ".join(f"{int(x)},{int(y)}" for x, y in room["points_px"])
        color = ROOM_COLORS.get(room["type"], ROOM_COLORS["undefined"])
        parts.append(f'<polygon points="{pts_s}" fill="{color}" fill-opacity="0.30" stroke="none"/>')
    parts.append('</g>')
    parts.append('<g id="walls" fill="none" stroke="#111111" stroke-width="3" stroke-linecap="round">')
    for wall in wall_segments:
        seg = np.asarray(wall["xy"], dtype=np.float32)
        parts.append(f'<line x1="{seg[0,0]:.1f}" y1="{seg[0,1]:.1f}" x2="{seg[1,0]:.1f}" y2="{seg[1,1]:.1f}"/>')
    parts.append('</g>')
    parts.append('<g id="openings" fill="none" stroke="#00bcd4" stroke-width="5" stroke-linecap="round">')
    for opening in opening_segments:
        seg = np.asarray(opening["xy"], dtype=np.float32)
        parts.append(f'<line x1="{seg[0,0]:.1f}" y1="{seg[0,1]:.1f}" x2="{seg[1,0]:.1f}" y2="{seg[1,1]:.1f}"/>')
    parts.append('</g>')
    if debug_mask_overlays and wall_mask is not None and opening_mask is not None:
        parts.append('<g id="debug-mask-overlays">')
        overlays = [(wall_mask, "#111111", 0.20), (opening_mask, "#1f77b4", 0.35)]
        if furniture_mask is not None:
            overlays.append((furniture_mask, "#8b4513", 0.28))
        for mask, color, opacity in overlays:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 10:
                    continue
                pts = c.reshape(-1, 2)
                pts_s = " ".join(f"{int(x)},{int(y)}" for x, y in pts)
                parts.append(f'<polygon points="{pts_s}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>')
        parts.append('</g>')
    parts.append('<g id="labels" font-size="10" text-anchor="middle" fill="#333333">')
    for room in rooms:
        pts = np.asarray(room["points_px"], dtype=np.float32)
        if len(pts) == 0:
            continue
        cx, cy = pts.mean(axis=0)
        parts.append(f'<text x="{cx:.1f}" y="{cy:.1f}">{_xml_text(room["type"])}</text>')
    parts.append('</g></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _extract_raw_predicted_rooms(
    pred: dict,
    grid_size: int,
    room_thr: float,
    corner_thr: float,
    min_room_area_px: float = 0.0,
) -> list[dict]:
    """
    Extract raw model room polygons directly from the corner head.

    This deliberately avoids:
    - auto layout selection
    - query mask fallback
    - polygon simplification
    - Manhattan snapping
    - rectangularisation
    - overlap suppression
    - duplicate-room removal
    - wall-mask-assisted cleanup
    """
    valid = torch.sigmoid(pred["room_valid_logits"])[0].detach().cpu().numpy()
    corner_valid = torch.sigmoid(pred["corner_valid_logits"])[0].detach().cpu().numpy()
    corners = pred["corner_xy"][0].detach().cpu().numpy()
    classes = torch.argmax(pred["room_class_logits"], dim=-1)[0].detach().cpu().numpy()

    rooms: list[dict] = []

    for ridx, score in enumerate(valid):
        if float(score) < float(room_thr):
            continue

        keep = corner_valid[ridx] >= float(corner_thr)
        pts = corners[ridx, keep]

        # Same fallback as your normal extraction, but still raw:
        # if too few corners pass the threshold, keep top 4 corners.
        if len(pts) < 3:
            topk = np.argsort(-corner_valid[ridx])[:4]
            topk = np.sort(topk)
            pts = corners[ridx, topk]

        pts_px = np.clip(
            np.round(pts * (grid_size - 1)),
            0,
            grid_size - 1
        ).astype(np.int32)

        if len(np.unique(pts_px, axis=0)) < 3:
            continue

        area = _polygon_area_px(pts_px)

        if area < float(min_room_area_px):
            continue

        cls = int(classes[ridx])
        rtype = ROOM_TYPES[cls] if 0 <= cls < len(ROOM_TYPES) else "other"

        rooms.append({
            "query_index": int(ridx),
            "type": rtype,
            "score": float(score),
            "points_px": pts_px.tolist(),
            "area_px": float(area),
            "source": "raw_corner_head_no_postprocess",
        })

    return rooms


def main() -> None:
    ap = argparse.ArgumentParser(description="Run trained RoomToFP v7-final inference on a Structured3D scene folder.")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
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
    ap.add_argument("--use-annotation-grid", action="store_true", help="Compatibility flag; annotation grid is already on by default.")
    ap.add_argument("--no-annotation-grid", action="store_true")
    ap.add_argument("--room-threshold", type=float, default=0.50)
    ap.add_argument("--corner-threshold", type=float, default=0.50)
    ap.add_argument("--mask-threshold", type=float, default=0.50)
    ap.add_argument("--min-room-area-px", type=float, default=80.0)
    ap.add_argument("--room-overlap-threshold", type=float, default=0.65)
    ap.add_argument("--simplify-eps-px", type=float, default=2.0)
    ap.add_argument("--snap-axis-threshold-px", type=float, default=6.0)
    ap.add_argument("--prefer-room-masks", action="store_true")
    ap.add_argument("--debug-mask-overlays", action="store_true")

    # v7.3 improvement: automatic evidence-based layout selection.
    # When enabled, inference tries several general candidate settings and picks
    # the full layout that best agrees with model/BEV evidence.
    ap.add_argument("--auto-select-layout", action="store_true")
    ap.add_argument("--layout-candidates", choices=["balanced", "strict", "recall"], default="balanced")
    ap.add_argument("--layout-score-room-mask", type=float, default=1.0)
    ap.add_argument("--layout-score-wall", type=float, default=0.8)
    ap.add_argument("--layout-score-opening", type=float, default=0.25)
    ap.add_argument("--layout-score-overlap-penalty", type=float, default=1.5)
    ap.add_argument("--layout-score-gap-penalty", type=float, default=0.4)
    ap.add_argument("--layout-score-complexity-penalty", type=float, default=0.03)
    ap.add_argument("--living-gap-fill", action="store_true", help="v7.3: conservatively fill small living-room gaps using room-mask evidence.")
    ap.add_argument("--living-gap-max-area-ratio", type=float, default=0.12)
    ap.add_argument("--living-gap-evidence-thr", type=float, default=0.45)
    ap.add_argument("--debug-layout-candidates", action="store_true", help="Save individual candidate metadata for v7.3 layout selection.")
    ap.add_argument("--debug-layout-candidate-previews", action="store_true", help="Save preview PNGs for each auto-layout candidate.")
    ap.add_argument("--wall-clean-close", type=int, default=3)
    ap.add_argument("--opening-clean-close", type=int, default=5)
    ap.add_argument("--min-opening-area-px", type=float, default=8.0)
    ap.add_argument("--min-opening-length-px", type=float, default=12.0)
    ap.add_argument("--opening-snap-distance-px", type=float, default=28.0)
    # v7.4 improvement: cleanup duplicate/implausible vector openings after snapping.
    ap.add_argument("--merge-opening-distance-px", type=float, default=18.0)
    ap.add_argument("--max-opening-length-px", type=float, default=80.0)
    ap.add_argument("--no-bev-opening-evidence", action="store_true", help="Disable BEV-assisted opening evidence.")
    ap.add_argument("--bev-opening-weight", type=float, default=1.0)
    ap.add_argument("--bev-opening-threshold", type=float, default=-1.0, help="Threshold for combined opening evidence. Negative uses --mask-threshold.")
    ap.add_argument("--include-furniture", action="store_true")
    ap.add_argument("--use-furniture-input", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Render raw model polygon predictions without cleanup, snapping, layout selection, mask fallback, opening extraction, or post-processing."
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(args.scene_dir)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    pts, colors, sem, sources = collect_scene_points(scene_dir, variants=variants, rgb_name=args.rgb_name, stride=args.stride)
    annotation_xy = None if args.no_annotation_grid else load_annotation_xy(scene_dir)
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
    obs = stack_roomtofp_observation(bev).astype(np.float32)
    if not args.use_furniture_input and obs.shape[-1] >= 5:
        obs[..., 4] = 0.0
    np.save(out / "roomtofp_observation.npy", obs)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    model_args = ckpt.get("args", {})
    ckpt_dims = _infer_model_dims_from_state_dict(state_dict)
    in_channels = _positive_int(model_args.get("in_channels", ckpt_dims.get("in_channels", obs.shape[-1])), ckpt_dims.get("in_channels", obs.shape[-1]))
    max_rooms = _positive_int(model_args.get("max_rooms", ckpt_dims.get("max_rooms", 20)), ckpt_dims.get("max_rooms", 20))
    max_corners = _positive_int(model_args.get("max_corners", ckpt_dims.get("max_corners", 16)), ckpt_dims.get("max_corners", 16))
    embed_dim = _positive_int(model_args.get("embed_dim", ckpt_dims.get("embed_dim", 256)), ckpt_dims.get("embed_dim", 256))
    room_classes = _positive_int(model_args.get("room_classes", ckpt_dims.get("room_classes", len(ROOM_TYPES))), ckpt_dims.get("room_classes", len(ROOM_TYPES)))
    predict_furniture = bool(model_args.get("predict_furniture", False)) or any(k.startswith("furniture_head") for k in state_dict.keys())

    print("[MODEL]")
    print(f"  in_channels={in_channels}")
    print(f"  max_rooms={max_rooms}")
    print(f"  max_corners={max_corners}")
    print(f"  embed_dim={embed_dim}")
    print(f"  room_classes={room_classes}")
    print(f"  predict_furniture={predict_furniture}")

    model = RoomToFPPolygonNet(
        in_channels=in_channels,
        max_rooms=max_rooms,
        max_corners=max_corners,
        embed_dim=embed_dim,
        room_classes=room_classes,
        dropout=float(model_args.get("dropout", 0.10)),
        predict_furniture=predict_furniture,
        base=int(model_args.get("base", 48)),
        mask_dim=int(model_args.get("mask_dim", 128)),
        decoder_layers=int(model_args.get("decoder_layers", 4)),
        corner_decoder_layers=int(model_args.get("corner_decoder_layers", 3)),
        num_heads=int(model_args.get("num_heads", 8)),
        transformer_encoder_layers=int(model_args.get("transformer_encoder_layers", 2)),
        deform_points=int(model_args.get("deform_points", 4)),
    ).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).to(args.device)
    with torch.no_grad():
        pred = model(x)

    wall_prob = torch.sigmoid(pred["wall_logits"])[0, 0].detach().cpu().numpy()
    opening_prob = torch.sigmoid(pred["opening_logits"])[0, 0].detach().cpu().numpy()
    wall_raw = wall_prob >= args.mask_threshold
    opening_raw = opening_prob >= args.mask_threshold

    # Improvement v7-final: BEV-assisted opening evidence.
    # Channel order is occupancy, wall, floor, opening, furniture, RGB.
    bev_opening = obs[..., 3] if obs.shape[-1] >= 4 else np.zeros_like(opening_prob)
    if args.no_bev_opening_evidence:
        opening_evidence = opening_prob
    else:
        opening_evidence = np.maximum(opening_prob, np.clip(args.bev_opening_weight * bev_opening, 0.0, 1.0))
    opening_thr = args.mask_threshold if args.bev_opening_threshold < 0 else args.bev_opening_threshold

    # v7.3 improvement:
    # Use the model's per-query room masks as dense room evidence for candidate
    # layout scoring and optional conservative living-room gap filling.
    if "room_mask_logits" in pred:
        room_union_prob = torch.sigmoid(pred["room_mask_logits"])[0].detach().cpu().numpy().max(axis=0)
    else:
        room_union_prob = np.zeros_like(wall_prob, dtype=np.float32)

    # v7.3 improvement:
    # Instead of committing to one room/corner threshold, optionally test several
    # general threshold/snap settings and select the full layout that scores best
    # against dense wall, room and opening evidence.
    if args.no_postprocess:
        rooms = _extract_raw_predicted_rooms(
            pred=pred,
            grid_size=args.grid_size,
            room_thr=args.room_threshold,
            corner_thr=args.corner_threshold,
            min_room_area_px=0.0,
        )

        layout_selection = {
            "score": None,
            "settings": {
                "room_thr": float(args.room_threshold),
                "corner_thr": float(args.corner_threshold),
                "postprocess": False,
                "source": "raw_corner_head",
            },
            "num_rooms": int(len(rooms)),
            "tried": [],
        }

    else:
        if args.auto_select_layout:
            rooms, layout_selection = _select_best_layout(
                pred=pred,
                grid_size=args.grid_size,
                mask_thr=args.mask_threshold,
                min_room_area_px=args.min_room_area_px,
                wall_prob=wall_prob,
                opening_evidence=opening_evidence,
                room_union_prob=room_union_prob,
                args=args,
            )
        else:
            rooms = _extract_predicted_rooms(
                pred,
                args.grid_size,
                args.room_threshold,
                args.corner_threshold,
                args.mask_threshold,
                min_room_area_px=args.min_room_area_px,
                overlap_thr=args.room_overlap_threshold,
                simplify_eps_px=args.simplify_eps_px,
                snap_axis_thr_px=args.snap_axis_threshold_px,
                prefer_room_masks=bool(args.prefer_room_masks),
                wall_prob=wall_prob,
            )

            layout_selection = {
                "score": None,
                "settings": {
                    "room_thr": float(args.room_threshold),
                    "corner_thr": float(args.corner_threshold),
                    "simplify": float(args.simplify_eps_px),
                    "snap": float(args.snap_axis_threshold_px),
                    "overlap": float(args.room_overlap_threshold),
                    "prefer_masks": bool(args.prefer_room_masks),
                    "postprocess": True,
                },
                "num_rooms": int(len(rooms)),
                "tried": [],
            }

    # v7.3 improvement:
    # Optional conservative gap filling. It is disabled unless explicitly requested
    # because it can improve living-room coverage but should be checked visually.
    if args.living_gap_fill and not args.no_postprocess:
        rooms = _fill_living_room_gaps(
            rooms=rooms,
            grid_size=args.grid_size,
            room_union_prob=room_union_prob,
            max_gap_area_ratio=args.living_gap_max_area_ratio,
            evidence_thr=args.living_gap_evidence_thr,
        )
        layout_selection["living_gap_fill_used"] = True
        layout_selection["num_rooms_after_gap_fill"] = int(len(rooms))
    else:
        layout_selection["living_gap_fill_used"] = False

    if args.debug_layout_candidates:
        (out / "layout_candidates.json").write_text(
            json.dumps(layout_selection, indent=2),
            encoding="utf-8",
        )

    if args.debug_layout_candidate_previews and args.auto_select_layout:
        cand_dir = out / "layout_candidate_previews"
        cand_dir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(layout_selection.get("tried", [])):
            st = rec.get("settings", {})
            cand_rooms = _extract_predicted_rooms(
                pred=pred,
                grid_size=args.grid_size,
                room_thr=float(st.get("room_thr", args.room_threshold)),
                corner_thr=float(st.get("corner_thr", args.corner_threshold)),
                mask_thr=args.mask_threshold,
                min_room_area_px=args.min_room_area_px,
                overlap_thr=float(st.get("overlap", args.room_overlap_threshold)),
                simplify_eps_px=float(st.get("simplify", args.simplify_eps_px)),
                snap_axis_thr_px=float(st.get("snap", args.snap_axis_threshold_px)),
                prefer_room_masks=bool(st.get("prefer_masks", False)),
                wall_prob=wall_prob,
            )
            cand_walls = _room_wall_segments(cand_rooms, min_length_px=8.0, dedup_quant_px=4.0)
            _write_vector_preview(
                cand_dir / f"candidate_{i:02d}_score_{float(rec.get('score', 0.0)):.4f}.png",
                cand_rooms,
                cand_walls,
                [],
                args.grid_size,
            )

    wall_clean = _clean_binary_mask(wall_prob, threshold=args.mask_threshold, close_kernel=args.wall_clean_close, open_kernel=1, min_area_px=24)
    opening_clean = _clean_binary_mask(opening_evidence, threshold=opening_thr, close_kernel=args.opening_clean_close, open_kernel=1, min_area_px=max(1, int(args.min_opening_area_px)))

    furniture = None
    if args.include_furniture and "furniture_logits" in pred:
        furniture_prob = torch.sigmoid(pred["furniture_logits"])[0, 0].detach().cpu().numpy()
        furniture = furniture_prob >= args.mask_threshold
        save_mask_png(furniture, out / "pred_furniture_mask.png")

    save_mask_png(wall_raw, out / "pred_wall_mask_raw.png")
    save_mask_png(opening_raw, out / "pred_opening_mask_raw.png")
    save_mask_png(wall_clean, out / "pred_wall_mask.png")
    save_mask_png(opening_clean, out / "pred_opening_mask.png")
    save_mask_png(opening_evidence >= opening_thr, out / "pred_opening_evidence_mask.png")

    if "boundary_logits" in pred:
        save_mask_png(torch.sigmoid(pred["boundary_logits"])[0, 0].detach().cpu().numpy() >= args.mask_threshold, out / "pred_boundary_mask.png")
    if "corner_heatmap_logits" in pred:
        save_mask_png(torch.sigmoid(pred["corner_heatmap_logits"])[0, 0].detach().cpu().numpy() >= args.mask_threshold, out / "pred_corner_heatmap.png")
    if "room_mask_logits" in pred:
        room_union = torch.sigmoid(pred["room_mask_logits"])[0].detach().cpu().numpy() >= args.mask_threshold
        save_mask_png(room_union.max(axis=0), out / "pred_room_union_mask.png")

    wall_segments = _room_wall_segments(rooms, min_length_px=8.0, dedup_quant_px=4.0)

    if args.no_postprocess:
        opening_segments = []
    else:
        opening_segments = _opening_components_to_segments(
            opening_clean,
            wall_segments,
            min_area_px=args.min_opening_area_px,
            min_len_px=args.min_opening_length_px,
            snap_dist_px=args.opening_snap_distance_px,
        )
        # v7.4 improvement:
        # Merge fragmented openings and remove implausibly long cyan segments after
        # snapping them to vector walls. This makes the preview closer to the sparse
        # Structured3D ground-truth opening style.
        opening_segments = _merge_nearby_openings(
            opening_segments,
            merge_dist_px=args.merge_opening_distance_px,
        )

        opening_segments = _filter_openings_by_length_and_wall(
            opening_segments,
            max_wall_fraction=0.55,
            max_abs_len_px=args.max_opening_length_px,
        )

    preview_name = "raw_no_postprocess_preview.png" if args.no_postprocess else "pred_floorplan_preview.png"
    svg_name = "raw_no_postprocess.svg" if args.no_postprocess else "pred_floorplan.svg"

    _write_vector_preview(
        out / preview_name,
        rooms,
        wall_segments,
        opening_segments,
        args.grid_size,
        debug_mask_overlays=False if args.no_postprocess else args.debug_mask_overlays,
        wall_mask=wall_clean,
        opening_mask=opening_clean,
        furniture_mask=furniture,
    )

    _write_vector_svg(
        out / svg_name,
        rooms,
        wall_segments,
        opening_segments,
        args.grid_size,
        debug_mask_overlays=False if args.no_postprocess else args.debug_mask_overlays,
        wall_mask=wall_clean,
        opening_mask=opening_clean,
        furniture_mask=furniture,
    )

    metadata = {
        "scene_dir": str(scene_dir),
        "checkpoint": str(Path(args.checkpoint)),
        "num_predicted_rooms": len(rooms),
        "predicted_rooms": rooms,
        "layout_selection": layout_selection,
        "auto_select_layout": bool(args.auto_select_layout),
        "layout_candidates_mode": str(args.layout_candidates),
        "living_gap_fill": bool(args.living_gap_fill),
        "num_wall_segments": len(wall_segments),
        "wall_segments": wall_segments,
        "num_opening_segments": len(opening_segments),
        "opening_segments": opening_segments,
        "grid_size": args.grid_size,
        "observation_channels": int(obs.shape[-1]),
        "predict_furniture": predict_furniture,
        "include_furniture_output": bool(args.include_furniture and furniture is not None),
        "use_furniture_input": bool(args.use_furniture_input),
        "prefer_room_masks": bool(args.prefer_room_masks),
        "clean_vector_render": not bool(args.no_postprocess),
        "debug_mask_overlays": bool(args.debug_mask_overlays),
        "annotation_grid_used": annotation_xy is not None,
        "bev_opening_evidence_used": not bool(args.no_bev_opening_evidence),
        "bev_opening_weight": float(args.bev_opening_weight),
        "opening_threshold_used": float(opening_thr),
        "model_dims": {"in_channels": in_channels, "max_rooms": max_rooms, "max_corners": max_corners, "embed_dim": embed_dim, "room_classes": room_classes},
        "outputs": {
            "preview_png": str(out / preview_name),
            "svg": str(out / svg_name),
            "wall_mask": str(out / "pred_wall_mask.png"),
            "opening_mask": str(out / "pred_opening_mask.png"),
            "opening_evidence_mask": str(out / "pred_opening_evidence_mask.png")
        },
        "sources": sources,
        "postprocess_disabled": bool(args.no_postprocess),
    }
    (out / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "num_predicted_rooms": len(rooms),
        "num_wall_segments": len(wall_segments),
        "num_opening_segments": len(opening_segments),
        "clean_vector_render": not bool(args.no_postprocess),
        "postprocess_disabled": bool(args.no_postprocess),
        "bev_opening_evidence_used": not bool(args.no_bev_opening_evidence),
        "auto_select_layout": bool(args.auto_select_layout),
        "layout_selection": layout_selection,
        "output_dir": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
