from __future__ import annotations

"""
RoomToFP v7-final clean-vector inference.

Final inference improvements:
- Polygon-first room output by default.
- Annotation-grid crop stabilisation ON by default for Structured3D, matching training preparation.
- Clean vector walls from predicted polygon edges.
- BEV-assisted opening extraction: combines model opening probability with BEV opening input channel.
- Opening blobs are converted into snapped vector segments on predicted wall edges.
- Raw masks are saved for debugging but not overlaid on final preview unless --debug-mask-overlays is used.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from room_to_floorplan.floorplan_svg import save_mask_png
from room_to_floorplan.roomtofp_polygon_model import RoomToFPPolygonNet
from room_to_floorplan.roomtofp_scene_utils import (
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


def _postprocess_rooms(rooms: list[dict], grid_size: int, min_area_px: float, overlap_thr: float, simplify_eps_px: float, snap_axis_thr_px: float) -> list[dict]:
    cleaned: list[dict] = []
    for room in sorted(rooms, key=lambda r: float(r.get("score", 0.0)), reverse=True):
        pts = np.asarray(room["points_px"], dtype=np.int32)
        pts = np.unique(pts, axis=0)
        if len(pts) < 3:
            continue
        pts = _sort_points_clockwise(pts)
        pts = _simplify_polygon_points(pts, simplify_eps_px)
        pts = _snap_axis_aligned(pts, snap_axis_thr_px)
        pts = np.clip(pts, 0, grid_size - 1).astype(np.int32)
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
    return accepted


def _extract_predicted_rooms(pred: dict, grid_size: int, room_thr: float, corner_thr: float, mask_thr: float, min_room_area_px: float, overlap_thr: float, simplify_eps_px: float, snap_axis_thr_px: float, prefer_room_masks: bool) -> list[dict]:
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
        if len(pts) < 3:
            pts = corners[ridx, :4]
        pts_px = np.clip(np.round(pts * (grid_size - 1)), 0, grid_size - 1).astype(np.int32)
        source = "corner_head"
        if prefer_room_masks and room_masks is not None:
            mask_poly = _largest_contour_polygon((room_masks[ridx] >= mask_thr).astype(np.uint8), simplify_eps_px)
            if mask_poly is not None and _polygon_area_px(mask_poly) >= min_room_area_px:
                pts_px = mask_poly
                source = "query_mask"
        if len(np.unique(pts_px, axis=0)) < 3:
            continue
        cls = int(classes[ridx])
        rtype = ROOM_TYPES[cls] if 0 <= cls < len(ROOM_TYPES) else "other"
        rooms.append({"query_index": int(ridx), "type": rtype, "score": float(score), "points_px": pts_px.tolist(), "source": source})

    return _postprocess_rooms(rooms, grid_size, min_room_area_px, overlap_thr, simplify_eps_px, snap_axis_thr_px)


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
    ap.add_argument("--wall-clean-close", type=int, default=3)
    ap.add_argument("--opening-clean-close", type=int, default=5)
    ap.add_argument("--min-opening-area-px", type=float, default=8.0)
    ap.add_argument("--min-opening-length-px", type=float, default=12.0)
    ap.add_argument("--opening-snap-distance-px", type=float, default=28.0)
    ap.add_argument("--no-bev-opening-evidence", action="store_true", help="Disable BEV-assisted opening evidence.")
    ap.add_argument("--bev-opening-weight", type=float, default=1.0)
    ap.add_argument("--bev-opening-threshold", type=float, default=-1.0, help="Threshold for combined opening evidence. Negative uses --mask-threshold.")
    ap.add_argument("--include-furniture", action="store_true")
    ap.add_argument("--use-furniture-input", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    )

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
    opening_segments = _opening_components_to_segments(opening_clean, wall_segments, min_area_px=args.min_opening_area_px, min_len_px=args.min_opening_length_px, snap_dist_px=args.opening_snap_distance_px)

    _write_vector_preview(out / "pred_floorplan_preview.png", rooms, wall_segments, opening_segments, args.grid_size, debug_mask_overlays=args.debug_mask_overlays, wall_mask=wall_clean, opening_mask=opening_clean, furniture_mask=furniture)
    _write_vector_svg(out / "pred_floorplan.svg", rooms, wall_segments, opening_segments, args.grid_size, debug_mask_overlays=args.debug_mask_overlays, wall_mask=wall_clean, opening_mask=opening_clean, furniture_mask=furniture)

    metadata = {
        "scene_dir": str(scene_dir),
        "checkpoint": str(Path(args.checkpoint)),
        "num_predicted_rooms": len(rooms),
        "predicted_rooms": rooms,
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
        "clean_vector_render": True,
        "debug_mask_overlays": bool(args.debug_mask_overlays),
        "annotation_grid_used": annotation_xy is not None,
        "bev_opening_evidence_used": not bool(args.no_bev_opening_evidence),
        "bev_opening_weight": float(args.bev_opening_weight),
        "opening_threshold_used": float(opening_thr),
        "model_dims": {"in_channels": in_channels, "max_rooms": max_rooms, "max_corners": max_corners, "embed_dim": embed_dim, "room_classes": room_classes},
        "outputs": {"preview_png": str(out / "pred_floorplan_preview.png"), "svg": str(out / "pred_floorplan.svg"), "wall_mask": str(out / "pred_wall_mask.png"), "opening_mask": str(out / "pred_opening_mask.png"), "opening_evidence_mask": str(out / "pred_opening_evidence_mask.png")},
        "sources": sources,
    }
    (out / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"num_predicted_rooms": len(rooms), "num_wall_segments": len(wall_segments), "num_opening_segments": len(opening_segments), "clean_vector_render": True, "bev_opening_evidence_used": not bool(args.no_bev_opening_evidence), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
