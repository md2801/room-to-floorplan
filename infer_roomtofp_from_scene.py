from __future__ import annotations

import argparse
import json
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

"""
Inference Flow:
scene folder -> build BEV obs -> load trained model -> model predicts room/walls/openings -> extract predicted room polygons -> save masks/previews/SVG/metadata
"""

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
    """
    Infer model dimensions from checkpoint weights.

    This avoids crashes when older checkpoint args accidentally store
    max_rooms=0 or max_corners=0.
    """
    dims = {}

    if "room_queries.weight" in state_dict:
        dims["max_rooms"] = int(state_dict["room_queries.weight"].shape[0])
        dims["embed_dim"] = int(state_dict["room_queries.weight"].shape[1])

    # Improvement v4: v4 checkpoints store max_corners in learned corner-query tokens.
    # Keep the old corner_head fallback only so older v3 checkpoints can still be inspected.
    if "corner_queries.weight" in state_dict:
        dims["max_corners"] = int(state_dict["corner_queries.weight"].shape[0])
    elif "corner_head.4.bias" in state_dict:
        dims["max_corners"] = int(state_dict["corner_head.4.bias"].numel() // 2)

    if "class_head.1.bias" in state_dict:
        dims["room_classes"] = int(state_dict["class_head.1.bias"].numel())

    if "encoder.stem.0.weight" in state_dict:
        dims["in_channels"] = int(state_dict["encoder.stem.0.weight"].shape[1])

    return dims


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """
    Converts hexcode to RGB tuple.
    """
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _polygon_area_px(points: np.ndarray) -> float:
    """Return absolute polygon area in pixels."""
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def _sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    """
    Sort predicted polygon points around their centroid.

    The model can sometimes output valid-looking points in a poor order. Sorting gives a more drawable polygon for inference output.
    This is post-processing only; training still uses the model's direct predicted order.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= 2:
        return pts.astype(np.int32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype(np.int32)


def _simplify_polygon_points(points: np.ndarray, eps_px: float) -> np.ndarray:
    """Simplify a predicted polygon with OpenCV's Douglas-Peucker approximation."""
    pts = np.asarray(points, dtype=np.int32)
    if len(pts) <= 3 or eps_px <= 0:
        return pts
    approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon=float(eps_px), closed=True).reshape(-1, 2)
    return approx.astype(np.int32) if len(approx) >= 3 else pts


def _snap_axis_aligned(points: np.ndarray, threshold_px: float) -> np.ndarray:
    """
    Snap near-horizontal/near-vertical polygon edges.

    Floorplans are usually Manhattan-like. This light post-process only changes edges that are already close to horizontal/vertical,
    so it improves visual cleanliness without forcing every edge aggressively.
    """
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
    # Improvement v3: use the new query room mask as a mask-guided polygon fallback/refinement, not only raw corner regression.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    pts = cv2.approxPolyDP(c, epsilon=max(1.0, float(simplify_eps_px)), closed=True).reshape(-1, 2)
    if len(pts) < 3:
        hull = cv2.convexHull(c).reshape(-1, 2)
        pts = hull
    return pts.astype(np.int32) if len(pts) >= 3 else None


def _room_mask(points: np.ndarray, grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


def _postprocess_rooms(
    rooms: list[dict],
    grid_size: int,
    min_area_px: float,
    overlap_thr: float,
    simplify_eps_px: float,
    snap_axis_thr_px: float,
) -> list[dict]:
    """
    Clean raw room-query polygons for inference display.

    This removes tiny/duplicate rooms, sorts/simplifies corners, lightly snaps near-axis-aligned edges, and suppresses lower-confidence
    rooms that heavily overlap an already accepted room.
    """
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


def _extract_predicted_rooms(
    pred: dict,
    grid_size: int,
    room_thr: float,
    corner_thr: float,
    mask_thr: float,
    min_room_area_px: float = 80.0,
    overlap_thr: float = 0.65,
    simplify_eps_px: float = 2.0,
    snap_axis_thr_px: float = 6.0,
    prefer_room_masks: bool = True,
) -> list[dict]:
    """
    Converts model raw polygon predictions into a clean Python list of predicted rooms.
    """
    valid = torch.sigmoid(pred["room_valid_logits"])[0].detach().cpu().numpy()
    corner_valid = torch.sigmoid(pred["corner_valid_logits"])[0].detach().cpu().numpy()
    corners = pred["corner_xy"][0].detach().cpu().numpy()
    classes = torch.argmax(pred["room_class_logits"], dim=-1)[0].detach().cpu().numpy()
    room_masks = None
    if "room_mask_logits" in pred:
        room_masks = torch.sigmoid(pred["room_mask_logits"])[0].detach().cpu().numpy()

    rooms: list[dict] = []

    for ridx, score in enumerate(valid):
        if float(score) < room_thr:
            continue

        keep = corner_valid[ridx] >= corner_thr
        pts = corners[ridx, keep]

        if len(pts) < 3:
            # Fallback: keep first 4 corners if the corner-valid head is underconfident.
            pts = corners[ridx, :4]

        pts_px = np.clip(np.round(pts * (grid_size - 1)), 0, grid_size - 1).astype(np.int32)
        source = "corner_head"

        if prefer_room_masks and room_masks is not None:
            mask = (room_masks[ridx] >= mask_thr).astype(np.uint8)
            mask_poly = _largest_contour_polygon(mask, simplify_eps_px=simplify_eps_px)
            if mask_poly is not None and _polygon_area_px(mask_poly) >= min_room_area_px:
                pts_px = mask_poly
                source = "query_mask"

        if len(np.unique(pts_px, axis=0)) < 3:
            continue

        cls = int(classes[ridx])
        rtype = ROOM_TYPES[cls] if 0 <= cls < len(ROOM_TYPES) else "other"

        rooms.append({
            "query_index": int(ridx),
            "type": rtype,
            "score": float(score),
            "points_px": pts_px.tolist(),
            "source": source,
        })

    return _postprocess_rooms(
        rooms,
        grid_size=grid_size,
        min_area_px=min_room_area_px,
        overlap_thr=overlap_thr,
        simplify_eps_px=simplify_eps_px,
        snap_axis_thr_px=snap_axis_thr_px,
    )


def _write_pred_svg(
    path: Path,
    rooms: list[dict],
    wall_mask: np.ndarray,
    opening_mask: np.ndarray,
    furniture_mask: np.ndarray | None = None,
) -> None:
    h, w = wall_mask.shape

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for room in rooms:
        pts_s = " ".join(f"{x},{y}" for x, y in room["points_px"])
        color = ROOM_COLORS.get(room["type"], ROOM_COLORS["undefined"])

        parts.append(
            f'<polygon points="{pts_s}" fill="{color}" fill-opacity="0.30" '
            f'stroke="#111111" stroke-width="3"/>'
        )

        pts = np.asarray(room["points_px"], dtype=np.float32)
        cx, cy = pts.mean(axis=0)

        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="10" '
            f'text-anchor="middle" fill="#333333">{room["type"]}</text>'
        )

    # Dense heads are added as translucent overlays for practical model debugging.
    overlays = [(wall_mask, "#111111", 0.20), (opening_mask, "#1f77b4", 0.35)]

    if furniture_mask is not None:
        overlays.append((furniture_mask, "#8b4513", 0.28))

    for mask, color, opacity in overlays:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for c in contours:
            if cv2.contourArea(c) < 10:
                continue

            pts = c.reshape(-1, 2)
            pts_s = " ".join(f"{int(x)},{int(y)}" for x, y in pts)

            parts.append(
                f'<polygon points="{pts_s}" fill="{color}" '
                f'fill-opacity="{opacity}" stroke="none"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_pred_preview(
    path: Path,
    rooms: list[dict],
    wall_mask: np.ndarray,
    opening_mask: np.ndarray,
    furniture_mask: np.ndarray | None = None,
) -> None:
    h, w = wall_mask.shape

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    for room in rooms:
        pts = [tuple(p) for p in room["points_px"]]
        color = _hex_to_rgb(ROOM_COLORS.get(room["type"], ROOM_COLORS["undefined"]))

        draw.polygon(pts, fill=(*color, 70), outline=(0, 0, 0, 220))

        arr = np.asarray(room["points_px"], dtype=np.float32)
        cx, cy = arr.mean(axis=0)

        draw.text(
            (float(cx), float(cy)),
            room["type"],
            fill=(45, 45, 45, 255),
            anchor="mm",
        )

    # Raster overlays.
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov = np.array(overlay)

    ov[wall_mask.astype(bool)] = [0, 0, 0, 65]
    ov[opening_mask.astype(bool)] = [31, 119, 180, 115]

    if furniture_mask is not None:
        ov[furniture_mask.astype(bool)] = [139, 69, 19, 95]

    img = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(ov)).convert("RGB")
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run trained RoomToFP inference on a Structured3D scene folder.")

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

    ap.add_argument(
        "--use-annotation-grid",
        action="store_true",
        help="Use annotation junctions only to stabilise the BEV crop during Structured3D evaluation.",
    )

    ap.add_argument("--room-threshold", type=float, default=0.50)
    ap.add_argument("--corner-threshold", type=float, default=0.50)
    ap.add_argument("--mask-threshold", type=float, default=0.50)
    ap.add_argument("--min-room-area-px", type=float, default=80.0, help="Remove predicted room polygons smaller than this area.")
    ap.add_argument("--room-overlap-threshold", type=float, default=0.65, help="Suppress lower-confidence rooms that overlap an accepted room by more than this ratio.")
    ap.add_argument("--simplify-eps-px", type=float, default=2.0, help="Polygon simplification epsilon used only for inference display/export.")
    ap.add_argument("--snap-axis-threshold-px", type=float, default=6.0, help="Snap nearly horizontal/vertical room edges within this many pixels.")
    ap.add_argument("--no-prefer-room-masks", action="store_true", help="Improvement v3 debug option: use raw polygon corners instead of query-mask guided polygons.")

    ap.add_argument(
        "--include-furniture",
        action="store_true",
        help="Write/draw furniture prediction if the checkpoint contains a furniture head. Off by default.",
    )
    ap.add_argument(
        "--use-furniture-input",
        action="store_true",
        help="Keep the furniture BEV input channel. Off by default for floorplan-only inference.",
    )

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    pts, colors, sem, sources = collect_scene_points(
        Path(args.scene_dir),
        variants=variants,
        rgb_name=args.rgb_name,
        stride=args.stride,
    )

    annotation_xy = load_annotation_xy(Path(args.scene_dir)) if args.use_annotation_grid else None

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

    in_channels = _positive_int(
        model_args.get("in_channels", ckpt_dims.get("in_channels", obs.shape[-1])),
        ckpt_dims.get("in_channels", obs.shape[-1]),
    )

    max_rooms = _positive_int(
        model_args.get("max_rooms", ckpt_dims.get("max_rooms", 20)),
        ckpt_dims.get("max_rooms", 20),
    )

    max_corners = _positive_int(
        model_args.get("max_corners", ckpt_dims.get("max_corners", 40)),
        ckpt_dims.get("max_corners", 40),
    )

    embed_dim = _positive_int(
        model_args.get("embed_dim", ckpt_dims.get("embed_dim", 256)),
        ckpt_dims.get("embed_dim", 256),
    )

    room_classes = _positive_int(
        model_args.get("room_classes", ckpt_dims.get("room_classes", len(ROOM_TYPES))),
        ckpt_dims.get("room_classes", len(ROOM_TYPES)),
    )

    predict_furniture = bool(model_args.get("predict_furniture", False)) or any(
        k.startswith("furniture_head") for k in state_dict.keys()
    )

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
        # Improvement v3: restore stronger FPN/query-mask model dimensions from checkpoint args.
        base=int(model_args.get("base", 48)),
        mask_dim=int(model_args.get("mask_dim", 128)),
        decoder_layers=int(model_args.get("decoder_layers", 4)),
        corner_decoder_layers=int(model_args.get("corner_decoder_layers", 2)),  # Improvement v4: restore corner-query decoder depth.
        num_heads=int(model_args.get("num_heads", 8)),
    ).to(args.device)

    model.load_state_dict(state_dict)
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
        prefer_room_masks=not args.no_prefer_room_masks,
    )

    wall = (
        torch.sigmoid(pred["wall_logits"])[0, 0]
        .detach()
        .cpu()
        .numpy()
        >= args.mask_threshold
    )

    opening = (
        torch.sigmoid(pred["opening_logits"])[0, 0]
        .detach()
        .cpu()
        .numpy()
        >= args.mask_threshold
    )

    furniture = None

    if args.include_furniture and "furniture_logits" in pred:
        furniture = (
            torch.sigmoid(pred["furniture_logits"])[0, 0]
            .detach()
            .cpu()
            .numpy()
            >= args.mask_threshold
        )

    save_mask_png(wall, out / "pred_wall_mask.png")
    save_mask_png(opening, out / "pred_opening_mask.png")

    if "boundary_logits" in pred:
        boundary = torch.sigmoid(pred["boundary_logits"])[0, 0].detach().cpu().numpy() >= args.mask_threshold
        save_mask_png(boundary, out / "pred_boundary_mask.png")
    if "corner_heatmap_logits" in pred:
        corners = torch.sigmoid(pred["corner_heatmap_logits"])[0, 0].detach().cpu().numpy() >= args.mask_threshold
        save_mask_png(corners, out / "pred_corner_heatmap.png")
    if "room_mask_logits" in pred:
        room_union = torch.sigmoid(pred["room_mask_logits"])[0].detach().cpu().numpy() >= args.mask_threshold
        save_mask_png(room_union.max(axis=0), out / "pred_room_union_mask.png")

    if furniture is not None:
        save_mask_png(furniture, out / "pred_furniture_mask.png")

    _write_pred_preview(out / "pred_floorplan_preview.png", rooms, wall, opening, furniture)
    _write_pred_svg(out / "pred_floorplan.svg", rooms, wall, opening, furniture)

    metadata = {
        "scene_dir": str(Path(args.scene_dir)),
        "checkpoint": str(Path(args.checkpoint)),
        "num_predicted_rooms": len(rooms),
        "predicted_rooms": rooms,
        "grid_size": args.grid_size,
        "observation_channels": int(obs.shape[-1]),
        "predict_furniture": predict_furniture,
        "include_furniture_output": bool(args.include_furniture and furniture is not None),
        "use_furniture_input": bool(args.use_furniture_input),
        "prefer_room_masks": bool(not args.no_prefer_room_masks),
        "model_dims": {
            "in_channels": in_channels,
            "max_rooms": max_rooms,
            "max_corners": max_corners,
            "embed_dim": embed_dim,
            "room_classes": room_classes,
        },
        "outputs": {
            "preview_png": str(out / "pred_floorplan_preview.png"),
            "svg": str(out / "pred_floorplan.svg"),
            "wall_mask": str(out / "pred_wall_mask.png"),
            "opening_mask": str(out / "pred_opening_mask.png"),
        },
        "sources": sources,
    }

    (out / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({
        "num_predicted_rooms": len(rooms),
        "output_dir": str(out),
        "model_dims": metadata["model_dims"],
    }, indent=2))


if __name__ == "__main__":
    main()
