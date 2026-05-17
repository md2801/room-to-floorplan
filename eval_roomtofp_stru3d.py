from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from room_to_floorplan.roomtofp_polygon_model import RoomToFPPolygonNet
from train_roomtofp_stru3d import LossConfig, RoomToFPDataset, run_epoch


def _positive_int(value, fallback: int) -> int:
    """
    Return value as int only if it is positive.
    Some older checkpoints may store 0 for max_rooms/max_corners,
    so this prevents building zero-sized model layers.
    """
    try:
        value = int(value)
        return value if value > 0 else int(fallback)
    except Exception:
        return int(fallback)


def _infer_model_dims_from_state_dict(state_dict: dict) -> dict:
    """
    Infer important model dimensions directly from checkpoint tensors.
    This is safer than trusting saved args if old args contain invalid zeros.
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

    # ConvEncoder first conv: [out_channels, in_channels, k, k]
    if "encoder.stem.0.weight" in state_dict:
        dims["in_channels"] = int(state_dict["encoder.stem.0.weight"].shape[1])

    return dims


def main():
    ap = argparse.ArgumentParser(description="Evaluate RoomToFP polygon predictor.")
    ap.add_argument("--dataset-root", default="data/roomtofp_polygon")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--split", default="test", help="Use test/val/train split file when present. Use 'all' for every sample.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="output/roomtofp_eval.json")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    # Load checkpoint on CPU first for safer inspection, then move model to target device.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    model_args = ckpt.get("args", {})
    ckpt_dims = _infer_model_dims_from_state_dict(state_dict)

    split = None if args.split == "all" else args.split
    ds = RoomToFPDataset(args.dataset_root, split=split)

    in_channels = _positive_int(
        model_args.get("in_channels", ckpt_dims.get("in_channels", ds.in_channels)),
        ckpt_dims.get("in_channels", ds.in_channels),
    )

    max_rooms = _positive_int(
        model_args.get("max_rooms", ckpt_dims.get("max_rooms", ds.max_rooms)),
        ckpt_dims.get("max_rooms", ds.max_rooms),
    )

    max_corners = _positive_int(
        model_args.get("max_corners", ckpt_dims.get("max_corners", ds.max_corners)),
        ckpt_dims.get("max_corners", ds.max_corners),
    )

    embed_dim = _positive_int(
        model_args.get("embed_dim", ckpt_dims.get("embed_dim", 256)),
        ckpt_dims.get("embed_dim", 256),
    )

    room_classes = _positive_int(
        model_args.get("room_classes", ckpt_dims.get("room_classes", 12)),
        ckpt_dims.get("room_classes", 12),
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
        # Improvement v3: pass architecture options saved by the stronger FPN/query-mask model.
        base=int(model_args.get("base", 48)),
        mask_dim=int(model_args.get("mask_dim", 128)),
        decoder_layers=int(model_args.get("decoder_layers", 4)),
        corner_decoder_layers=int(model_args.get("corner_decoder_layers", 2)),  # Improvement v4: restore corner-query decoder depth.
        num_heads=int(model_args.get("num_heads", 8)),
    ).to(args.device)

    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    cfg = LossConfig(
        use_furniture=bool(model_args.get("use_furniture_loss", False)) and predict_furniture,
        aux_furniture=float(model_args.get("aux_furniture", 0.0)),
        aux_wall=float(model_args.get("aux_wall", 0.8)),
        aux_opening=float(model_args.get("aux_opening", 0.7)),
        aux_boundary=float(model_args.get("aux_boundary", 0.35)),
        aux_corner_heatmap=float(model_args.get("aux_corner_heatmap", 0.25)),
        aux_room_mask=float(model_args.get("aux_room_mask", 1.0)),
        coord_weight=float(model_args.get("coord_weight", 5.0)),
        polygon_edge_weight=float(model_args.get("polygon_edge_weight", 0.75)),
        polygon_angle_weight=float(model_args.get("polygon_angle_weight", 0.20)),  # Improvement v4: keep eval loss config aligned with training.
        polygon_area_weight=float(model_args.get("polygon_area_weight", 0.35)),
        room_overlap_weight=float(model_args.get("room_overlap_weight", 0.10)),
        boundary_weight=float(model_args.get("boundary_weight", 0.15)),
        room_pos_weight_max=float(model_args.get("room_pos_weight_max", 8.0)),
        dice_weight=float(model_args.get("dice_weight", 1.0)),
        focal_gamma=float(model_args.get("focal_gamma", 2.0)),
        mask_loss=str(model_args.get("mask_loss", "focal_dice")),
        hungarian=not bool(model_args.get("no_hungarian", False)),
        cyclic=not bool(model_args.get("no_cyclic", False)),
        match_class_weight=float(model_args.get("match_class_weight", 0.25)),
        match_room_conf_weight=float(model_args.get("match_room_conf_weight", 0.15)),
    )

    metrics = run_epoch(model, loader, args.device, None, cfg)
    metrics["split"] = args.split
    metrics["num_samples"] = len(ds)
    metrics["checkpoint_epoch"] = ckpt.get("epoch")
    metrics["predict_furniture"] = predict_furniture
    metrics["use_furniture_loss"] = bool(model_args.get("use_furniture_loss", False)) and predict_furniture

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()