from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from room_to_floorplan.floorplan_svg import save_mask_png
from room_to_floorplan.roomtofp_polygon_model import RoomToFPPolygonNet
from train_roomtofp_stru3d import RoomToFPDataset


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


def _norm_img(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return (x * 255).astype(np.uint8)


def _panel(title: str, arr: np.ndarray, size: int = 256) -> Image.Image:
    if arr.ndim == 2:
        img = Image.fromarray(_norm_img(arr)).convert("RGB")
    else:
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).convert("RGB")

    img = img.resize((size, size))
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(img, (0, 24))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(0, 0, 0))
    return canvas


def _concat(panels: list[Image.Image]) -> Image.Image:
    w = sum(p.width for p in panels)
    h = max(p.height for p in panels)

    out = Image.new("RGB", (w, h), "white")
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Save side-by-side RoomToFP prediction visualisations.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", default="output/roomtofp_prediction_viz")
    ap.add_argument("--split", default="test", help="test/val/train/all")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--include-furniture", action="store_true", help="Include furniture panels only if the checkpoint has a furniture head. Off by default.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = RoomToFPDataset(args.dataset_root, split=None if args.split == "all" else args.split)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    model_args = ckpt.get("args", {})
    ckpt_dims = _infer_model_dims_from_state_dict(state_dict)

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
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )

    written = 0
    manifest = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(args.device, non_blocking=True)
            pred = model(x)

            for bi in range(x.shape[0]):
                if written >= args.num_samples:
                    break

                obs = batch["x"][bi].numpy()

                if obs.shape[0] >= 8:
                    rgb = obs[5:8].transpose(1, 2, 0)
                else:
                    rgb = np.repeat(obs[0:1], 3, axis=0).transpose(1, 2, 0)

                gt_wall = batch["wall_mask"][bi, 0].numpy()
                gt_open = batch["opening_mask"][bi, 0].numpy()
                gt_boundary = batch["boundary_mask"][bi, 0].numpy() if "boundary_mask" in batch else np.maximum(gt_wall, gt_open)
                gt_room_union = batch["room_mask"][bi].numpy().max(axis=0) if "room_mask" in batch else np.zeros_like(gt_wall)

                pr_wall = (
                    torch.sigmoid(pred["wall_logits"])[bi, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    >= args.threshold
                ).astype(np.float32)

                pr_open = (
                    torch.sigmoid(pred["opening_logits"])[bi, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    >= args.threshold
                ).astype(np.float32)

                # Improvement v3: visualize room-query masks and dense boundary targets so failures are easier to diagnose.
                pred_room_union = np.zeros_like(gt_wall)
                if "room_mask_logits" in pred:
                    pred_room_union = (torch.sigmoid(pred["room_mask_logits"])[bi].detach().cpu().numpy() >= args.threshold).astype(np.float32).max(axis=0)
                pred_boundary = np.zeros_like(gt_wall)
                if "boundary_logits" in pred:
                    pred_boundary = (torch.sigmoid(pred["boundary_logits"])[bi, 0].detach().cpu().numpy() >= args.threshold).astype(np.float32)

                panels = [
                    _panel("input rgb", rgb),
                    _panel("obs wall", obs[1]),
                    _panel("GT wall", gt_wall),
                    _panel("Pred wall", pr_wall),
                    _panel("GT opening", gt_open),
                    _panel("Pred opening", pr_open),
                    _panel("GT rooms", gt_room_union),
                    _panel("Pred rooms", pred_room_union),
                    _panel("GT boundary", gt_boundary),
                    _panel("Pred boundary", pred_boundary),
                ]

                pr_furn = None
                if args.include_furniture and "furniture_logits" in pred:
                    gt_furn = batch["furniture_mask"][bi, 0].numpy()
                    pr_furn = (
                        torch.sigmoid(pred["furniture_logits"])[bi, 0]
                        .detach()
                        .cpu()
                        .numpy()
                        >= args.threshold
                    ).astype(np.float32)
                    panels.extend([
                        _panel("GT furniture", gt_furn),
                        _panel("Pred furniture", pr_furn),
                    ])

                img = _concat(panels)

                path = out / f"sample_{written:04d}.png"
                img.save(path)

                save_mask_png(pr_wall, out / f"sample_{written:04d}_pred_wall.png")
                save_mask_png(pr_open, out / f"sample_{written:04d}_pred_opening.png")

                if pr_furn is not None:
                    save_mask_png(pr_furn, out / f"sample_{written:04d}_pred_furniture.png")

                source_idx = batch_idx * args.batch_size + bi
                source_file = str(ds.files[source_idx]) if hasattr(ds, "files") and source_idx < len(ds.files) else "unknown"
                manifest.append({"index": written, "source_file": source_file, "image": str(path)})
                written += 1

            if written >= args.num_samples:
                break

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {written} visualisations to {out}")


if __name__ == "__main__":
    main()