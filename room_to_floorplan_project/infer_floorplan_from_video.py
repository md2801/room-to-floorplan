from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from room_to_floorplan.geometry import make_bev, masks_to_svg, save_point_cloud_ply, stack_observation
from room_to_floorplan.model import FloorplanUNet
from room_to_floorplan.video import extract_video_frames, reconstruct_sparse_point_cloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--grid-size", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.output_dir)
    frames_dir = out / "frames"
    out.mkdir(parents=True, exist_ok=True)

    frame_paths = extract_video_frames(args.video, str(frames_dir), sample_every=args.sample_every, max_frames=args.max_frames)
    points, colors = reconstruct_sparse_point_cloud(frame_paths)
    if len(points) == 0:
        raise RuntimeError("Could not reconstruct enough 3D points from the video. Try slower camera motion and more overlap.")

    # Since real video has no semantics, use occupancy as the observation basis.
    fake_sem = np.zeros((len(points),), dtype=np.uint8)
    bev = make_bev(points, fake_sem, colors, grid_size=args.grid_size, z_clip=(-2.5, 2.5))
    obs = stack_observation(bev)
    obs_u8 = np.clip(obs[..., :3] * 255, 0, 255).astype(np.uint8)
    Image.fromarray(obs_u8).save(out / "observation.png")
    save_point_cloud_ply(points, colors, out / "point_cloud.ply")

    if args.weights:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FloorplanUNet().to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()
        x = torch.from_numpy(obs.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0].cpu().numpy()
        structure = (pred[0] > 0.5).astype(np.uint8) * 255
        furniture = (pred[1] > 0.5).astype(np.uint8) * 255
    else:
        structure = (bev["occupancy"] > 0.15).astype(np.uint8) * 255
        furniture = np.zeros_like(structure)

    Image.fromarray(structure).save(out / "pred_structure.png")
    Image.fromarray(furniture).save(out / "pred_furniture.png")

    combined = np.zeros((*structure.shape, 3), dtype=np.uint8)
    combined[..., 1] = structure
    combined[..., 0] = furniture
    Image.fromarray(combined).save(out / "combined.png")
    masks_to_svg(structure > 0, out / "floorplan.svg")

    summary = {
        "video": args.video,
        "frames_used": len(frame_paths),
        "num_points": int(len(points)),
        "weights": args.weights,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
