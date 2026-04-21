from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from room_to_floorplan.structured3d import Structured3DZip
from room_to_floorplan.geometry import (
    make_bev,
    panorama_depth_to_world,
    perspective_depth_to_world,
    save_bev_preview,
    stack_observation,
    target_from_bev,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perspective-zip", type=str, default=None)
    ap.add_argument("--panorama-zip", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    print("=" * 80)
    print("BUILD TRAINING DATA")
    print("=" * 80)
    print(f"Perspective zip : {args.perspective_zip}")
    print(f"Panorama zip    : {args.panorama_zip}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Grid size       : {args.grid_size}")
    print(f"Stride          : {args.stride}")
    print("=" * 80)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory ready: {out.resolve()}")

    samples = []
    idx = 0
    pano_count = 0
    persp_count = 0
    pano_errors = 0
    persp_errors = 0

    if args.panorama_zip:
        print("\n" + "-" * 80)
        print("[PANORAMA] Opening panorama zip...")
        ds = Structured3DZip(args.panorama_zip)
        print(f"[PANORAMA] has_panorama(): {ds.has_panorama()}")
        print(f"[PANORAMA] has_perspective(): {ds.has_perspective()}")

        rooms = ds.list_rooms()
        print(f"[PANORAMA] Rooms found: {len(rooms)}")
        if rooms:
            print(f"[PANORAMA] First few rooms: {rooms[:10]}")

        found_any = False

        for view_idx, view in enumerate(ds.iter_panorama_views(), start=1):
            found_any = True
            print("\n" + "-" * 80)
            print(f"[PANORAMA] Processing view #{view_idx}")
            print(f"[PANORAMA] room_id: {view.room_id}")
            print(f"[PANORAMA] rgb shape: {view.rgb.shape}, dtype: {view.rgb.dtype}")
            print(f"[PANORAMA] depth shape: {view.depth_mm.shape}, dtype: {view.depth_mm.dtype}")
            print(f"[PANORAMA] semantic shape: {view.semantic.shape}, dtype: {view.semantic.dtype}")
            print(f"[PANORAMA] camera_xyz_mm shape: {np.shape(view.camera_xyz_mm)}")

            try:
                pts, colors, sem = panorama_depth_to_world(
                    view.depth_mm, view.rgb, view.semantic, view.camera_xyz_mm, stride=args.stride
                )
                print(f"[PANORAMA] points shape: {pts.shape}, dtype: {pts.dtype}")
                print(f"[PANORAMA] colors shape: {colors.shape}, dtype: {colors.dtype}")
                print(f"[PANORAMA] semantic vector shape: {sem.shape}, dtype: {sem.dtype}")
                if sem.size > 0:
                    uniq = np.unique(sem)
                    print(f"[PANORAMA] unique semantic IDs count: {len(uniq)}")
                    print(f"[PANORAMA] first few semantic IDs: {uniq[:20]}")

                bev = make_bev(pts, sem, colors, grid_size=args.grid_size)
                print(f"[PANORAMA] BEV occupancy shape : {bev['occupancy'].shape}")
                print(f"[PANORAMA] BEV structure shape : {bev['structure'].shape}")
                print(f"[PANORAMA] BEV furniture shape : {bev['furniture'].shape}")
                if bev["rgb"] is not None:
                    print(f"[PANORAMA] BEV rgb shape       : {bev['rgb'].shape}")
                else:
                    print("[PANORAMA] BEV rgb shape       : None")

                obs = stack_observation(bev)
                target = target_from_bev(bev)
                print(f"[PANORAMA] obs shape: {obs.shape}, dtype: {obs.dtype}")
                print(f"[PANORAMA] target shape: {target.shape}, dtype: {target.dtype}")

                stem = f"pano_room_{view.room_id}"
                obs_path = out / f"{stem}_obs.npy"
                target_path = out / f"{stem}_target.npy"
                preview_path = out / f"{stem}_preview.png"

                print(f"[DEBUG] obs min/max: {obs.min():.4f} / {obs.max():.4f}")
                print(f"[DEBUG] target unique values: {np.unique(target)}")

                np.save(obs_path, obs)
                np.save(target_path, target)
                save_bev_preview(obs, target, preview_path)

                print(f"[PANORAMA] Saved obs    : {obs_path}")
                print(f"[PANORAMA] Saved target : {target_path}")
                print(f"[PANORAMA] Saved preview: {preview_path}")

                samples.append({
                    "id": stem,
                    "source": "panorama",
                    "room_id": view.room_id,
                })
                idx += 1
                pano_count += 1

            except Exception as e:
                pano_errors += 1
                print(f"[PANORAMA][ERROR] room_id={view.room_id} failed with: {type(e).__name__}: {e}")

        if not found_any:
            print("[PANORAMA][WARN] iter_panorama_views() yielded 0 views.")

    if args.perspective_zip:
        print("\n" + "-" * 80)
        print("[PERSPECTIVE] Opening perspective zip...")
        ds = Structured3DZip(args.perspective_zip)
        print(f"[PERSPECTIVE] has_panorama(): {ds.has_panorama()}")
        print(f"[PERSPECTIVE] has_perspective(): {ds.has_perspective()}")

        rooms = ds.list_rooms()
        print(f"[PERSPECTIVE] Rooms found: {len(rooms)}")
        if rooms:
            print(f"[PERSPECTIVE] First few rooms: {rooms[:10]}")

        found_any = False

        for view_idx, view in enumerate(ds.iter_perspective_views(), start=1):
            found_any = True
            print("\n" + "-" * 80)
            print(f"[PERSPECTIVE] Processing view #{view_idx}")
            print(f"[PERSPECTIVE] room_id: {view.room_id}")
            print(f"[PERSPECTIVE] position_id: {view.position_id}")
            print(f"[PERSPECTIVE] rgb shape: {view.rgb.shape}, dtype: {view.rgb.dtype}")
            print(f"[PERSPECTIVE] depth shape: {view.depth_mm.shape}, dtype: {view.depth_mm.dtype}")
            print(f"[PERSPECTIVE] semantic shape: {view.semantic.shape}, dtype: {view.semantic.dtype}")
            print(f"[PERSPECTIVE] pose shape: {view.pose.shape}, dtype: {view.pose.dtype}")
            print(f"[PERSPECTIVE] intrinsics shape: {view.intrinsics.shape}, dtype: {view.intrinsics.dtype}")

            try:
                pts, colors, sem = perspective_depth_to_world(
                    view.depth_mm, view.rgb, view.semantic, view.pose, view.intrinsics, stride=args.stride
                )
                print(f"[PERSPECTIVE] points shape: {pts.shape}, dtype: {pts.dtype}")
                print(f"[PERSPECTIVE] colors shape: {colors.shape}, dtype: {colors.dtype}")
                print(f"[PERSPECTIVE] semantic vector shape: {sem.shape}, dtype: {sem.dtype}")
                if sem.size > 0:
                    uniq = np.unique(sem)
                    print(f"[PERSPECTIVE] unique semantic IDs count: {len(uniq)}")
                    print(f"[PERSPECTIVE] first few semantic IDs: {uniq[:20]}")

                bev = make_bev(pts, sem, colors, grid_size=args.grid_size)
                print(f"[PERSPECTIVE] BEV occupancy shape : {bev['occupancy'].shape}")
                print(f"[PERSPECTIVE] BEV structure shape : {bev['structure'].shape}")
                print(f"[PERSPECTIVE] BEV furniture shape : {bev['furniture'].shape}")
                if bev["rgb"] is not None:
                    print(f"[PERSPECTIVE] BEV rgb shape       : {bev['rgb'].shape}")
                else:
                    print("[PERSPECTIVE] BEV rgb shape       : None")

                obs = stack_observation(bev)
                target = target_from_bev(bev)
                print(f"[PERSPECTIVE] obs shape: {obs.shape}, dtype: {obs.dtype}")
                print(f"[PERSPECTIVE] target shape: {target.shape}, dtype: {target.dtype}")

                stem = f"persp_room_{view.room_id}_pos_{view.position_id}"
                obs_path = out / f"{stem}_obs.npy"
                target_path = out / f"{stem}_target.npy"
                preview_path = out / f"{stem}_preview.png"

                print(f"[DEBUG] obs min/max: {obs.min():.4f} / {obs.max():.4f}")
                print(f"[DEBUG] target unique values: {np.unique(target)}")

                np.save(obs_path, obs)
                np.save(target_path, target)
                save_bev_preview(obs, target, preview_path)

                print(f"[PERSPECTIVE] Saved obs    : {obs_path}")
                print(f"[PERSPECTIVE] Saved target : {target_path}")
                print(f"[PERSPECTIVE] Saved preview: {preview_path}")

                samples.append({
                    "id": stem,
                    "source": "perspective",
                    "room_id": view.room_id,
                    "position_id": view.position_id,
                })
                idx += 1
                persp_count += 1

            except Exception as e:
                persp_errors += 1
                print(
                    f"[PERSPECTIVE][ERROR] room_id={view.room_id}, "
                    f"position_id={view.position_id} failed with: {type(e).__name__}: {e}"
                )

        if not found_any:
            print("[PERSPECTIVE][WARN] iter_perspective_views() yielded 0 views.")

    index_path = out / "index.json"
    index_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(f"Total samples saved    : {idx}")
    print(f"Panorama samples saved : {pano_count}")
    print(f"Perspective saved      : {persp_count}")
    print(f"Panorama errors        : {pano_errors}")
    print(f"Perspective errors     : {persp_errors}")
    print(f"Index saved to         : {index_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()