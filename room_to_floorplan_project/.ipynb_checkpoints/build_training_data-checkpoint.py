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


def safe_stem(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


def already_exists(out_dir: Path, stem: str, save_previews: bool) -> bool:
    obs_ok = (out_dir / f"{stem}_obs.npy").exists()
    target_ok = (out_dir / f"{stem}_target.npy").exists()
    preview_ok = True if not save_previews else (out_dir / f"{stem}_preview.png").exists()
    return obs_ok and target_ok and preview_ok


def reached_limit(
    total_count: int,
    subset_count: int,
    max_total: int | None,
    max_subset: int | None,
) -> bool:
    if max_total is not None and total_count >= max_total:
        return True
    if max_subset is not None and subset_count >= max_subset:
        return True
    return False


def is_too_weak(target: np.ndarray, min_structure_pixels: float, min_furniture_pixels: float) -> bool:
    structure_pixels = float(target[0].sum())
    furniture_pixels = float(target[1].sum())
    return structure_pixels < min_structure_pixels or furniture_pixels < min_furniture_pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perspective-zips", nargs="*", default=[])
    ap.add_argument("--panorama-zips", nargs="*", default=[])
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)

    # storage / runtime control
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Maximum total number of samples to save across all zips")
    ap.add_argument("--max-panorama-samples", type=int, default=None,
                    help="Maximum number of panorama samples to save")
    ap.add_argument("--max-perspective-samples", type=int, default=None,
                    help="Maximum number of perspective samples to save")
    ap.add_argument("--save-previews", action="store_true",
                    help="Also save preview PNGs (off by default to save disk space)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip samples whose output files already exist")

    # quality control
    ap.add_argument("--min-structure-pixels", type=float, default=80.0,
                    help="Skip very weak samples whose structure target has fewer pixels than this")
    ap.add_argument("--min-furniture-pixels", type=float, default=0.0,
                    help="Optional furniture threshold; keep at 0 to allow empty-furniture rooms")

    args = ap.parse_args()

    print("=" * 80)
    print("BUILD TRAINING DATA")
    print("=" * 80)
    print(f"Perspective zips: {len(args.perspective_zips)}")
    for p in args.perspective_zips:
        print(f"- {p}")
    print(f"Panorama zips: {len(args.panorama_zips)}")
    for p in args.panorama_zips:
        print(f"- {p}")
    print(f"Output dir: {args.output_dir}")
    print(f"Grid size: {args.grid_size}")
    print(f"Stride: {args.stride}")
    print(f"Max total samples: {args.max_samples}")
    print(f"Max panorama samples: {args.max_panorama_samples}")
    print(f"Max perspective samples: {args.max_perspective_samples}")
    print(f"Save previews: {args.save_previews}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Min structure pixels: {args.min_structure_pixels}")
    print(f"Min furniture pixels: {args.min_furniture_pixels}")
    print("=" * 80)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    idx = 0
    pano_count = 0
    persp_count = 0
    pano_errors = 0
    persp_errors = 0
    skipped_existing = 0
    skipped_weak = 0

    stop_all = False

    pano_bev_kwargs = dict(
        grid_size=args.grid_size,
        meters_per_pixel=None,
        z_clip=(-0.5, 3.0),
        height_band_above_floor=(0.15, 1.60),
        xy_padding_m=0.35,
        splat_radius=2,
    )

    persp_bev_kwargs = dict(
        grid_size=args.grid_size,
        meters_per_pixel=None,
        z_clip=(-0.5, 3.0),
        height_band_above_floor=(0.20, 1.50),
        xy_padding_m=0.35,
        splat_radius=1,
    )

    # Panorama zips
    for zip_path in args.panorama_zips:
        if stop_all:
            break

        zip_tag = safe_stem(zip_path)

        print(f"[PANORAMA] Opening zip: {zip_path}")
        ds = Structured3DZip(zip_path)
        scenes = ds.list_scenes()
        print(f"[PANORAMA] Scenes found: {len(scenes)}")

        found_any = False

        for view_idx, view in enumerate(ds.iter_panorama_views(), start=1):
            if reached_limit(idx, pano_count, args.max_samples, args.max_panorama_samples):
                print("[PANORAMA] Reached configured sample limit. Stopping panorama generation.")
                stop_all = args.max_samples is not None and idx >= args.max_samples
                break

            found_any = True
            stem = f"{zip_tag}_{view.scene_id}_pano_room_{view.room_id}"

            if args.skip_existing and already_exists(out, stem, args.save_previews):
                skipped_existing += 1
                if skipped_existing % 50 == 0:
                    print(f"[SKIP] Existing samples skipped so far: {skipped_existing}")
                continue

            try:
                pts, colors, sem = panorama_depth_to_world(
                    view.depth_mm,
                    view.rgb,
                    view.semantic,
                    view.camera_xyz_mm,
                    stride=args.stride,
                )

                bev = make_bev(pts, sem, colors, **pano_bev_kwargs)
                obs = stack_observation(bev)
                target = target_from_bev(bev)

                if is_too_weak(target, args.min_structure_pixels, args.min_furniture_pixels):
                    skipped_weak += 1
                    continue

                np.save(out / f"{stem}_obs.npy", obs)
                np.save(out / f"{stem}_target.npy", target)

                if args.save_previews:
                    save_bev_preview(obs, target, out / f"{stem}_preview.png")

                samples.append({
                    "id": stem,
                    "source": "panorama",
                    "zip_name": zip_tag,
                    "scene_id": view.scene_id,
                    "room_id": view.room_id,
                    "grid_size": args.grid_size,
                    "stride": args.stride,
                    "meters_per_pixel": float(bev["meters_per_pixel"][0]),
                    "origin_xy": bev["origin_xy"].tolist(),
                    "center_xy": bev["center_xy"].tolist(),
                    "floor_z": float(bev["floor_z"][0]),
                    "height_band_above_floor": bev["height_band_above_floor"].tolist(),
                })

                idx += 1
                pano_count += 1

                if pano_count % 25 == 0:
                    print(f"[PANORAMA] Saved {pano_count} panorama samples so far...")
                if idx % 50 == 0:
                    print(f"[TOTAL] Saved {idx} total samples so far...")

            except Exception as e:
                pano_errors += 1
                print(
                    f"[PANORAMA][ERROR] zip={zip_tag}, scene_id={view.scene_id}, "
                    f"room_id={view.room_id}: {type(e).__name__}: {e}"
                )

        if not found_any:
            print(f"[PANORAMA][WARN] No panorama views found in: {zip_path}")

    # Perspective zips
    for zip_path in args.perspective_zips:
        if stop_all:
            break

        if reached_limit(idx, persp_count, args.max_samples, None):
            print("[PERSPECTIVE] Global sample limit reached before starting perspective zips.")
            break

        zip_tag = safe_stem(zip_path)

        print("\n" + "-" * 80)
        print(f"[PERSPECTIVE] Opening zip: {zip_path}")

        ds = Structured3DZip(zip_path)
        scenes = ds.list_scenes()
        print(f"[PERSPECTIVE] Scenes found: {len(scenes)}")

        found_any = False

        for view_idx, view in enumerate(ds.iter_perspective_views(), start=1):
            if reached_limit(idx, persp_count, args.max_samples, args.max_perspective_samples):
                print("[PERSPECTIVE] Reached configured sample limit. Stopping perspective generation.")
                stop_all = args.max_samples is not None and idx >= args.max_samples
                break

            found_any = True
            stem = f"{zip_tag}_{view.scene_id}_persp_room_{view.room_id}_pos_{view.position_id}"

            if args.skip_existing and already_exists(out, stem, args.save_previews):
                skipped_existing += 1
                if skipped_existing % 50 == 0:
                    print(f"[SKIP] Existing samples skipped so far: {skipped_existing}")
                continue

            try:
                pts, colors, sem = perspective_depth_to_world(
                    view.depth_mm,
                    view.rgb,
                    view.semantic,
                    view.pose,
                    view.intrinsics,
                    stride=args.stride,
                )

                bev = make_bev(pts, sem, colors, **persp_bev_kwargs)
                obs = stack_observation(bev)
                target = target_from_bev(bev)

                if is_too_weak(target, args.min_structure_pixels, args.min_furniture_pixels):
                    skipped_weak += 1
                    continue

                np.save(out / f"{stem}_obs.npy", obs)
                np.save(out / f"{stem}_target.npy", target)

                if args.save_previews:
                    save_bev_preview(obs, target, out / f"{stem}_preview.png")

                samples.append({
                    "id": stem,
                    "source": "perspective",
                    "zip_name": zip_tag,
                    "scene_id": view.scene_id,
                    "room_id": view.room_id,
                    "position_id": view.position_id,
                    "grid_size": args.grid_size,
                    "stride": args.stride,
                    "meters_per_pixel": float(bev["meters_per_pixel"][0]),
                    "origin_xy": bev["origin_xy"].tolist(),
                    "center_xy": bev["center_xy"].tolist(),
                    "floor_z": float(bev["floor_z"][0]),
                    "height_band_above_floor": bev["height_band_above_floor"].tolist(),
                })

                idx += 1
                persp_count += 1

                if persp_count % 50 == 0:
                    print(f"[PERSPECTIVE] Saved {persp_count} perspective samples so far...")
                if idx % 50 == 0:
                    print(f"[TOTAL] Saved {idx} total samples so far...")

            except Exception as e:
                persp_errors += 1
                print(
                    f"[PERSPECTIVE][ERROR] zip={zip_tag}, scene_id={view.scene_id}, "
                    f"room_id={view.room_id}, position_id={view.position_id}: "
                    f"{type(e).__name__}: {e}"
                )

        if not found_any:
            print(f"[PERSPECTIVE][WARN] No perspective views found in: {zip_path}")

    index_path = out / "index.json"
    index_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(f"Total samples saved: {idx}")
    print(f"Panorama samples saved: {pano_count}")
    print(f"Perspective samples saved: {persp_count}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped weak samples: {skipped_weak}")
    print(f"Panorama errors: {pano_errors}")
    print(f"Perspective errors: {persp_errors}")
    print(f"Preview saving enabled: {args.save_previews}")
    print(f"Index saved to: {index_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()