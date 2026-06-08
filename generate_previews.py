from __future__ import annotations

"""
Generate ONLY annotation-based floorplan_preview.png files for Structured3D scenes.

This script:
- Uses annotation_3d.json.
- Uses the same preview generation path as the prepare/render code.
- Uses BEV/crop settings so previews align with your RoomToFP grid.
- Does NOT save .npz samples.
- Does NOT save BEV previews, masks, SVG, metadata, NPY files, or split files.
- Does NOT overwrite existing previews unless --overwrite is used.

Final output per scene:
    output_dir/
        scene_00000/
            floorplan_preview.png
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from render_structured3d_roomtofp_floorplan import build_roomtofp_scene_outputs


def find_scene_dirs(root: Path) -> list[Path]:
    root = Path(root)

    if root.name.startswith("scene_") and (root / "annotation_3d.json").exists():
        return [root]

    return sorted(
        p for p in root.rglob("scene_*")
        if p.is_dir() and (p / "annotation_3d.json").exists()
    )


def make_unique_output_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    parent = base_dir.parent
    stem = base_dir.name

    for i in range(1, 10000):
        candidate = parent / f"{stem}_{i:03d}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create unique output folder for {base_dir}")


def build_renderer_args(args: argparse.Namespace) -> SimpleNamespace:
    """
    Args object expected by build_roomtofp_scene_outputs(...).

    This includes all fields used by render_structured3d_roomtofp_floorplan.py,
    but the script still only copies floorplan_preview.png into the final output.
    """
    return SimpleNamespace(
        grid_size=args.grid_size,
        stride=args.stride,
        z_min=args.z_min,
        z_max=args.z_max,
        height_lo=args.height_lo,
        height_hi=args.height_hi,
        xy_padding_m=args.xy_padding_m,
        splat_radius=args.splat_radius,
        rgb_name=args.rgb_name,

        # Required by make_annotation_masks / preview renderer
        wall_thickness=args.wall_thickness,

        # Required by parse_room_polygons
        min_room_area_m2=args.min_room_area_m2,
        exclude_balcony=args.exclude_balcony,
        keep_undefined_label=args.keep_undefined_label,

        # Required by parse_opening_lines
        min_opening_length_m=args.min_opening_length_m,
        max_opening_wall_distance_m=args.max_opening_wall_distance_m,

        # Required by furniture extraction, even when furniture is off
        furniture_source=args.furniture_source if args.include_furniture else "off",
        min_furniture_area_px=args.min_furniture_area_px,
        min_furniture_width_px=args.min_furniture_width_px,
        min_furniture_height_px=args.min_furniture_height_px,
        max_furniture_aspect_ratio=args.max_furniture_aspect_ratio,
        furniture_room_erosion_px=args.furniture_room_erosion_px,
    )


def generate_preview_only(
    scene_dir: Path,
    final_out_dir: Path,
    variants: list[str],
    args: argparse.Namespace,
) -> Path:
    """
    Generate full renderer outputs in a temporary directory, then copy only
    floorplan_preview.png to the final output folder.
    """
    scene_dir = Path(scene_dir)
    final_out_dir = Path(final_out_dir)

    renderer_args = build_renderer_args(args)

    with tempfile.TemporaryDirectory(prefix=f"{scene_dir.name}_preview_tmp_") as tmp:
        tmp_dir = Path(tmp)

        build_roomtofp_scene_outputs(
            scene_dir,
            tmp_dir,
            variants,
            renderer_args,
        )

        src_preview = tmp_dir / "floorplan_preview.png"

        if not src_preview.exists():
            raise FileNotFoundError(
                f"Renderer did not create expected file: {src_preview}"
            )

        final_out_dir.mkdir(parents=True, exist_ok=True)

        dst_preview = final_out_dir / "floorplan_preview.png"
        shutil.copy2(src_preview, dst_preview)

    return dst_preview


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate ONLY floorplan_preview.png from Structured3D annotation_3d.json."
    )

    ap.add_argument(
        "--scene-dir",
        nargs="*",
        default=[],
        help="One or more explicit Structured3D scene_* folders.",
    )

    ap.add_argument(
        "--structured3d-root",
        nargs="*",
        default=[],
        help="One or more roots containing scene_* folders.",
    )

    ap.add_argument(
        "--scene-names",
        nargs="*",
        default=[],
        help="Optional filter, e.g. scene_00000 scene_00001.",
    )

    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output root. Each scene gets its own folder containing only floorplan_preview.png.",
    )

    ap.add_argument("--max-scenes", type=int, default=0, help="0 means all found scenes.")

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

    ap.add_argument("--min-room-area-m2", type=float, default=1.0)
    ap.add_argument("--exclude-balcony", action="store_true")
    ap.add_argument("--keep-undefined-label", action="store_true")

    ap.add_argument("--wall-thickness", type=int, default=5)

    ap.add_argument("--min-opening-length-m", type=float, default=0.35)
    ap.add_argument("--max-opening-wall-distance-m", type=float, default=0.20)

    ap.add_argument("--min-furniture-area-px", type=int, default=120)
    ap.add_argument("--min-furniture-width-px", type=int, default=8)
    ap.add_argument("--min-furniture-height-px", type=int, default=8)
    ap.add_argument("--max-furniture-aspect-ratio", type=float, default=12.0)
    ap.add_argument("--furniture-room-erosion-px", type=int, default=1)

    ap.add_argument("--include-furniture", action="store_true")
    ap.add_argument(
        "--furniture-source",
        choices=["auto", "annotation", "semantic", "both", "off"],
        default="auto",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene preview folders.",
    )

    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scene if floorplan_preview.png already exists.",
    )

    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scene_dirs: list[Path] = []
    seen: set[str] = set()

    for s in args.scene_dir:
        p = Path(s).resolve()
        if not (p / "annotation_3d.json").exists():
            print(f"--Warning-- skipped explicit scene, missing annotation_3d.json: {p}")
            continue

        key = str(p)
        if key not in seen:
            scene_dirs.append(p)
            seen.add(key)

    for root in args.structured3d_root:
        root_path = Path(root).resolve()
        found = find_scene_dirs(root_path)
        print(f"--Scene Info-- {root_path}: found {len(found)} scenes")

        for p in found:
            key = str(p.resolve())
            if key not in seen:
                scene_dirs.append(p.resolve())
                seen.add(key)

    if args.scene_names:
        wanted = set(args.scene_names)
        scene_dirs = [p for p in scene_dirs if p.name in wanted]

    scene_dirs = sorted(scene_dirs)

    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]

    if not scene_dirs:
        raise ValueError("No scenes found. Provide --scene-dir or --structured3d-root.")

    print(f"--Scene Info-- Total scenes to preview: {len(scene_dirs)}")
    print(f"--Output Root-- {output_root}")

    for scene_dir in tqdm(scene_dirs, desc="preview only"):
        base_out = output_root / scene_dir.name
        preview_path = base_out / "floorplan_preview.png"

        if preview_path.exists() and args.skip_existing:
            print(f"--Skip Existing-- {preview_path}")
            continue

        if base_out.exists() and args.overwrite:
            shutil.rmtree(base_out)

        final_out = base_out if args.overwrite else make_unique_output_dir(base_out)

        try:
            dst = generate_preview_only(
                scene_dir=scene_dir,
                final_out_dir=final_out,
                variants=variants,
                args=args,
            )
            print(f"--Saved-- {dst}")

        except Exception as e:
            print(f"--Warning-- Failed {scene_dir}: {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()