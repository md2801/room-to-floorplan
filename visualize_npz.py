from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


CHANNEL_NAMES_8 = [
    "occupancy",
    "wall",
    "floor",
    "opening",
    "furniture",
    "rgb_r",
    "rgb_g",
    "rgb_b",
]

CHANNEL_NAMES_10 = [
    "occupancy",
    "wall",
    "floor",
    "opening",
    "furniture",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "observed_corner_prior",
    "distance_to_observed_wall",
]


def to_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Convert BEV/observation to HWC format.

    Supports:
        CHW: [C, H, W]
        HWC: [H, W, C]
    """
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    if arr.shape[0] in (8, 10) and arr.shape[-1] not in (8, 10):
        arr = np.transpose(arr, (1, 2, 0))

    return arr.astype(np.float32)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x

    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))

    if mx - mn <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mn) / (mx - mn)).astype(np.float32)


def get_first_existing(data: np.lib.npyio.NpzFile, keys: list[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in data:
            return data[k]
    return None


def get_rgb_from_obs(obs: np.ndarray) -> np.ndarray:
    """
    Build RGB preview from channels 5, 6, 7 if available.
    """
    h, w, c = obs.shape

    if c >= 8:
        rgb = obs[..., 5:8].astype(np.float32)
        rgb = normalize01(rgb)
        return rgb

    return np.zeros((h, w, 3), dtype=np.float32)


def make_overlay(
    base_rgb: np.ndarray,
    room_mask: Optional[np.ndarray] = None,
    wall_mask: Optional[np.ndarray] = None,
    opening_mask: Optional[np.ndarray] = None,
    boundary_mask: Optional[np.ndarray] = None,
    corner_heatmap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create visual overlay:
        room_mask      -> soft grey/white fill
        wall_mask      -> red
        opening_mask   -> green
        boundary_mask  -> blue
        corner_heatmap -> yellow
    """
    out = base_rgb.copy().astype(np.float32)
    out = normalize01(out)

    h, w = out.shape[:2]

    def squeeze_mask(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if m is None:
            return None

        m = np.asarray(m, dtype=np.float32)

        if m.ndim == 3:
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[-1] == 1:
                m = m[..., 0]
            else:
                m = np.max(m, axis=0)

        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        return normalize01(m)

    room = squeeze_mask(room_mask)
    wall = squeeze_mask(wall_mask)
    opening = squeeze_mask(opening_mask)
    boundary = squeeze_mask(boundary_mask)
    corners = squeeze_mask(corner_heatmap)

    if room is not None:
        out = out * (1.0 - 0.25 * room[..., None]) + np.array([0.9, 0.9, 0.9], dtype=np.float32) * (0.25 * room[..., None])

    if wall is not None:
        out[..., 0] = np.maximum(out[..., 0], wall)
        out[..., 1] = out[..., 1] * (1.0 - 0.5 * wall)
        out[..., 2] = out[..., 2] * (1.0 - 0.5 * wall)

    if opening is not None:
        out[..., 1] = np.maximum(out[..., 1], opening)
        out[..., 0] = out[..., 0] * (1.0 - 0.4 * opening)

    if boundary is not None:
        out[..., 2] = np.maximum(out[..., 2], boundary)
        out[..., 0] = out[..., 0] * (1.0 - 0.3 * boundary)

    if corners is not None:
        out[..., 0] = np.maximum(out[..., 0], corners)
        out[..., 1] = np.maximum(out[..., 1], corners)

    return np.clip(out, 0.0, 1.0)


def draw_polygon_targets(
    ax,
    polygon_xy: Optional[np.ndarray],
    corner_valid: Optional[np.ndarray],
    grid_size: int,
    color: str = "cyan",
) -> None:
    """
    Draw sparse polygon/corner targets.

    polygon_xy is expected normalized:
        [max_rooms, max_corners, 2]
    """
    if polygon_xy is None:
        return

    poly = np.asarray(polygon_xy, dtype=np.float32)

    if poly.ndim != 3 or poly.shape[-1] != 2:
        return

    valid = None
    if corner_valid is not None:
        valid = np.asarray(corner_valid, dtype=np.float32)

    max_rooms = poly.shape[0]

    for ri in range(max_rooms):
        if valid is not None:
            mask = valid[ri] > 0.5
            pts = poly[ri][mask]
        else:
            pts = poly[ri]

        if len(pts) < 3:
            continue

        pts_px = pts.copy()

        # Detect normalized points.
        if float(np.nanmax(pts_px)) <= 1.5:
            pts_px[:, 0] *= grid_size - 1
            pts_px[:, 1] *= grid_size - 1

        closed = np.concatenate([pts_px, pts_px[:1]], axis=0)

        ax.plot(closed[:, 0], closed[:, 1], linewidth=1.2, color=color, alpha=0.9)
        ax.scatter(pts_px[:, 0], pts_px[:, 1], s=8, color=color, alpha=0.9)


def draw_boundary_sequence(
    ax,
    boundary: Optional[np.ndarray],
    target_valid: Optional[np.ndarray],
    grid_size: int,
    max_draw: int = 20,
    color: str = "magenta",
) -> None:
    """
    Draw dense boundary_64 or boundary_128 targets.

    Expected:
        [max_rooms, N, 2], normalized.
    """
    if boundary is None:
        return

    b = np.asarray(boundary, dtype=np.float32)

    if b.ndim != 3 or b.shape[-1] != 2:
        return

    valid = None
    if target_valid is not None:
        valid = np.asarray(target_valid, dtype=np.float32)

    for ri in range(min(b.shape[0], max_draw)):
        if valid is not None and valid[ri] < 0.5:
            continue

        pts = b[ri].copy()

        if len(pts) < 3:
            continue

        if float(np.nanmax(pts)) <= 1.5:
            pts[:, 0] *= grid_size - 1
            pts[:, 1] *= grid_size - 1

        closed = np.concatenate([pts, pts[:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1], linewidth=0.8, color=color, alpha=0.65)


def visualize_npz(
    npz_path: Path,
    out_dir: Path,
    show: bool = False,
    save_prefix: Optional[str] = None,
) -> None:
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    if "bev" in data:
        obs = to_hwc(data["bev"])
        input_key = "bev"
    elif "observation" in data:
        obs = to_hwc(data["observation"])
        input_key = "observation"
    else:
        raise KeyError(f"{npz_path.name} has no 'bev' or 'observation' key")

    h, w, c = obs.shape
    grid_size = h

    if c == 8:
        channel_names = CHANNEL_NAMES_8
    elif c == 10:
        channel_names = CHANNEL_NAMES_10
    else:
        channel_names = [f"ch_{i}" for i in range(c)]

    room_mask = get_first_existing(data, ["room_mask", "room_mask_merged"])
    wall_mask = get_first_existing(data, ["wall_mask"])
    opening_mask = get_first_existing(data, ["opening_mask"])
    boundary_mask = get_first_existing(data, ["boundary_mask", "edge_mask"])
    corner_heatmap = get_first_existing(data, ["corner_heatmap"])
    input_corner_prior = get_first_existing(data, ["input_corner_prior"])
    distance_to_wall = get_first_existing(data, ["distance_to_wall"])
    distance_to_wall_target = get_first_existing(data, ["distance_to_wall_target"])

    polygon_xy = get_first_existing(data, ["polygon_xy", "corners"])
    corner_valid = get_first_existing(data, ["corner_valid"])
    room_valid = get_first_existing(data, ["room_valid"])
    boundary_64 = get_first_existing(data, ["boundary_64"])
    boundary_128 = get_first_existing(data, ["boundary_128"])

    rgb = get_rgb_from_obs(obs)

    num_rooms = int(np.asarray(data["num_rooms"]).reshape(-1)[0]) if "num_rooms" in data else -1

    prefix = save_prefix or npz_path.stem

    # # ------------------------------------------------------------
    # # Figure 1: input channels
    # # ------------------------------------------------------------
    # cols = 4
    # rows = int(np.ceil(c / cols))

    # fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    # axes = np.asarray(axes).reshape(-1)

    # for i in range(len(axes)):
    #     ax = axes[i]
    #     ax.axis("off")

    #     if i >= c:
    #         continue

    #     ax.imshow(normalize01(obs[..., i]), cmap="gray")
    #     ax.set_title(f"{i}: {channel_names[i]}")

    # fig.suptitle(
    #     f"{npz_path.name} | input={input_key} | shape={obs.shape} | num_rooms={num_rooms}",
    #     fontsize=14,
    # )
    # fig.tight_layout()

    # out_channels = out_dir / f"{prefix}_01_input_channels.png"
    # fig.savefig(out_channels, dpi=160)
    # if show:
    #     plt.show()
    # plt.close(fig)

    # # ------------------------------------------------------------
    # # Figure 2: target overview
    # # ------------------------------------------------------------
    # overview_items = [
    #     ("RGB preview", rgb, None),
    #     ("Room mask", room_mask, "gray"),
    #     ("Wall mask", wall_mask, "gray"),
    #     ("Opening mask", opening_mask, "gray"),
    #     ("Boundary / edge mask", boundary_mask, "gray"),
    #     ("GT corner heatmap", corner_heatmap, "hot"),
    #     ("Input corner prior", input_corner_prior, "hot"),
    #     ("Distance to wall input", distance_to_wall, "viridis"),
    #     ("Distance to wall target", distance_to_wall_target, "viridis"),
    # ]

    # fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # axes = axes.reshape(-1)

    # for ax, (title, arr, cmap) in zip(axes, overview_items):
    #     ax.axis("off")
    #     ax.set_title(title)

    #     if arr is None:
    #         ax.text(0.5, 0.5, "missing", ha="center", va="center")
    #         continue

    #     arr = np.asarray(arr, dtype=np.float32)

    #     if arr.ndim == 3:
    #         if arr.shape[0] == 1:
    #             arr = arr[0]
    #         elif arr.shape[-1] == 1:
    #             arr = arr[..., 0]
    #         elif arr.shape[-1] == 3:
    #             arr = normalize01(arr)
    #         else:
    #             arr = np.max(arr, axis=0)

    #     if arr.ndim == 2:
    #         ax.imshow(normalize01(arr), cmap=cmap or "gray")
    #     else:
    #         ax.imshow(normalize01(arr))

    # fig.suptitle(f"{npz_path.name} | target overview", fontsize=14)
    # fig.tight_layout()

    # out_targets = out_dir / f"{prefix}_02_target_overview.png"
    # fig.savefig(out_targets, dpi=160)
    # if show:
    #     plt.show()
    # plt.close(fig)

    # ------------------------------------------------------------
    # Figure 3: overlay with polygon/boundary sequence
    # ------------------------------------------------------------
    overlay = make_overlay(
        base_rgb=rgb,
        room_mask=room_mask,
        wall_mask=wall_mask,
        opening_mask=opening_mask,
        boundary_mask=boundary_mask,
        corner_heatmap=corner_heatmap,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.set_title(
        "Overlay: room fill + wall red + opening green + boundary blue + corners yellow"
    )
    ax.axis("off")

    draw_boundary_sequence(
        ax,
        boundary=boundary_128 if boundary_128 is not None else boundary_64,
        target_valid=room_valid,
        grid_size=grid_size,
        color="magenta",
    )

    draw_polygon_targets(
        ax,
        polygon_xy=polygon_xy,
        corner_valid=corner_valid,
        grid_size=grid_size,
        color="cyan",
    )

    out_overlay = out_dir / f"{prefix}_03_overlay_polygons.png"
    fig.savefig(out_overlay, dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------
    # Print quick summary
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print(f"Sample: {npz_path}")
    print(f"Saved:")
    # print(f"  {out_channels}")
    # print(f"  {out_targets}")
    print(f"  {out_overlay}")
    print()
    print("Keys:")
    for key in sorted(data.files):
        value = data[key]
        print(f"  {key:28s} shape={str(value.shape):18s} dtype={value.dtype}")
    print("=" * 80)


def find_npz_files(path: Path) -> list[Path]:
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".npz":
        return [path]

    if (path / "samples").exists():
        return sorted((path / "samples").glob("*.npz"))

    return sorted(path.glob("*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize RoomToFP .npz samples."
    )

    ap.add_argument(
        "--input",
        required=True,
        help="Path to one .npz file, a samples folder, or a dataset folder containing samples/.",
    )
    ap.add_argument(
        "--out-dir",
        default="npz_visualizations",
        help="Folder where visualization PNGs will be saved.",
    )
    ap.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of samples to visualize when input is a folder.",
    )
    ap.add_argument(
        "--random",
        action="store_true",
        help="Randomly select samples instead of taking the first N.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively as well as saving them.",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    npz_files = find_npz_files(input_path)

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {input_path}")

    if input_path.is_file():
        selected = npz_files
    else:
        selected = npz_files.copy()

        if args.random:
            rng = random.Random(int(args.seed))
            rng.shuffle(selected)

        selected = selected[: max(1, int(args.num))]

    print(f"Found {len(npz_files)} .npz files")
    print(f"Visualizing {len(selected)} samples")

    for p in selected:
        visualize_npz(
            npz_path=p,
            out_dir=out_dir,
            show=bool(args.show),
        )


if __name__ == "__main__":
    main()