from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def to_hwc_rgb(bev: np.ndarray) -> np.ndarray:
    """
    Converts BEV RGB channels to an HWC image if available.

    Expected BEV shape:
        (C, H, W)

    Your sample has 8 channels:
        0 occupancy
        1 wall
        2 floor / room evidence
        3 opening
        4 furniture
        5 RGB-R
        6 RGB-G
        7 RGB-B
    """
    if bev.ndim != 3:
        raise ValueError(f"Expected bev shape (C,H,W), got {bev.shape}")

    c, h, w = bev.shape

    if c >= 8:
        rgb = np.stack([bev[5], bev[6], bev[7]], axis=-1)
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    fallback = np.repeat(bev[0][..., None], 3, axis=-1)
    return fallback


def normalize_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def get_mask(sample: dict, key: str) -> np.ndarray | None:
    if key not in sample:
        return None

    arr = sample[key]

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    return arr


def plot_single_mask(ax, mask: np.ndarray | None, title: str):
    ax.set_title(title)
    ax.axis("off")

    if mask is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        return

    ax.imshow(mask, cmap="gray", origin="upper")
    ax.text(
        4,
        16,
        f"shape={mask.shape}\nmin={mask.min():.3f}, max={mask.max():.3f}",
        color="yellow",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )


def polygon_points_to_pixels(points: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Converts normalized polygon points to pixel coordinates.

    Your sample polygon_xy appears normalized to 0..1, so this converts:
        x_pixel = x * grid_size
        y_pixel = y * grid_size
    """
    pts = np.asarray(points, dtype=np.float32).copy()

    if pts.size == 0:
        return pts

    if np.nanmax(pts) <= 1.5:
        pts[:, 0] *= grid_size
        pts[:, 1] *= grid_size

    return pts


def draw_room_polygons(
    ax,
    polygon_xy: np.ndarray,
    corner_valid: np.ndarray,
    room_valid: np.ndarray,
    grid_size: int,
    show_labels: bool = True,
):
    valid_room_indices = np.where(room_valid > 0.5)[0]

    for room_idx in valid_room_indices:
        valid = corner_valid[room_idx] > 0.5
        pts = polygon_xy[room_idx][valid]

        if len(pts) < 3:
            continue

        pts_px = polygon_points_to_pixels(pts, grid_size)

        patch = Polygon(
            pts_px,
            closed=True,
            fill=False,
            linewidth=2.0,
        )
        ax.add_patch(patch)

        ax.scatter(pts_px[:, 0], pts_px[:, 1], s=16)

        if show_labels:
            cx = float(np.mean(pts_px[:, 0]))
            cy = float(np.mean(pts_px[:, 1]))
            ax.text(
                cx,
                cy,
                f"R{room_idx}\n{len(pts)} pts",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )


def make_overview(sample: dict, out_dir: Path):
    bev = sample["bev"]
    grid_size = int(sample.get("grid_size", bev.shape[-1]))

    rgb = to_hwc_rgb(bev)

    masks_to_show = [
        ("BEV occupancy", bev[0]),
        ("BEV wall channel", bev[1]),
        ("BEV floor/room channel", bev[2]),
        ("BEV opening channel", bev[3]),
        ("RGB channels", rgb),
        ("room_mask", get_mask(sample, "room_mask")),
        ("room_mask_merged", get_mask(sample, "room_mask_merged")),
        ("wall_mask", get_mask(sample, "wall_mask")),
        ("opening_mask", get_mask(sample, "opening_mask")),
        ("boundary_mask", get_mask(sample, "boundary_mask")),
        ("edge_mask", get_mask(sample, "edge_mask")),
        ("corner_heatmap", get_mask(sample, "corner_heatmap")),
        ("distance_to_wall_target", get_mask(sample, "distance_to_wall_target")),
    ]

    n = len(masks_to_show)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.2 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (title, img) in zip(axes, masks_to_show):
        ax.set_title(title)
        ax.axis("off")

        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            continue

        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img, origin="upper")
        else:
            ax.imshow(img, cmap="gray", origin="upper")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"RoomToFP sample overview | source={sample.get('source', 'unknown')} | "
        f"num_rooms={sample.get('num_rooms', 'unknown')} | grid={grid_size}",
        fontsize=14,
    )

    fig.tight_layout()
    out_path = out_dir / "sample_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved overview: {out_path}")


def make_polygon_overlay(sample: dict, out_dir: Path):
    bev = sample["bev"]
    grid_size = int(sample.get("grid_size", bev.shape[-1]))

    polygon_xy = sample["polygon_xy"]
    corner_valid = sample["corner_valid"]
    room_valid = sample["room_valid"]

    base = get_mask(sample, "room_mask_merged")
    if base is None:
        base = bev[0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(base, cmap="gray", origin="upper")
    ax.set_title("GT room polygons over room_mask_merged")
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_room_polygons(
        ax=ax,
        polygon_xy=polygon_xy,
        corner_valid=corner_valid,
        room_valid=room_valid,
        grid_size=grid_size,
        show_labels=True,
    )

    fig.tight_layout()
    out_path = out_dir / "polygon_overlay.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved polygon overlay: {out_path}")


def make_room_masks_grid(sample: dict, out_dir: Path):
    if "room_masks" not in sample:
        print("No room_masks found.")
        return

    room_masks = sample["room_masks"]
    room_valid = sample["room_valid"]
    polygon_xy = sample["polygon_xy"]
    corner_valid = sample["corner_valid"]
    grid_size = int(sample.get("grid_size", room_masks.shape[-1]))

    valid_room_indices = np.where(room_valid > 0.5)[0]

    if len(valid_room_indices) == 0:
        print("No valid rooms found.")
        return

    cols = 4
    rows = int(np.ceil(len(valid_room_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, room_idx in zip(axes, valid_room_indices):
        mask = room_masks[room_idx]
        valid = corner_valid[room_idx] > 0.5
        pts = polygon_xy[room_idx][valid]
        pts_px = polygon_points_to_pixels(pts, grid_size)

        ax.imshow(mask, cmap="gray", origin="upper")
        ax.set_title(f"Room {room_idx} | corners={len(pts)}")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        if len(pts_px) >= 3:
            patch = Polygon(
                pts_px,
                closed=True,
                fill=False,
                linewidth=2.0,
            )
            ax.add_patch(patch)
            ax.scatter(pts_px[:, 0], pts_px[:, 1], s=12)

    for ax in axes[len(valid_room_indices):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / "room_masks_with_polygons.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved per-room masks: {out_path}")


def print_sample_summary(sample: dict):
    print("\n===== SAMPLE SUMMARY =====")

    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                print(f"{key:28s} scalar={value.item()} dtype={value.dtype}")
            else:
                if np.issubdtype(value.dtype, np.number):
                    print(
                        f"{key:28s} shape={str(value.shape):18s} "
                        f"dtype={value.dtype} "
                        f"min={np.nanmin(value):.4f} max={np.nanmax(value):.4f}"
                    )
                else:
                    print(f"{key:28s} shape={str(value.shape):18s} dtype={value.dtype}")
        else:
            print(f"{key:28s} {type(value)}")

    if "room_valid" in sample and "corner_valid" in sample:
        room_valid = sample["room_valid"]
        corner_valid = sample["corner_valid"]
        valid_room_indices = np.where(room_valid > 0.5)[0]

        print("\n===== VALID ROOMS =====")
        print(f"Valid rooms: {len(valid_room_indices)}")

        for room_idx in valid_room_indices:
            n_corners = int((corner_valid[room_idx] > 0.5).sum())
            print(f"Room {room_idx:02d}: {n_corners} valid corners")


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    sample = {}

    for key in data.files:
        arr = data[key]

        if arr.ndim == 0:
            try:
                sample[key] = arr.item()
            except Exception:
                sample[key] = arr
        else:
            sample[key] = arr

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Path to .npz sample file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sample_visualization",
        help="Output directory for visualizations.",
    )

    args = parser.parse_args()

    sample_path = Path(args.sample)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = load_npz(sample_path)

    print_sample_summary(sample)
    make_overview(sample, out_dir)
    make_polygon_overlay(sample, out_dir)
    make_room_masks_grid(sample, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()