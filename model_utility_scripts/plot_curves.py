from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_history(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "history" in data:
        data = data["history"]

    if not isinstance(data, list):
        raise ValueError("Expected history.json to contain a list of epoch dictionaries.")

    rows = []
    for item in data:
        row = {
            "epoch": item.get("epoch"),
            "lr": item.get("lr"),
        }

        train = item.get("train", {}) or {}
        val = item.get("val", {}) or {}

        for k, v in train.items():
            row[f"train_{k}"] = v

        for k, v in val.items():
            row[f"val_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_curve(df: pd.DataFrame, cols: list[str], title: str, ylabel: str, out_path: Path) -> None:
    available = [c for c in cols if c in df.columns]

    if not available:
        print(f"No available columns for {title}")
        return

    plt.figure(figsize=(10, 6))

    for col in available:
        plt.plot(df["epoch"], df[col], marker="o", linewidth=2, label=col)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot RoomToFP training curves from history.json")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--history", default="history.json")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / args.history

    if not history_path.exists():
        raise FileNotFoundError(f"Could not find history file: {history_path}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_history(history_path)

    print("Loaded columns:")
    for c in df.columns:
        print(f"  - {c}")

    df.to_csv(out_dir / "history_flattened.csv", index=False)
    print(f"{out_dir / 'history_flattened.csv'}")

    plot_curve(
        df,
        ["train_loss", "val_loss"],
        "Total Loss Curve",
        "Loss",
        out_dir / "total_loss_curve.png",
    )

    plot_curve(
        df,
        ["train_coord_l1", "val_coord_l1"],
        "Coordinate L1 Curve",
        "Coordinate L1",
        out_dir / "coord_l1_curve.png",
    )

    plot_curve(
        df,
        ["train_wall_dice", "val_wall_dice"],
        "Wall Dice Curve",
        "Dice",
        out_dir / "wall_dice_curve.png",
    )

    plot_curve(
        df,
        ["train_opening_dice", "val_opening_dice"],
        "Opening Dice Curve",
        "Dice",
        out_dir / "opening_dice_curve.png",
    )

    plot_curve(
        df,
        ["train_matched_room_mask_dice", "val_matched_room_mask_dice"],
        "Matched Room Mask Dice Curve",
        "Dice",
        out_dir / "matched_room_mask_dice_curve.png",
    )

    plot_curve(
        df,
        ["train_boundary_dice", "val_boundary_dice"],
        "Boundary Dice Curve",
        "Dice",
        out_dir / "boundary_dice_curve.png",
    )

    plot_curve(
        df,
        ["train_corner_heatmap_dice", "val_corner_heatmap_dice"],
        "Corner Heatmap Dice Curve",
        "Dice",
        out_dir / "corner_heatmap_dice_curve.png",
    )

    plot_curve(
        df,
        [
            "val_wall_dice",
            "val_opening_dice",
            "val_matched_room_mask_dice",
            "val_boundary_dice",
            "val_corner_heatmap_dice",
        ],
        "Validation Dice Metrics",
        "Dice",
        out_dir / "validation_dice_metrics_curve.png",
    )

    plot_curve(
        df,
        ["lr"],
        "Learning Rate Curve",
        "Learning Rate",
        out_dir / "learning_rate_curve.png",
    )


if __name__ == "__main__":
    main()