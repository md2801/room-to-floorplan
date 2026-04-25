from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from room_to_floorplan.model import FloorplanUNet

import matplotlib
matplotlib.use("Agg")


class BEVDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.obs_files = sorted(self.root.glob("*_obs.npy"))

        print("=" * 80)
        print("[DATASET]")
        print(f"Root: {self.root.resolve()}")
        print(f"Observation files: {len(self.obs_files)}")
        if not self.obs_files:
            print("[WARN] No *_obs.npy files found")

    def __len__(self):
        return len(self.obs_files)

    def __getitem__(self, idx):
        obs_file = self.obs_files[idx]
        target_file = Path(str(obs_file).replace("_obs.npy", "_target.npy"))

        x = np.load(obs_file).transpose(2, 0, 1).astype(np.float32)
        y = np.load(target_file).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "obs_file": obs_file.name,
            "target_file": target_file.name,
        }


def load_model_from_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = FloorplanUNet().to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # fallback in case an older checkpoint only stored raw state_dict
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def save_prediction_figure(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
):
    occ = x[0]
    structure_in = x[1]
    furniture_in = x[2]
    rgb = np.transpose(x[3:6], (1, 2, 0))

    gt_structure = y_true[0]
    gt_furniture = y_true[1]

    pred_structure = y_pred[0]
    pred_furniture = y_pred[1]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title)

    axes[0, 0].imshow(occ, cmap="gray")
    axes[0, 0].set_title("Occupancy")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(structure_in, cmap="gray")
    axes[0, 1].set_title("Structure input")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(furniture_in, cmap="gray")
    axes[0, 2].set_title("Furniture input")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(np.clip(rgb, 0.0, 1.0))
    axes[0, 3].set_title("RGB BEV")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(gt_structure, cmap="gray")
    axes[1, 0].set_title("GT Structure")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_structure, cmap="gray")
    axes[1, 1].set_title("Pred Structure")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(gt_furniture, cmap="gray")
    axes[1, 2].set_title("GT Furniture")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(pred_furniture, cmap="gray")
    axes[1, 3].set_title("Pred Furniture")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best.pt or last.pt")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to save prediction images")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 80)
    print("VISUALIZE MODEL PREDICTIONS")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BEVDataset(args.data_dir)
    if len(dataset) == 0:
        print("[FATAL] Dataset is empty.")
        return

    rng = random.Random(args.seed)
    chosen_indices = list(range(len(dataset)))
    rng.shuffle(chosen_indices)
    chosen_indices = chosen_indices[: min(args.num_samples, len(dataset))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    model = load_model_from_checkpoint(Path(args.checkpoint), device=device)

    saved = 0
    with torch.no_grad():
        for idx in chosen_indices:
            sample = dataset[idx]

            x = sample["x"].unsqueeze(0).to(device)
            y = sample["y"].numpy()
            obs_file = sample["obs_file"]

            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs >= args.threshold).float().cpu().numpy()[0]

            x_np = sample["x"].numpy()

            out_path = out_dir / f"{Path(obs_file).stem}_pred.png"
            save_prediction_figure(
                x=x_np,
                y_true=y,
                y_pred=pred,
                out_path=out_path,
                title=obs_file,
            )
            saved += 1
            print(f"[SAVED] {out_path}")

    print("=" * 80)
    print(f"DONE. Saved {saved} prediction visualizations to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()