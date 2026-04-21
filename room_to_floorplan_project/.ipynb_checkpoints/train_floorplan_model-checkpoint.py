from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from room_to_floorplan.model import FloorplanUNet


class BEVDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.obs_files = sorted(self.root.glob("*_obs.npy"))

        print("=" * 80)
        print("[DATASET]")
        print(f"Root              : {self.root.resolve()}")
        print(f"Observation files : {len(self.obs_files)}")
        if not self.obs_files:
            print("[WARN] No *_obs.npy files found")
        print("=" * 80)

    def __len__(self):
        return len(self.obs_files)

    def __getitem__(self, idx):
        obs_file = self.obs_files[idx]
        target_file = Path(str(obs_file).replace("_obs.npy", "_target.npy"))

        x = np.load(obs_file).transpose(2, 0, 1).astype(np.float32)
        y = np.load(target_file).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def summarize_dataset_shapes(ds: BEVDataset):
    obs_shapes = {}
    target_shapes = {}

    for obs_file in ds.obs_files:
        target_file = Path(str(obs_file).replace("_obs.npy", "_target.npy"))
        try:
            x = np.load(obs_file)
            y = np.load(target_file)
            obs_shapes[x.shape] = obs_shapes.get(x.shape, 0) + 1
            target_shapes[y.shape] = target_shapes.get(y.shape, 0) + 1
        except Exception as e:
            print(f"[SHAPE CHECK][ERROR] {obs_file.name}: {type(e).__name__}: {e}")

    print("=" * 80)
    print("[SHAPE CHECK]")
    print("Observation shapes:")
    for shape, count in sorted(obs_shapes.items(), key=lambda kv: str(kv[0])):
        print(f"  {shape} -> {count}")

    print("Target shapes:")
    for shape, count in sorted(target_shapes.items(), key=lambda kv: str(kv[0])):
        print(f"  {shape} -> {count}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    print("=" * 80)
    print("TRAIN FLOORPLAN MODEL")
    print(f"Data dir   : {args.data_dir}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"Output dir : {args.output_dir}")
    print("=" * 80)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = BEVDataset(args.data_dir)
    if len(ds) == 0:
        print("[FATAL] Dataset is empty. No training can be performed.")
        return

    summarize_dataset_shapes(ds)

    n_val = max(1, int(0.2 * len(ds)))
    n_train = max(1, len(ds) - n_val)

    print("[SPLIT]")
    print(f"Total : {len(ds)}")
    print(f"Train : {n_train}")
    print(f"Val   : {n_val}")
    print("=" * 80)

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    if device == "cuda":
        print(f"[CUDA] Device count: {torch.cuda.device_count()}")
        print(f"[CUDA] Device name : {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    model = FloorplanUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    hist = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_dl, desc=f"train {epoch + 1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()

            train_loss += float(loss.item())

        train_loss /= max(len(train_dl), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
                val_loss += float(loss.item())

        val_loss /= max(len(val_dl), 1)

        hist_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        hist.append(hist_entry)

        print(
            f"[EPOCH {epoch + 1:02d}] "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = out / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CHECKPOINT] Saved best model -> {ckpt_path}")

    history_path = out / "history.json"
    history_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best val loss : {best_val:.6f}")
    print(f"History saved : {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()