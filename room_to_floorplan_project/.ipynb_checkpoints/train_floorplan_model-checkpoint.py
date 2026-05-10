from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from room_to_floorplan.model import FloorplanUNet, FloorplanUNetAttention, FloorplanUNetPlusPlus

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
        print(f"{shape} -> {count}")

    print("Target shapes:")
    for shape, count in sorted(target_shapes.items(), key=lambda kv: str(kv[0])):
        print(f"{shape} -> {count}")


def compute_segmentation_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    dims = (0, 2, 3)

    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=dims)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)

    tp_o = tp.sum()
    fp_o = fp.sum()
    fn_o = fn.sum()
    tn_o = tn.sum()

    overall_precision = tp_o / (tp_o + fp_o + eps)
    overall_recall = tp_o / (tp_o + fn_o + eps)
    overall_f1 = (2.0 * overall_precision * overall_recall) / (overall_precision + overall_recall + eps)
    overall_dice = (2.0 * tp_o) / (2.0 * tp_o + fp_o + fn_o + eps)
    overall_iou = tp_o / (tp_o + fp_o + fn_o + eps)
    overall_pixel_acc = (tp_o + tn_o) / (tp_o + tn_o + fp_o + fn_o + eps)

    return {
        "overall": {
            "precision": float(overall_precision.item()),
            "recall": float(overall_recall.item()),
            "f1": float(overall_f1.item()),
            "dice": float(overall_dice.item()),
            "iou": float(overall_iou.item()),
            "pixel_acc": float(overall_pixel_acc.item()),
        },
        "per_channel": {
            "structure": {
                "precision": float(precision[0].item()) if len(precision) > 0 else 0.0,
                "recall": float(recall[0].item()) if len(recall) > 0 else 0.0,
                "f1": float(f1[0].item()) if len(f1) > 0 else 0.0,
                "dice": float(dice[0].item()) if len(dice) > 0 else 0.0,
                "iou": float(iou[0].item()) if len(iou) > 0 else 0.0,
                "pixel_acc": float(pixel_acc[0].item()) if len(pixel_acc) > 0 else 0.0,
            },
            "furniture": {
                "precision": float(precision[1].item()) if len(precision) > 1 else 0.0,
                "recall": float(recall[1].item()) if len(recall) > 1 else 0.0,
                "f1": float(f1[1].item()) if len(f1) > 1 else 0.0,
                "dice": float(dice[1].item()) if len(dice) > 1 else 0.0,
                "iou": float(iou[1].item()) if len(iou) > 1 else 0.0,
                "pixel_acc": float(pixel_acc[1].item()) if len(pixel_acc) > 1 else 0.0,
            },
        },
    }


def init_metric_sums():
    return {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "overall_precision": 0.0,
        "overall_recall": 0.0,
        "overall_f1": 0.0,
        "overall_dice": 0.0,
        "overall_iou": 0.0,
        "overall_pixel_acc": 0.0,
        "structure_precision": 0.0,
        "structure_recall": 0.0,
        "structure_f1": 0.0,
        "structure_dice": 0.0,
        "structure_iou": 0.0,
        "structure_pixel_acc": 0.0,
        "furniture_precision": 0.0,
        "furniture_recall": 0.0,
        "furniture_f1": 0.0,
        "furniture_dice": 0.0,
        "furniture_iou": 0.0,
        "furniture_pixel_acc": 0.0,
        "num_batches": 0,
    }


def update_metric_sums(metric_sums, total_loss, bce_loss, d_loss, metric_dict):
    metric_sums["loss"] += float(total_loss.item())
    metric_sums["bce"] += float(bce_loss.item())
    metric_sums["dice_loss"] += float(d_loss.item())

    metric_sums["overall_precision"] += metric_dict["overall"]["precision"]
    metric_sums["overall_recall"] += metric_dict["overall"]["recall"]
    metric_sums["overall_f1"] += metric_dict["overall"]["f1"]
    metric_sums["overall_dice"] += metric_dict["overall"]["dice"]
    metric_sums["overall_iou"] += metric_dict["overall"]["iou"]
    metric_sums["overall_pixel_acc"] += metric_dict["overall"]["pixel_acc"]

    metric_sums["structure_precision"] += metric_dict["per_channel"]["structure"]["precision"]
    metric_sums["structure_recall"] += metric_dict["per_channel"]["structure"]["recall"]
    metric_sums["structure_f1"] += metric_dict["per_channel"]["structure"]["f1"]
    metric_sums["structure_dice"] += metric_dict["per_channel"]["structure"]["dice"]
    metric_sums["structure_iou"] += metric_dict["per_channel"]["structure"]["iou"]
    metric_sums["structure_pixel_acc"] += metric_dict["per_channel"]["structure"]["pixel_acc"]

    metric_sums["furniture_precision"] += metric_dict["per_channel"]["furniture"]["precision"]
    metric_sums["furniture_recall"] += metric_dict["per_channel"]["furniture"]["recall"]
    metric_sums["furniture_f1"] += metric_dict["per_channel"]["furniture"]["f1"]
    metric_sums["furniture_dice"] += metric_dict["per_channel"]["furniture"]["dice"]
    metric_sums["furniture_iou"] += metric_dict["per_channel"]["furniture"]["iou"]
    metric_sums["furniture_pixel_acc"] += metric_dict["per_channel"]["furniture"]["pixel_acc"]

    metric_sums["num_batches"] += 1


def finalize_metric_sums(metric_sums):
    n = max(metric_sums["num_batches"], 1)
    return {
        "loss": metric_sums["loss"] / n,
        "bce": metric_sums["bce"] / n,
        "dice_loss": metric_sums["dice_loss"] / n,
        "overall_precision": metric_sums["overall_precision"] / n,
        "overall_recall": metric_sums["overall_recall"] / n,
        "overall_f1": metric_sums["overall_f1"] / n,
        "overall_dice": metric_sums["overall_dice"] / n,
        "overall_iou": metric_sums["overall_iou"] / n,
        "overall_pixel_acc": metric_sums["overall_pixel_acc"] / n,
        "structure_precision": metric_sums["structure_precision"] / n,
        "structure_recall": metric_sums["structure_recall"] / n,
        "structure_f1": metric_sums["structure_f1"] / n,
        "structure_dice": metric_sums["structure_dice"] / n,
        "structure_iou": metric_sums["structure_iou"] / n,
        "structure_pixel_acc": metric_sums["structure_pixel_acc"] / n,
        "furniture_precision": metric_sums["furniture_precision"] / n,
        "furniture_recall": metric_sums["furniture_recall"] / n,
        "furniture_f1": metric_sums["furniture_f1"] / n,
        "furniture_dice": metric_sums["furniture_dice"] / n,
        "furniture_iou": metric_sums["furniture_iou"] / n,
        "furniture_pixel_acc": metric_sums["furniture_pixel_acc"] / n,
    }


def plot_training_curves(hist, out_path: Path):
    epochs = [h["epoch"] for h in hist]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h["train_loss"] for h in hist], label="Train Loss")
    plt.plot(epochs, [h["val_loss"] for h in hist], label="Val Loss")
    plt.plot(epochs, [h["test_loss"] for h in hist], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_curves(hist, out_path: Path):
    epochs = [h["epoch"] for h in hist]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h["val_overall_dice"] for h in hist], label="Val Overall Dice")
    plt.plot(epochs, [h["val_overall_iou"] for h in hist], label="Val Overall IoU")
    plt.plot(epochs, [h["val_structure_dice"] for h in hist], label="Val Structure Dice")
    plt.plot(epochs, [h["val_furniture_dice"] for h in hist], label="Val Furniture Dice")
    plt.plot(epochs, [h["test_overall_dice"] for h in hist], label="Test Overall Dice")
    plt.plot(epochs, [h["test_overall_iou"] for h in hist], label="Test Overall IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args_dict: dict,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val,
        "train_args": args_dict,
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val = float(checkpoint.get("best_val_loss", float("inf")))
    return checkpoint, start_epoch, best_val


def evaluate_model(model, data_loader, device, threshold, split_name):
    model.eval()
    metric_sums = init_metric_sums()

    bar = tqdm(
        data_loader,
        desc=split_name,
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for batch in bar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits = model(x)

            bce = F.binary_cross_entropy_with_logits(logits, y)
            d_loss = dice_loss(logits, y)
            loss = bce + d_loss

            metric_dict = compute_segmentation_metrics_from_logits(
                logits, y, threshold=threshold
            )

            update_metric_sums(metric_sums, loss, bce, d_loss, metric_dict)

            running = finalize_metric_sums(metric_sums)
            bar.set_postfix({
                "loss": f"{running['loss']:.4f}",
                "dice": f"{running['overall_dice']:.4f}",
                "iou": f"{running['overall_iou']:.4f}",
            })

    return finalize_metric_sums(metric_sums)


def train_one_model(model_name, args, train_dl, val_dl, test_dl, device):
    out = Path(args.output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"TRAINING MODEL: {model_name}")
    print(f"Output dir: {out}")
    print("=" * 80)

    model = build_model(model_name, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    hist = []
    start_epoch = 0

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    hist = []
    start_epoch = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        checkpoint, start_epoch, best_val = load_checkpoint(
            ckpt_path,
            model,
            optimizer=opt,
            map_location=device,
        )
        print("[RESUME]")
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Start epoch: {start_epoch + 1}")
        print(f"Best val loss: {best_val:.6f}")
        print("=" * 80)

        history_path = out / "history.json"
        if history_path.exists():
            try:
                hist = json.loads(history_path.read_text(encoding="utf-8"))
                print(f"[RESUME] Loaded existing history from: {history_path}")
            except Exception as e:
                print(f"[RESUME][WARN] Failed to load history.json: {type(e).__name__}: {e}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sums = init_metric_sums()

        train_bar = tqdm(
            train_dl, 
            desc=f"train {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            leave=False,
            # ascii=True
        )

        for batch in train_bar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            opt.zero_grad()
            logits = model(x)

            bce = F.binary_cross_entropy_with_logits(logits, y)
            d_loss = dice_loss(logits, y)
            loss = bce + d_loss

            loss.backward()
            opt.step()

            metric_dict = compute_segmentation_metrics_from_logits(
                logits, y, threshold=args.threshold
            )
            update_metric_sums(train_sums, loss, bce, d_loss, metric_dict)

            running = finalize_metric_sums(train_sums)
            train_bar.set_postfix({
                "loss": f"{running['loss']:.4f}",
                "dice": f"{running['overall_dice']:.4f}",
                "iou": f"{running['overall_iou']:.4f}",
            })

        train_metrics = finalize_metric_sums(train_sums)

        val_metrics = evaluate_model(
            model=model,
            data_loader=val_dl,
            device=device,
            threshold=args.threshold,
            split_name=f"val   {epoch + 1}/{args.epochs}",
        )

        test_metrics = evaluate_model(
            model=model,
            data_loader=test_dl,
            device=device,
            threshold=args.threshold,
            split_name=f"test  {epoch + 1}/{args.epochs}",
        )

        hist_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_overall_precision": train_metrics["overall_precision"],
            "train_overall_recall": train_metrics["overall_recall"],
            "train_overall_f1": train_metrics["overall_f1"],
            "train_overall_dice": train_metrics["overall_dice"],
            "train_overall_iou": train_metrics["overall_iou"],
            "train_overall_pixel_acc": train_metrics["overall_pixel_acc"],
            "train_structure_dice": train_metrics["structure_dice"],
            "train_structure_iou": train_metrics["structure_iou"],
            "train_furniture_dice": train_metrics["furniture_dice"],
            "train_furniture_iou": train_metrics["furniture_iou"],

            "val_loss": val_metrics["loss"],
            "val_bce": val_metrics["bce"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_overall_precision": val_metrics["overall_precision"],
            "val_overall_recall": val_metrics["overall_recall"],
            "val_overall_f1": val_metrics["overall_f1"],
            "val_overall_dice": val_metrics["overall_dice"],
            "val_overall_iou": val_metrics["overall_iou"],
            "val_overall_pixel_acc": val_metrics["overall_pixel_acc"],
            "val_structure_precision": val_metrics["structure_precision"],
            "val_structure_recall": val_metrics["structure_recall"],
            "val_structure_f1": val_metrics["structure_f1"],
            "val_structure_dice": val_metrics["structure_dice"],
            "val_structure_iou": val_metrics["structure_iou"],
            "val_structure_pixel_acc": val_metrics["structure_pixel_acc"],
            "val_furniture_precision": val_metrics["furniture_precision"],
            "val_furniture_recall": val_metrics["furniture_recall"],
            "val_furniture_f1": val_metrics["furniture_f1"],
            "val_furniture_dice": val_metrics["furniture_dice"],
            "val_furniture_iou": val_metrics["furniture_iou"],
            "val_furniture_pixel_acc": val_metrics["furniture_pixel_acc"],

            "test_loss": test_metrics["loss"],
            "test_bce": test_metrics["bce"],
            "test_dice_loss": test_metrics["dice_loss"],
            "test_overall_precision": test_metrics["overall_precision"],
            "test_overall_recall": test_metrics["overall_recall"],
            "test_overall_f1": test_metrics["overall_f1"],
            "test_overall_dice": test_metrics["overall_dice"],
            "test_overall_iou": test_metrics["overall_iou"],
            "test_overall_pixel_acc": test_metrics["overall_pixel_acc"],
            "test_structure_precision": test_metrics["structure_precision"],
            "test_structure_recall": test_metrics["structure_recall"],
            "test_structure_f1": test_metrics["structure_f1"],
            "test_structure_dice": test_metrics["structure_dice"],
            "test_structure_iou": test_metrics["structure_iou"],
            "test_structure_pixel_acc": test_metrics["structure_pixel_acc"],
            "test_furniture_precision": test_metrics["furniture_precision"],
            "test_furniture_recall": test_metrics["furniture_recall"],
            "test_furniture_f1": test_metrics["furniture_f1"],
            "test_furniture_dice": test_metrics["furniture_dice"],
            "test_furniture_iou": test_metrics["furniture_iou"],
            "test_furniture_pixel_acc": test_metrics["furniture_pixel_acc"],
        }
        hist.append(hist_entry)

        print(
            f"[EPOCH {epoch + 1:02d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"test_loss={test_metrics['loss']:.6f} | "
            f"val_dice={val_metrics['overall_dice']:.4f} "
            f"test_dice={test_metrics['overall_dice']:.4f} | "
            f"val_iou={val_metrics['overall_iou']:.4f} "
            f"test_iou={test_metrics['overall_iou']:.4f} | "
            f"test_f1={test_metrics['overall_f1']:.4f}"
        )

        history_path = out / "history.json"
        history_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")

        last_ckpt_path = out / "last.pt"
        save_checkpoint(
            last_ckpt_path,
            model,
            opt,
            epoch=epoch,
            best_val=best_val,
            args_dict={**vars(args), "model": model_name},
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_ckpt_path = out / "best.pt"
            save_checkpoint(
                best_ckpt_path,
                model,
                opt,
                epoch=epoch,
                best_val=best_val,
                args_dict={**vars(args), "model": model_name},
            )
            print(f"[CHECKPOINT] Saved best model -> {best_ckpt_path}")

    summary = {
        "best_val_loss": best_val,
        "best_epoch_by_val_loss": int(np.argmin([h["val_loss"] for h in hist]) + 1),
        "final_epoch_metrics": hist[-1] if hist else {},
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_training_curves(hist, out / "loss_curves.png")
    plot_metric_curves(hist, out / "metric_curves.png")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best val loss: {best_val:.6f}")
    print(f"History saved: {out / 'history.json'}")
    print(f"Summary saved: {summary_path}")
    print(f"Loss curves saved: {out / 'loss_curves.png'}")
    print(f"Metric curves saved: {out / 'metric_curves.png'}")
    print(f"Last checkpoint saved: {out / 'last.pt'}")
    print(f"Best checkpoint saved: {out / 'best.pt'}")
    print("=" * 80)


def build_model(model_name: str, device: str):
    if model_name == "unet":
        return FloorplanUNet().to(device)
    elif model_name == "unetattention":
        return FloorplanUNetAttention().to(device)
    elif model_name == "unetplusplus":
        return FloorplanUNetPlusPlus().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--resume", type=str, default=None, help="Path to a saved checkpoint .pt file to continue training from")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["unet"],
        choices=["unet", "unetattention", "unetplusplus"],
        help="One or more model types to train"
    )
    args = ap.parse_args()

    print("=" * 80)
    print("TRAIN FLOORPLAN MODEL")
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Threshold: {args.threshold}")
    print(f"Output dir: {args.output_dir}")
    print(f"Resume checkpoint: {args.resume}")
    print(f"Using CNN architecture: {args.models}")
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

    n_total = len(ds)

    n_test = max(1, int(0.15 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Dataset too small for train/val/test split. Total samples: {n_total}"
        )

    print("[SPLIT]")
    print(f"Total: {n_total}")
    print(f"Train: {n_train}")
    print(f"Val: {n_val}")
    print(f"Test: {n_test}")
    print("=" * 80)

    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    if device == "cuda":
        print(f"[CUDA] Device count: {torch.cuda.device_count()}")
        print(f"[CUDA] Device name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    for model_name in args.models:
        train_one_model(
            model_name=model_name,
            args=args,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            device=device,
        )


if __name__ == "__main__":
    main()