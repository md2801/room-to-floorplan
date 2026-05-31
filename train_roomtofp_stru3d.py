from __future__ import annotations

"""
RoomToFP v8 fine-tuning script.

v8 improvements:
- Fine-tuning-first training flow:
    Use --resume best_ema.pt --resume-model-only to start safely from the best v7 EMA checkpoint.

- Stronger overfitting protection:
    1) Lower LR fine-tuning support.
    2) Stronger weight decay.
    3) Optional backbone freezing for the first N epochs.
    4) EMA decay tuning.
    5) Short patience / visual-score early stopping.
    6) Best-loss and best-visual checkpoints saved separately.

- Better generalisation:
    1) BEV Gaussian noise augmentation.
    2) RGB noise augmentation.
    3) BEV channel dropout.
    4) BEV pixel dropout on structural channels.
    5) Optional room class label smoothing.
    6) Optional room class weighting.

- Better checkpoint selection:
    v7 selected best checkpoint using validation loss only.
    v8 also saves best_visual.pt / best_visual_ema.pt using a composite score:
        room mask dice
        wall dice
        boundary dice
        opening dice
        corner heatmap dice
        coord_l1 penalty
        invalid polygon penalty

- Checkpoint averaging / model soup:
    1) --soup-only can average a list of checkpoints and exit.
    2) --init-soup-checkpoints can average several checkpoints and use that as the fine-tune starting point.
    3) --save-final-soup averages useful final EMA checkpoints after fine-tuning.

Expected dataset format:
    samples/*.npz created by prepare_roomtofp_dataset.py.
"""

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from room_to_floorplan.roomtofp_polygon_model import RoomToFPPolygonNet


ROOM_TYPES = [
    "undefined", "living room", "kitchen", "bedroom", "bathroom", "balcony",
    "corridor", "dining room", "study", "office", "storage", "other",
]


# Utility helpers
def ensure_chw_mask_np(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """
    v7/v8 fix:
    Normalise all mask arrays to C x H x W.

    This prevents dataloader batches like:
        B x 1 x 1 x H x W

    when the model expects:
        B x 1 x H x W
    """
    mask = np.asarray(mask, dtype=np.float32)

    if mask.ndim == 2:
        return mask[None, :, :].astype(np.float32)

    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask.transpose(2, 0, 1).astype(np.float32)

    if mask.ndim == 3:
        return mask.astype(np.float32)

    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    if mask.ndim == 2:
        mask = mask[None, :, :]

    if mask.ndim != 3:
        raise ValueError(f"{name} has unexpected shape after cleanup: {mask.shape}")

    return mask.astype(np.float32)


def _split_checkpoint_list(value: str) -> list[Path]:
    if not value:
        return []
    parts = [p.strip().strip('"').strip("'") for p in value.split(";") if p.strip()]
    return [Path(p) for p in parts]


def _checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint has no model_state_dict: {path}")
    return ckpt["model_state_dict"]


def average_checkpoints(checkpoint_paths: list[Path], output_path: Path) -> None:
    """
    v8 improvement:
    Average multiple compatible checkpoints into one model-soup checkpoint.

    Use this for nearby EMA checkpoints, for example:
        epoch_035_ema.pt;best_ema.pt;epoch_040_ema.pt

    Do not average heavily overfitted late checkpoints.
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints for checkpoint averaging.")

    checkpoint_paths = [Path(p) for p in checkpoint_paths]
    for p in checkpoint_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    print("[SOUP] averaging checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")
    first_state = first_ckpt["model_state_dict"]

    avg_state: dict[str, torch.Tensor] = {}
    keys = list(first_state.keys())

    for k in keys:
        v0 = first_state[k]

        if torch.is_floating_point(v0):
            acc = v0.float().clone()
            count = 1

            for path in checkpoint_paths[1:]:
                sd = _checkpoint_model_state(path)
                if k not in sd:
                    print(f"[SOUP] skip missing key in {path.name}: {k}")
                    continue
                v = sd[k]
                if tuple(v.shape) != tuple(v0.shape):
                    print(f"[SOUP] skip shape mismatch for key {k}: {tuple(v.shape)} vs {tuple(v0.shape)}")
                    continue
                acc += v.float()
                count += 1

            avg_state[k] = (acc / float(count)).to(dtype=v0.dtype)
        else:
            # Non-floating buffers copied from first checkpoint.
            avg_state[k] = v0.clone()

    payload = dict(first_ckpt)
    payload["model_state_dict"] = avg_state
    payload["model_version"] = "v8-model-soup"
    payload["soup_sources"] = [str(p) for p in checkpoint_paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[SOUP] saved averaged checkpoint to: {output_path}")


# Dataset + v8 augmentations
class RoomToFPNPZDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        augment: bool = True,
        bev_noise_std: float = 0.0,
        rgb_noise_std: float = 0.0,
        channel_dropout_p: float = 0.0,
        structural_pixel_dropout_p: float = 0.0,
    ):
        self.root = Path(dataset_root)
        self.split = str(split)
        self.augment = bool(augment and split == "train")

        # v8 improvement:
        # Non-geometric augmentation that does not require changing polygon labels.
        self.bev_noise_std = float(bev_noise_std)
        self.rgb_noise_std = float(rgb_noise_std)
        self.channel_dropout_p = float(channel_dropout_p)
        self.structural_pixel_dropout_p = float(structural_pixel_dropout_p)

        split_path = self.root / f"{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        self.names = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.sample_dir = self.root / "samples"
        if not self.sample_dir.exists():
            self.sample_dir = self.root

        if not self.names:
            raise ValueError(f"No samples listed in {split_path}")

    def __len__(self) -> int:
        return len(self.names)

    def _load(self, name: str) -> dict[str, np.ndarray]:
        path = self.sample_dir / name
        if not path.exists():
            path = self.root / name

        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    @staticmethod
    def _flip_x(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        s = {k: np.array(v, copy=True) for k, v in sample.items()}

        s["observation"] = s["observation"][:, ::-1, :].copy()

        for key in [
            "wall_mask", "opening_mask", "furniture_mask",
            "boundary_mask", "corner_heatmap", "room_mask",
        ]:
            if key in s:
                s[key] = np.flip(s[key], axis=-1).copy()

        if "polygon_xy" in s:
            s["polygon_xy"][..., 0] = 1.0 - s["polygon_xy"][..., 0]

        return s

    @staticmethod
    def _flip_y(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        s = {k: np.array(v, copy=True) for k, v in sample.items()}

        s["observation"] = s["observation"][::-1, :, :].copy()

        for key in [
            "wall_mask", "opening_mask", "furniture_mask",
            "boundary_mask", "corner_heatmap", "room_mask",
        ]:
            if key in s:
                s[key] = np.flip(s[key], axis=-2).copy()

        if "polygon_xy" in s:
            s["polygon_xy"][..., 1] = 1.0 - s["polygon_xy"][..., 1]

        return s

    def _augment_observation_v8(self, obs: np.ndarray) -> np.ndarray:
        """
        v8 improvement:
        Add label-safe BEV/input augmentations.

        These augmentations help prevent memorising exact BEV textures:
        - Gaussian noise on non-RGB BEV channels.
        - Gaussian noise on RGB channels.
        - Whole-channel dropout for structural channels.
        - Pixel dropout for structural channels.

        Channel order expected:
            0 occupancy
            1 wall
            2 floor
            3 opening
            4 furniture
            5-7 RGB
        """
        obs = obs.astype(np.float32, copy=True)
        h, w, c = obs.shape

        structural_c = min(c, 5)

        if self.channel_dropout_p > 0 and structural_c > 0:
            for ch in range(structural_c):
                if random.random() < self.channel_dropout_p:
                    obs[..., ch] = 0.0

        if self.structural_pixel_dropout_p > 0 and structural_c > 0:
            keep = (
                np.random.rand(h, w, 1).astype(np.float32)
                >= float(self.structural_pixel_dropout_p)
            )
            obs[..., :structural_c] *= keep

        if self.bev_noise_std > 0 and structural_c > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=float(self.bev_noise_std),
                size=obs[..., :structural_c].shape,
            ).astype(np.float32)
            obs[..., :structural_c] = obs[..., :structural_c] + noise

        if self.rgb_noise_std > 0 and c >= 8:
            noise = np.random.normal(
                loc=0.0,
                scale=float(self.rgb_noise_std),
                size=obs[..., 5:8].shape,
            ).astype(np.float32)
            obs[..., 5:8] = obs[..., 5:8] + noise

        obs = np.clip(obs, 0.0, 1.0)
        return obs.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._load(self.names[idx])

        if self.augment:
            if random.random() < 0.5:
                s = self._flip_x(s)
            if random.random() < 0.5:
                s = self._flip_y(s)

        obs = s["observation"].astype(np.float32)

        if self.augment:
            obs = self._augment_observation_v8(obs)

        wall_mask = ensure_chw_mask_np(np.clip(s["wall_mask"], 0.0, 1.0), "wall_mask")
        opening_mask = ensure_chw_mask_np(np.clip(s["opening_mask"], 0.0, 1.0), "opening_mask")
        boundary_mask = ensure_chw_mask_np(np.clip(s["boundary_mask"], 0.0, 1.0), "boundary_mask")
        corner_heatmap = ensure_chw_mask_np(np.clip(s["corner_heatmap"], 0.0, 1.0), "corner_heatmap")
        room_mask = ensure_chw_mask_np(np.clip(s["room_mask"], 0.0, 1.0), "room_mask")

        if "furniture_mask" in s:
            furniture_mask = ensure_chw_mask_np(np.clip(s["furniture_mask"], 0.0, 1.0), "furniture_mask")
        else:
            furniture_mask = np.zeros_like(wall_mask, dtype=np.float32)

        out: dict[str, torch.Tensor] = {
            "x": torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32)).float(),
            "polygon_xy": torch.from_numpy(s["polygon_xy"].astype(np.float32)).float(),
            "corner_valid": torch.from_numpy(s["corner_valid"].astype(np.float32)).float(),
            "room_valid": torch.from_numpy(s["room_valid"].astype(np.float32)).float(),
            "room_class": torch.from_numpy(s["room_class"].astype(np.int64)).long(),
            "wall_mask": torch.from_numpy(wall_mask).float(),
            "opening_mask": torch.from_numpy(opening_mask).float(),
            "room_mask": torch.from_numpy(room_mask).float(),
            "boundary_mask": torch.from_numpy(boundary_mask).float(),
            "corner_heatmap": torch.from_numpy(corner_heatmap).float(),
            "furniture_mask": torch.from_numpy(furniture_mask).float(),
        }

        return out


def compute_room_class_weights(
    dataset: RoomToFPNPZDataset,
    num_classes: int,
    mode: str = "none",
    max_weight: float = 4.0,
) -> torch.Tensor | None:
    """
    v8 improvement:
    Optional room-class weighting.

    Helps prevent common classes from dominating room type prediction.
    Use cautiously. Recommended:
        --class-weighting inv_sqrt
    """
    mode = str(mode).lower().strip()

    if mode == "none":
        return None

    counts = np.zeros((num_classes,), dtype=np.float64)

    for name in tqdm(dataset.names, desc="[CLASS WEIGHTS] scanning train split"):
        s = dataset._load(name)
        rv = np.asarray(s["room_valid"]).reshape(-1)
        rc = np.asarray(s["room_class"]).reshape(-1)

        for valid, cls in zip(rv, rc):
            if float(valid) > 0.5:
                cls_int = int(cls)
                if 0 <= cls_int < num_classes:
                    counts[cls_int] += 1.0

    counts = np.maximum(counts, 1.0)

    if mode == "inverse":
        weights = counts.sum() / counts
    elif mode == "inv_sqrt":
        weights = np.sqrt(counts.sum() / counts)
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")

    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.25, float(max_weight))

    print("[CLASS WEIGHTS]")
    for i, (name, count, weight) in enumerate(zip(ROOM_TYPES, counts, weights)):
        print(f"  {i:02d} {name:12s} count={int(count):6d} weight={weight:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# Loss helpers
@dataclass
class LossConfig:
    mask_loss: str = "focal_dice"
    dice_weight: float = 1.0
    focal_gamma: float = 2.0

    coord_weight: float = 10.0
    polygon_edge_weight: float = 1.5
    polygon_area_weight: float = 0.50
    polygon_angle_weight: float = 0.35
    polygon_raster_weight: float = 1.0
    polygon_raster_size: int = 64

    aux_decoder_weight: float = 0.35
    dn_weight: float = 0.50
    dn_noise: float = 0.07

    room_overlap_weight: float = 0.10
    aux_room_mask: float = 0.6
    aux_wall: float = 0.6
    aux_opening: float = 0.8
    aux_boundary: float = 0.35
    aux_corner_heatmap: float = 0.25
    aux_furniture: float = 0.0
    boundary_weight: float = 0.25

    match_class_weight: float = 0.20
    match_room_conf_weight: float = 0.20

    use_hungarian: bool = True
    use_cyclic: bool = True
    room_pos_weight_max: float = 4.0

    # v8 improvement:
    # Regularise room type predictions.
    class_label_smoothing: float = 0.0
    room_class_weights: torch.Tensor | None = None


def align_target_to_logits(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    target = target.float().clamp(0.0, 1.0)

    while target.dim() > logits.dim():
        squeezed = False
        for dim in range(1, target.dim()):
            if target.shape[dim] == 1:
                target = target.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            break

    if target.dim() == logits.dim() - 1:
        target = target.unsqueeze(1)

    while target.dim() < logits.dim():
        target = target.unsqueeze(1)

    if target.shape != logits.shape:
        try:
            target = target.expand_as(logits)
        except RuntimeError as e:
            raise ValueError(
                f"Could not align target shape {tuple(target.shape)} "
                f"to logits shape {tuple(logits.shape)}"
            ) from e

    return target


def dice_score_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    target = align_target_to_logits(target, logits)
    pred = (torch.sigmoid(logits) >= threshold).float()

    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()

    if float(denom.detach().cpu()) <= 0:
        return 1.0

    return float(((2 * inter + eps) / (denom + eps)).detach().cpu())


def soft_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = align_target_to_logits(target, logits)
    prob = torch.sigmoid(logits)

    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    target = align_target_to_logits(target, logits)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    focal_weight = (1.0 - pt).pow(float(gamma))

    return (focal_weight * bce).mean()


def mask_loss(logits: torch.Tensor, target: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    target = align_target_to_logits(target, logits)

    if cfg.mask_loss == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)

    if cfg.mask_loss == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss_from_logits(logits, target)
        return bce + cfg.dice_weight * dice

    if cfg.mask_loss == "focal_dice":
        focal = focal_bce_with_logits(logits, target, cfg.focal_gamma)
        dice = soft_dice_loss_from_logits(logits, target)
        return focal + cfg.dice_weight * dice

    raise ValueError(f"Unknown mask loss: {cfg.mask_loss}")


def class_ce_loss(logits: torch.Tensor, target: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """
    v8 improvement:
    Centralised class CE with optional label smoothing and class weights.
    """
    weight = cfg.room_class_weights
    if weight is not None:
        weight = weight.to(device=logits.device, dtype=logits.dtype)

    return F.cross_entropy(
        logits,
        target,
        weight=weight,
        label_smoothing=float(cfg.class_label_smoothing),
    )


def polygon_area_norm(poly: torch.Tensor) -> torch.Tensor:
    x = poly[..., 0]
    y = poly[..., 1]
    x2 = torch.roll(x, shifts=-1, dims=-1)
    y2 = torch.roll(y, shifts=-1, dims=-1)
    return 0.5 * torch.abs((x * y2 - y * x2).sum(dim=-1))


def edge_lengths(poly: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(torch.roll(poly, shifts=-1, dims=-2) - poly, dim=-1)


def manhattan_angle_loss(poly: torch.Tensor) -> torch.Tensor:
    e = torch.roll(poly, shifts=-1, dims=-2) - poly
    dx = e[..., 0].abs()
    dy = e[..., 1].abs()
    length = torch.sqrt(dx * dx + dy * dy + 1e-8)

    # Horizontal/vertical edges have min(dx,dy)/length near 0.
    return (torch.minimum(dx, dy) / length).mean()


def cyclic_align_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    use_cyclic: bool = True,
) -> torch.Tensor:
    if not use_cyclic:
        return target

    k = int(valid.sum().item())
    if k <= 2:
        return target

    p = pred[:k]
    t = target[:k]

    candidates = []
    for shift in range(k):
        candidates.append(torch.roll(t, shifts=shift, dims=0))

    tr = torch.flip(t, dims=[0])
    for shift in range(k):
        candidates.append(torch.roll(tr, shifts=shift, dims=0))

    costs = torch.stack([torch.abs(p - c).mean() for c in candidates])
    best = candidates[int(torch.argmin(costs).item())]

    out = target.clone()
    out[:k] = best

    return out


def pair_cost(
    pred_xy: torch.Tensor,
    tgt_xy: torch.Tensor,
    tgt_valid: torch.Tensor,
    use_cyclic: bool,
) -> float:
    aligned = cyclic_align_target(
        pred_xy.detach(),
        tgt_xy.detach(),
        tgt_valid.detach(),
        use_cyclic,
    )

    v = tgt_valid.bool()
    if v.sum() == 0:
        return 1e3

    return float(torch.abs(pred_xy[v] - aligned[v]).mean().detach().cpu())


def hungarian_match(pred: dict, batch: dict, cfg: LossConfig) -> list[list[tuple[int, int]]]:
    b, m, n, _ = pred["corner_xy"].shape
    matches: list[list[tuple[int, int]]] = []

    room_prob = torch.sigmoid(pred["room_valid_logits"]).detach()
    class_logits = pred["room_class_logits"].detach()

    class_weight = cfg.room_class_weights
    if class_weight is not None:
        class_weight = class_weight.to(device=class_logits.device, dtype=class_logits.dtype)

    for bi in range(b):
        gt_ids = torch.where(batch["room_valid"][bi] > 0.5)[0].tolist()

        if not gt_ids:
            matches.append([])
            continue

        cost = np.zeros((m, len(gt_ids)), dtype=np.float32)

        for pi in range(m):
            for gj, gi in enumerate(gt_ids):
                c = pair_cost(
                    pred["corner_xy"][bi, pi],
                    batch["polygon_xy"][bi, gi],
                    batch["corner_valid"][bi, gi],
                    cfg.use_cyclic,
                )

                cls = int(batch["room_class"][bi, gi].item())
                ce = float(
                    F.cross_entropy(
                        class_logits[bi, pi].unsqueeze(0),
                        torch.tensor([cls], device=class_logits.device),
                        weight=class_weight,
                        reduction="mean",
                    ).detach().cpu()
                )

                conf = float((1.0 - room_prob[bi, pi]).detach().cpu())
                cost[pi, gj] = c + cfg.match_class_weight * ce + cfg.match_room_conf_weight * conf

        if cfg.use_hungarian and linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            matches.append([(int(r), int(gt_ids[c])) for r, c in zip(rows, cols)])
        else:
            pairs = []
            used_p, used_g = set(), set()

            flat = [
                (cost[i, j], i, j)
                for i in range(m)
                for j in range(len(gt_ids))
            ]

            for _, i, j in sorted(flat, key=lambda x: x[0]):
                if i in used_p or j in used_g:
                    continue
                used_p.add(i)
                used_g.add(j)
                pairs.append((i, gt_ids[j]))

            matches.append(pairs)

    return matches


def polygon_raster_loss(
    pred_poly: torch.Tensor,
    tgt_mask: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """
    v7/v8:
    Differentiable approximation of polygon-vs-mask overlap.

    This is not an exact rasterizer, but gives gradients to coordinates.
    """
    s = int(size)

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, s, device=pred_poly.device, dtype=pred_poly.dtype),
        torch.linspace(0, 1, s, device=pred_poly.device, dtype=pred_poly.dtype),
        indexing="ij",
    )

    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)

    v0 = pred_poly
    v1 = torch.roll(pred_poly, shifts=-1, dims=0)
    edges = v1 - v0

    gp = grid[:, None, :] - v0[None, :, :]
    t = (gp * edges[None]).sum(-1) / (edges.pow(2).sum(-1)[None] + 1e-6)
    t = t.clamp(0, 1)

    proj = v0[None] + t[..., None] * edges[None]
    dist = torch.linalg.norm(grid[:, None, :] - proj, dim=-1).min(dim=1).values
    boundary_prob = torch.exp(-dist * 80.0).view(s, s)

    mn = pred_poly.min(dim=0).values
    mx = pred_poly.max(dim=0).values

    inside_x = torch.sigmoid((xx - mn[0]) * 80) * torch.sigmoid((mx[0] - xx) * 80)
    inside_y = torch.sigmoid((yy - mn[1]) * 80) * torch.sigmoid((mx[1] - yy) * 80)

    pred_mask = (inside_x * inside_y).clamp(0, 1)
    pred_mask = torch.maximum(pred_mask, boundary_prob * 0.25)

    tgt = F.interpolate(
        tgt_mask[None, None].float(),
        size=(s, s),
        mode="nearest",
    )[0, 0]

    inter = (pred_mask * tgt).sum()
    dice = (2 * inter + 1e-6) / (pred_mask.sum() + tgt.sum() + 1e-6)

    return 1.0 - dice


def make_denoising_inputs(
    batch: dict,
    noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xy = batch["polygon_xy"].clone()
    cv = batch["corner_valid"].clone()
    rv = batch["room_valid"].clone()

    noise_tensor = torch.randn_like(xy) * float(noise)
    xy_noisy = (xy + noise_tensor * cv.unsqueeze(-1)).clamp(0.0, 1.0)

    return xy_noisy, cv, rv


def compute_losses(
    pred: dict,
    batch: dict,
    cfg: LossConfig,
    include_aux: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = pred["corner_xy"].device
    matches = hungarian_match(pred, batch, cfg)

    room_targets = torch.zeros_like(pred["room_valid_logits"])

    class_losses: list[torch.Tensor] = []
    coord_losses: list[torch.Tensor] = []
    corner_bce_losses: list[torch.Tensor] = []
    edge_losses: list[torch.Tensor] = []
    area_losses: list[torch.Tensor] = []
    angle_losses: list[torch.Tensor] = []
    raster_losses: list[torch.Tensor] = []
    room_mask_losses: list[torch.Tensor] = []

    invalid_count = 0
    total_poly = 0

    for bi, pairs in enumerate(matches):
        for pi, gi in pairs:
            room_targets[bi, pi] = 1.0

            tgt_valid = batch["corner_valid"][bi, gi]
            pred_xy = pred["corner_xy"][bi, pi]
            tgt_xy = cyclic_align_target(
                pred_xy,
                batch["polygon_xy"][bi, gi],
                tgt_valid,
                cfg.use_cyclic,
            )

            valid = tgt_valid.bool()

            if valid.sum() >= 3:
                class_losses.append(
                    class_ce_loss(
                        pred["room_class_logits"][bi, pi].unsqueeze(0),
                        batch["room_class"][bi, gi].unsqueeze(0),
                        cfg,
                    )
                )

                coord_losses.append(torch.abs(pred_xy[valid] - tgt_xy[valid]).mean())
                corner_bce_losses.append(
                    F.binary_cross_entropy_with_logits(
                        pred["corner_valid_logits"][bi, pi],
                        tgt_valid,
                    )
                )

                ppoly = pred_xy[valid]
                tpoly = tgt_xy[valid]

                edge_losses.append(torch.abs(edge_lengths(ppoly) - edge_lengths(tpoly)).mean())
                area_losses.append(torch.abs(polygon_area_norm(ppoly) - polygon_area_norm(tpoly)))
                angle_losses.append(manhattan_angle_loss(ppoly))
                raster_losses.append(
                    polygon_raster_loss(ppoly, batch["room_mask"][bi, gi], cfg.polygon_raster_size)
                )

                if "room_mask_logits" in pred:
                    room_mask_losses.append(
                        mask_loss(
                            pred["room_mask_logits"][bi, pi].unsqueeze(0).unsqueeze(0),
                            batch["room_mask"][bi, gi].unsqueeze(0).unsqueeze(0),
                            cfg,
                        )
                    )

                total_poly += 1

                if float(polygon_area_norm(ppoly).detach().cpu()) < 1e-4:
                    invalid_count += 1

    pos = room_targets.sum()
    neg = room_targets.numel() - pos
    pos_weight_val = float((neg / pos.clamp_min(1.0)).detach().cpu()) if pos > 0 else 1.0
    pos_weight_val = min(pos_weight_val, cfg.room_pos_weight_max)

    room_bce = F.binary_cross_entropy_with_logits(
        pred["room_valid_logits"],
        room_targets,
        pos_weight=torch.tensor(pos_weight_val, device=device),
    )

    def mean_or_zero(xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(xs).mean() if xs else torch.zeros((), device=device)

    coord_l = mean_or_zero(coord_losses)
    class_l = mean_or_zero(class_losses)
    corner_bce = mean_or_zero(corner_bce_losses)
    edge_l = mean_or_zero(edge_losses)
    area_l = mean_or_zero(area_losses)
    angle_l = mean_or_zero(angle_losses)
    raster_l = mean_or_zero(raster_losses)
    room_mask_l = mean_or_zero(room_mask_losses)

    wall_l = mask_loss(pred["wall_logits"], batch["wall_mask"], cfg)
    opening_l = mask_loss(pred["opening_logits"], batch["opening_mask"], cfg)
    boundary_l = mask_loss(pred["boundary_logits"], batch["boundary_mask"], cfg)
    corner_heat_l = mask_loss(pred["corner_heatmap_logits"], batch["corner_heatmap"], cfg)

    furniture_l = torch.zeros((), device=device)
    if "furniture_logits" in pred and cfg.aux_furniture > 0:
        furniture_l = mask_loss(pred["furniture_logits"], batch["furniture_mask"], cfg)

    loss = (
        room_bce
        + corner_bce
        + class_l
        + cfg.coord_weight * coord_l
        + cfg.polygon_edge_weight * edge_l
        + cfg.polygon_area_weight * area_l
        + cfg.polygon_angle_weight * angle_l
        + cfg.polygon_raster_weight * raster_l
        + cfg.aux_room_mask * room_mask_l
        + cfg.aux_wall * wall_l
        + cfg.aux_opening * opening_l
        + cfg.aux_boundary * boundary_l
        + cfg.aux_corner_heatmap * corner_heat_l
        + cfg.aux_furniture * furniture_l
    )

    aux_l_total = torch.zeros((), device=device)

    if include_aux and cfg.aux_decoder_weight > 0 and pred.get("aux_outputs"):
        aux_shared = {
            k: pred[k]
            for k in [
                "wall_logits",
                "opening_logits",
                "boundary_logits",
                "corner_heatmap_logits",
                "room_mask_logits",
            ]
            if k in pred
        }

        for aux in pred["aux_outputs"]:
            aux_loss, _ = compute_losses(aux | aux_shared, batch, cfg, include_aux=False)
            aux_l_total = aux_l_total + aux_loss

        aux_l_total = aux_l_total / max(1, len(pred["aux_outputs"]))
        loss = loss + cfg.aux_decoder_weight * aux_l_total

    dn_l = torch.zeros((), device=device)

    if cfg.dn_weight > 0 and "dn_outputs" in pred:
        dn = pred["dn_outputs"]
        rv = batch["room_valid"]
        cv = batch["corner_valid"]

        dn_room_bce = F.binary_cross_entropy_with_logits(dn["room_valid_logits"], rv)
        dn_corner_bce = F.binary_cross_entropy_with_logits(dn["corner_valid_logits"], cv)

        valid = (rv.unsqueeze(-1) * cv).bool()
        dn_coord = (
            torch.abs(dn["corner_xy"] - batch["polygon_xy"])[valid].mean()
            if valid.any()
            else torch.zeros((), device=device)
        )

        cls_mask = rv.bool()
        dn_cls = (
            class_ce_loss(dn["room_class_logits"][cls_mask], batch["room_class"][cls_mask], cfg)
            if cls_mask.any()
            else torch.zeros((), device=device)
        )

        dn_l = dn_room_bce + dn_corner_bce + cfg.coord_weight * dn_coord + dn_cls
        loss = loss + cfg.dn_weight * dn_l

    parts = {
        "loss": float(loss.detach().cpu()),
        "coord_l1": float(coord_l.detach().cpu()),
        "class_loss": float(class_l.detach().cpu()),
        "corner_bce": float(corner_bce.detach().cpu()),
        "edge_loss": float(edge_l.detach().cpu()),
        "area_loss": float(area_l.detach().cpu()),
        "angle_loss": float(angle_l.detach().cpu()),
        "raster_loss": float(raster_l.detach().cpu()),
        "wall_dice": dice_score_from_logits(pred["wall_logits"], batch["wall_mask"]),
        "opening_dice": dice_score_from_logits(pred["opening_logits"], batch["opening_mask"]),
        "boundary_dice": dice_score_from_logits(pred["boundary_logits"], batch["boundary_mask"]),
        "corner_heatmap_dice": dice_score_from_logits(pred["corner_heatmap_logits"], batch["corner_heatmap"]),
        "matched_room_mask_dice": 1.0,
        "invalid_polygon_rate": float(invalid_count / max(1, total_poly)),
        "aux_decoder_loss": float(aux_l_total.detach().cpu()),
        "dn_loss": float(dn_l.detach().cpu()),
    }

    if "room_mask_logits" in pred:
        vals = []
        for bi, pairs in enumerate(matches):
            for pi, gi in pairs:
                vals.append(
                    dice_score_from_logits(
                        pred["room_mask_logits"][bi, pi].unsqueeze(0).unsqueeze(0),
                        batch["room_mask"][bi, gi].unsqueeze(0).unsqueeze(0),
                    )
                )
        if vals:
            parts["matched_room_mask_dice"] = float(np.mean(vals))

    return loss, parts


# EMA
class ModelEMA:
    """
    v7/v8:
    Exponential Moving Average checkpointing.

    EMA usually produces cleaner inference for transformer/polygon models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()

        for k, v in state.items():
            if not torch.is_floating_point(v):
                continue

            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue

            if self.shadow[k].device != v.device or self.shadow[k].dtype != v.dtype:
                self.shadow[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)

            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {
            k: v.detach().clone()
            for k, v in state.items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model_state = model.state_dict()

        for k, v in self.shadow.items():
            if k not in model_state:
                continue

            model_tensor = model_state[k]
            if not torch.is_floating_point(model_tensor):
                continue

            model_tensor.copy_(v.to(device=model_tensor.device, dtype=model_tensor.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        model_state = model.state_dict()

        self.backup = {
            k: v.detach().clone()
            for k, v in model_state.items()
            if k in self.shadow and torch.is_floating_point(v)
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        model_state = model.state_dict()

        for k, v in self.backup.items():
            if k in model_state:
                model_state[k].copy_(v.to(device=model_state[k].device, dtype=model_state[k].dtype))

        self.backup = {}


# Training helpers
def freeze_backbone_v8(model: nn.Module, freeze: bool = True) -> None:
    """
    v8 improvement:
    Optional freeze for backbone/encoder early in fine-tuning.

    This helps avoid destroying the already-learned v7 representation when
    starting from best_ema.pt.
    """
    prefixes = ("backbone", "encoder")

    n = 0
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = not freeze
            n += 1

    print(f"[FREEZE] freeze_backbone={freeze} affected_params={n}")


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def average_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}

    keys = sorted(set().union(*[x.keys() for x in items]))

    return {
        k: float(np.mean([x[k] for x in items if k in x]))
        for k in keys
    }


def compute_visual_score(metrics: dict[str, float]) -> float:
    """
    v8 improvement:
    Composite visual/generalisation score for checkpoint selection.

    Higher is better.

    This is not a replacement for validation loss. It is an additional checkpoint
    criterion because lower total loss does not always mean better floorplan output.
    """
    coord = float(metrics.get("coord_l1", 1.0))
    invalid = float(metrics.get("invalid_polygon_rate", 0.0))

    score = (
        1.00 * float(metrics.get("matched_room_mask_dice", 0.0))
        + 1.00 * float(metrics.get("wall_dice", 0.0))
        + 0.50 * float(metrics.get("boundary_dice", 0.0))
        + 0.30 * float(metrics.get("opening_dice", 0.0))
        + 0.20 * float(metrics.get("corner_heatmap_dice", 0.0))
        - 5.00 * coord
        - 0.50 * invalid
    )

    return float(score)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    cfg: LossConfig,
    device: str,
    train: bool,
    amp: bool,
    accum_steps: int = 1,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train(train)
    all_parts: list[dict[str, float]] = []

    pbar = tqdm(loader, desc="train" if train else "val")

    if train:
        optimizer.zero_grad(set_to_none=True)

    accum_steps = max(1, int(accum_steps))

    for step, batch in enumerate(pbar, start=1):
        batch = move_batch(batch, device)

        dn_xy = dn_cv = dn_rv = None
        if train and cfg.dn_weight > 0:
            dn_xy, dn_cv, dn_rv = make_denoising_inputs(batch, cfg.dn_noise)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(
                    batch["x"],
                    dn_polygon_xy=dn_xy,
                    dn_corner_valid=dn_cv,
                    dn_room_valid=dn_rv,
                )
                loss, parts = compute_losses(pred, batch, cfg, include_aux=True)
                loss_to_backprop = loss / accum_steps

            if train:
                do_step = (step % accum_steps == 0) or (step == len(loader))

                if scaler is not None:
                    scaler.scale(loss_to_backprop).backward()

                    if do_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        if ema is not None:
                            ema.update(model)
                else:
                    loss_to_backprop.backward()

                    if do_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        if ema is not None:
                            ema.update(model)

        all_parts.append(parts)
        running = average_dict(all_parts)
        visual = compute_visual_score(running)

        pbar.set_postfix({
            "loss": f"{running.get('loss', 0.0):.2f}",
            "coord": f"{running.get('coord_l1', 0.0):.3f}",
            "wdice": f"{running.get('wall_dice', 0.0):.3f}",
            "odice": f"{running.get('opening_dice', 0.0):.3f}",
            "rmdice": f"{running.get('matched_room_mask_dice', 0.0):.3f}",
            "vis": f"{visual:.3f}",
        })

    final = average_dict(all_parts)
    final["visual_score"] = compute_visual_score(final)
    return final


# Checkpoint helpers
def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val: float,
    best_epoch: int,
    best_visual: float,
    best_visual_epoch: int,
    args,
    history,
    ema: ModelEMA | None = None,
    model_version: str = "v8",
):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "best_visual": float(best_visual),
        "best_visual_epoch": int(best_visual_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "history": history,
        "model_version": model_version,
    }

    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()

    torch.save(payload, path)


def save_ema_model_checkpoint(
    path: Path,
    model,
    ema: ModelEMA,
    optimizer,
    scheduler,
    epoch: int,
    best_val: float,
    best_epoch: int,
    best_visual: float,
    best_visual_epoch: int,
    args,
    history,
    model_version: str = "v8-ema",
):
    """
    Save a normal checkpoint whose model_state_dict is the EMA-smoothed weights,
    so inference works unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_state = copy.deepcopy(model.state_dict())

    ema.store(model)
    ema.copy_to(model)

    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "best_visual": float(best_visual),
        "best_visual_epoch": int(best_visual_epoch),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "history": history,
        "model_version": model_version,
    }

    model.load_state_dict(raw_state)
    ema.restore(model)

    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    model_only: bool = False,
    ema: ModelEMA | None = None,
):
    ckpt = torch.load(path, map_location="cpu")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print(f"[RESUME] loaded {path}")

    if missing:
        print(f"[RESUME] missing keys: {len(missing)}")
    if unexpected:
        print(f"[RESUME] unexpected keys: {len(unexpected)}")

    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])
        print("[RESUME] loaded EMA state")

    if model_only:
        return 0, float("inf"), 0, -float("inf"), 0, []

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return (
        int(ckpt.get("epoch", 0)),
        float(ckpt.get("best_val", float("inf"))),
        int(ckpt.get("best_epoch", 0)),
        float(ckpt.get("best_visual", -float("inf"))),
        int(ckpt.get("best_visual_epoch", 0)),
        ckpt.get("history", []),
    )


def run_fixed_scene_monitoring(
    infer_script: str,
    scene_dirs: str,
    checkpoint: Path,
    output_root: Path,
    epoch: int,
    device: str,
    grid_size: int,
    variants: str,
    room_threshold: float,
    corner_threshold: float,
) -> None:
    """
    v7/v8:
    Fixed-scene visual monitoring.

    This is useful because scalar val loss may not perfectly match visual quality.
    """
    if not scene_dirs:
        return

    script = Path(infer_script)
    if not script.exists():
        print(f"[MONITOR] skipped: inference script not found: {script}")
        return

    scenes = [s.strip() for s in scene_dirs.split(";") if s.strip()]

    for scene in scenes:
        scene_path = Path(scene)
        scene_name = scene_path.name
        out_dir = output_root / f"epoch_{epoch:03d}" / scene_name

        cmd = [
            sys.executable,
            str(script),
            "--scene-dir", str(scene_path),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(out_dir),
            "--variants", variants,
            "--grid-size", str(grid_size),
            "--device", device,
            "--room-threshold", str(room_threshold),
            "--corner-threshold", str(corner_threshold),
            "--simplify-eps-px", "2",
            "--snap-axis-threshold-px", "8",
            "--opening-snap-distance-px", "22",
            "--min-opening-area-px", "18",
            "--min-opening-length-px", "20",
        ]

        print("[MONITOR]", " ".join(cmd))

        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[MONITOR] failed for {scene}: {e}")


def save_final_soup_if_requested(out: Path, args) -> None:
    """
    v8 improvement:
    Save a final averaged EMA checkpoint using the best available final EMA checkpoints.
    """
    if not args.save_final_soup:
        return

    candidates = [
        out / "best_ema.pt",
        out / "best_visual_ema.pt",
        out / "last_ema.pt",
    ]

    existing = []
    seen = set()

    for p in candidates:
        if p.exists() and str(p.resolve()) not in seen:
            existing.append(p)
            seen.add(str(p.resolve()))

    if len(existing) < 2:
        print("[SOUP] skipped final soup: fewer than 2 EMA checkpoints available.")
        return

    average_checkpoints(existing, out / "final_soup_ema.pt")


# Main
def main() -> None:
    ap = argparse.ArgumentParser(description="Train/fine-tune RoomToFP v8 polygon model")

    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--mask-dim", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--decoder-layers", type=int, default=4)
    ap.add_argument("--corner-decoder-layers", type=int, default=3)
    ap.add_argument("--transformer-encoder-layers", type=int, default=2)
    ap.add_argument("--deform-points", type=int, default=4)
    ap.add_argument("--max-rooms", type=int, default=20)
    ap.add_argument("--max-corners", type=int, default=16)
    ap.add_argument("--room-classes", type=int, default=len(ROOM_TYPES))
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--predict-furniture", action="store_true")

    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--warmup-epochs", type=int, default=1)

    # v8 regularisation augmentations
    ap.add_argument("--bev-noise-std", type=float, default=0.015)
    ap.add_argument("--rgb-noise-std", type=float, default=0.010)
    ap.add_argument("--channel-dropout-p", type=float, default=0.03)
    ap.add_argument("--structural-pixel-dropout-p", type=float, default=0.01)

    # v8 fine-tuning safety
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.9995)

    ap.add_argument("--fixed-scene-dirs", default="")
    ap.add_argument("--fixed-scene-every", type=int, default=0)
    ap.add_argument("--infer-script", default="infer_roomtofp_from_scene_v7_4.py")
    ap.add_argument("--monitor-room-threshold", type=float, default=0.35)
    ap.add_argument("--monitor-corner-threshold", type=float, default=0.45)

    # Loss config
    ap.add_argument("--mask-loss", choices=["bce", "bce_dice", "focal_dice"], default="focal_dice")
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--focal-gamma", type=float, default=2.0)

    ap.add_argument("--coord-weight", type=float, default=8.0)
    ap.add_argument("--polygon-edge-weight", type=float, default=2.0)
    ap.add_argument("--polygon-area-weight", type=float, default=0.35)
    ap.add_argument("--polygon-angle-weight", type=float, default=0.60)
    ap.add_argument("--polygon-raster-weight", type=float, default=0.75)
    ap.add_argument("--polygon-raster-size", type=int, default=64)

    ap.add_argument("--aux-decoder-weight", type=float, default=0.35)
    ap.add_argument("--dn-weight", type=float, default=0.25)
    ap.add_argument("--dn-noise", type=float, default=0.04)

    ap.add_argument("--room-overlap-weight", type=float, default=0.10)
    ap.add_argument("--aux-room-mask", type=float, default=0.6)
    ap.add_argument("--aux-wall", type=float, default=0.7)
    ap.add_argument("--aux-opening", type=float, default=0.5)
    ap.add_argument("--aux-boundary", type=float, default=0.45)
    ap.add_argument("--aux-corner-heatmap", type=float, default=0.35)
    ap.add_argument("--aux-furniture", type=float, default=0.0)
    ap.add_argument("--boundary-weight", type=float, default=0.25)

    ap.add_argument("--match-class-weight", type=float, default=0.20)
    ap.add_argument("--match-room-conf-weight", type=float, default=0.20)
    ap.add_argument("--no-hungarian", action="store_true")
    ap.add_argument("--no-cyclic", action="store_true")

    # v8 room-type regularisation
    ap.add_argument("--class-label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weighting", choices=["none", "inverse", "inv_sqrt"], default="inv_sqrt")
    ap.add_argument("--class-weight-max", type=float, default=4.0)

    # Scheduler / early stop
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--visual-min-delta", type=float, default=0.002)
    ap.add_argument("--early-stop-metric", choices=["loss", "visual", "both"], default="visual")

    ap.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int, default=2)
    ap.add_argument("--min-lr", type=float, default=1e-6)

    ap.add_argument("--save-every", type=int, default=2)

    # Resume / fine-tune
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-model-only", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")

    # v8 model soup / checkpoint averaging
    ap.add_argument("--soup-only", action="store_true")
    ap.add_argument("--soup-checkpoints", default="", help="Semicolon-separated checkpoints to average.")
    ap.add_argument("--soup-output", default="", help="Output checkpoint path for --soup-only.")
    ap.add_argument("--init-soup-checkpoints", default="", help="Average these checkpoints and use as initial model weights.")
    ap.add_argument("--save-final-soup", action="store_true")

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # v8 standalone soup mode
    if args.soup_only:
        soup_paths = _split_checkpoint_list(args.soup_checkpoints)
        if not args.soup_output:
            raise ValueError("--soup-output is required with --soup-only")
        average_checkpoints(soup_paths, Path(args.soup_output))
        return

    train_ds = RoomToFPNPZDataset(
        args.dataset_root,
        "train",
        augment=not args.no_augment,
        bev_noise_std=args.bev_noise_std,
        rgb_noise_std=args.rgb_noise_std,
        channel_dropout_p=args.channel_dropout_p,
        structural_pixel_dropout_p=args.structural_pixel_dropout_p,
    )

    val_ds = RoomToFPNPZDataset(
        args.dataset_root,
        "val",
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample = train_ds[0]
    in_channels = int(sample["x"].shape[0])

    model = RoomToFPPolygonNet(
        in_channels=in_channels,
        max_rooms=args.max_rooms,
        max_corners=args.max_corners,
        embed_dim=args.embed_dim,
        room_classes=args.room_classes,
        dropout=args.dropout,
        predict_furniture=args.predict_furniture,
        base=args.base,
        mask_dim=args.mask_dim,
        decoder_layers=args.decoder_layers,
        corner_decoder_layers=args.corner_decoder_layers,
        num_heads=args.num_heads,
        transformer_encoder_layers=args.transformer_encoder_layers,
        deform_points=args.deform_points,
    ).to(args.device)

    # v8 optional soup initialisation before optimiser creation.
    init_soup_paths = _split_checkpoint_list(args.init_soup_checkpoints)
    if init_soup_paths:
        init_soup_path = out / "init_soup.pt"
        average_checkpoints(init_soup_paths, init_soup_path)
        print(f"[INIT SOUP] loading {init_soup_path}")
        ckpt = torch.load(init_soup_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    else:
        scheduler = None

    class_weights = compute_room_class_weights(
        train_ds,
        num_classes=args.room_classes,
        mode=args.class_weighting,
        max_weight=args.class_weight_max,
    )

    cfg = LossConfig(
        mask_loss=args.mask_loss,
        dice_weight=args.dice_weight,
        focal_gamma=args.focal_gamma,

        coord_weight=args.coord_weight,
        polygon_edge_weight=args.polygon_edge_weight,
        polygon_area_weight=args.polygon_area_weight,
        polygon_angle_weight=args.polygon_angle_weight,
        polygon_raster_weight=args.polygon_raster_weight,
        polygon_raster_size=args.polygon_raster_size,

        aux_decoder_weight=args.aux_decoder_weight,
        dn_weight=args.dn_weight,
        dn_noise=args.dn_noise,

        room_overlap_weight=args.room_overlap_weight,
        aux_room_mask=args.aux_room_mask,
        aux_wall=args.aux_wall,
        aux_opening=args.aux_opening,
        aux_boundary=args.aux_boundary,
        aux_corner_heatmap=args.aux_corner_heatmap,
        aux_furniture=args.aux_furniture,
        boundary_weight=args.boundary_weight,

        match_class_weight=args.match_class_weight,
        match_room_conf_weight=args.match_room_conf_weight,
        use_hungarian=not args.no_hungarian,
        use_cyclic=not args.no_cyclic,

        class_label_smoothing=args.class_label_smoothing,
        room_class_weights=class_weights,
    )

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    ema = None

    start_epoch = 0
    best_val = float("inf")
    best_epoch = 0
    best_visual = -float("inf")
    best_visual_epoch = 0
    history: list[dict[str, Any]] = []

    if args.resume:
        start_epoch, best_val, best_epoch, best_visual, best_visual_epoch, history = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            model_only=args.resume_model_only,
            ema=None,
        )

    # v8 FIX:
    # Create EMA only after loading the resume checkpoint.
    # This ensures EMA starts from the loaded v7 best_ema weights, not random model weights.
    if args.ema:
        ema = ModelEMA(model, decay=args.ema_decay)

        if args.resume and not args.resume_model_only:
            ckpt_for_ema = torch.load(Path(args.resume), map_location="cpu")
            if ckpt_for_ema.get("ema_state_dict") is not None:
                ema.load_state_dict(ckpt_for_ema["ema_state_dict"])
                print("[EMA] loaded EMA state from checkpoint")
            else:
                print("[EMA] no ema_state_dict found; initialized EMA from loaded model weights")
        else:
            print("[EMA] initialized EMA from loaded model weights")

    # v8 fine-tuning safety:
    # If using --resume-model-only, optimiser/scheduler should start fresh.
    # Freeze after loading checkpoint so loaded weights are unchanged.
    if args.freeze_backbone_epochs > 0:
        freeze_backbone_v8(model, freeze=True)

        # Rebuild optimiser after freezing so only trainable params are optimised.
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                min_lr=args.min_lr,
            )
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )
        else:
            scheduler = None

    amp = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None

    bad_epochs = 0
    unfrozen = args.freeze_backbone_epochs <= 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # v8: unfreeze after N fine-tuning epochs.
        if not unfrozen and epoch > args.freeze_backbone_epochs:
            freeze_backbone_v8(model, freeze=False)
            unfrozen = True

            # Rebuild optimiser after unfreezing.
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=max(args.lr * 0.75, args.min_lr),
                weight_decay=args.weight_decay,
            )

            if args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=args.plateau_factor,
                    patience=args.plateau_patience,
                    min_lr=args.min_lr,
                )
            elif args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - epoch + 1),
                    eta_min=args.min_lr,
                )
            else:
                scheduler = None

        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs and start_epoch == 0:
            warmup_lr = args.lr * float(epoch) / float(max(1, args.warmup_epochs))
            set_optimizer_lr(optimizer, warmup_lr)

        lr = float(optimizer.param_groups[0]["lr"])
        print(f"\nEPOCH {epoch}/{args.epochs} | lr={lr:.8f}")

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            args.device,
            train=True,
            amp=amp,
            accum_steps=args.accum_steps,
            ema=ema,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                scaler,
                cfg,
                args.device,
                train=False,
                amp=amp,
                accum_steps=1,
                ema=None,
            )

        val_loss = float(val_metrics.get("loss", float("inf")))
        val_visual = float(val_metrics.get("visual_score", compute_visual_score(val_metrics)))

        rec = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
            "val_loss": val_loss,
            "val_visual": val_visual,
        }

        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if scheduler is not None and not (args.warmup_epochs > 0 and epoch <= args.warmup_epochs and start_epoch == 0):
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        save_checkpoint(
            out / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val,
            best_epoch,
            best_visual,
            best_visual_epoch,
            args,
            history,
            ema=ema,
            model_version="v8-last",
        )

        if ema is not None:
            save_ema_model_checkpoint(
                out / "last_ema.pt",
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                best_val,
                best_epoch,
                best_visual,
                best_visual_epoch,
                args,
                history,
                model_version="v8-last-ema",
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                out / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                best_epoch,
                best_visual,
                best_visual_epoch,
                args,
                history,
                ema=ema,
                model_version="v8-epoch",
            )

            if ema is not None:
                save_ema_model_checkpoint(
                    out / f"epoch_{epoch:03d}_ema.pt",
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    best_epoch,
                    best_visual,
                    best_visual_epoch,
                    args,
                    history,
                    model_version="v8-epoch-ema",
                )

        improved_loss = val_loss < best_val - args.min_delta
        improved_visual = val_visual > best_visual + args.visual_min_delta

        if improved_loss:
            best_val = val_loss
            best_epoch = epoch

            save_checkpoint(
                out / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                best_epoch,
                best_visual,
                best_visual_epoch,
                args,
                history,
                ema=ema,
                model_version="v8-best-loss",
            )

            if ema is not None:
                save_ema_model_checkpoint(
                    out / "best_ema.pt",
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    best_epoch,
                    best_visual,
                    best_visual_epoch,
                    args,
                    history,
                    model_version="v8-best-loss-ema",
                )

            print(f"--Save Best Loss-- epoch={epoch} val_loss={val_loss:.6f}")

        if improved_visual:
            best_visual = val_visual
            best_visual_epoch = epoch

            save_checkpoint(
                out / "best_visual.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                best_epoch,
                best_visual,
                best_visual_epoch,
                args,
                history,
                ema=ema,
                model_version="v8-best-visual",
            )

            if ema is not None:
                save_ema_model_checkpoint(
                    out / "best_visual_ema.pt",
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    best_epoch,
                    best_visual,
                    best_visual_epoch,
                    args,
                    history,
                    model_version="v8-best-visual-ema",
                )

            print(f"--Save Best Visual-- epoch={epoch} visual_score={val_visual:.6f}")

        if args.early_stop_metric == "loss":
            early_improved = improved_loss
        elif args.early_stop_metric == "visual":
            early_improved = improved_visual
        else:
            early_improved = improved_loss or improved_visual

        if early_improved:
            bad_epochs = 0
        else:
            bad_epochs += 1
            print(
                f"--No Early-Stop Improvement-- "
                f"val_loss={val_loss:.6f} best_loss={best_val:.6f}@{best_epoch} | "
                f"visual={val_visual:.6f} best_visual={best_visual:.6f}@{best_visual_epoch} | "
                f"patience={bad_epochs}/{args.patience}"
            )

        if args.fixed_scene_every > 0 and epoch % args.fixed_scene_every == 0:
            monitor_ckpt = out / ("last_ema.pt" if ema is not None else "last.pt")
            run_fixed_scene_monitoring(
                infer_script=args.infer_script,
                scene_dirs=args.fixed_scene_dirs,
                checkpoint=monitor_ckpt,
                output_root=out / "visual_monitor",
                epoch=epoch,
                device=args.device,
                grid_size=int(sample["x"].shape[-1]),
                variants="full",
                room_threshold=args.monitor_room_threshold,
                corner_threshold=args.monitor_corner_threshold,
            )

        if not args.no_early_stop and bad_epochs >= args.patience:
            print("Early stopping.")
            break

    save_final_soup_if_requested(out, args)

    print("\n[DONE]")
    print(f"  best_loss={best_val:.6f} at epoch={best_epoch}")
    print(f"  best_visual={best_visual:.6f} at epoch={best_visual_epoch}")
    print(f"  output_dir={out}")


if __name__ == "__main__":
    main()