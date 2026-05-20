from __future__ import annotations

"""
RoomToFP v7 training script.

Major v7 upgrades:
- Improvement v7: works with the v7 ResNet/FPN + Transformer encoder + deformable-style model.
- Improvement v7: iterative corner-refinement supervision through aux_outputs.
- Improvement v7: denoising training branch using noisy GT polygons.
- Improvement v7: keeps v6 polygon rasterization loss and polygon-first objective.

Expected dataset format: samples/*.npz created by prepare_roomtofp_dataset.py.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import copy
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

def ensure_chw_mask_np(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """
    Improvement v7 fix: normalise mask arrays to C x H x W.

    Valid incoming shapes:
        H x W
        1 x H x W
        H x W x 1
        1 x 1 x H x W

    Output:
        C x H x W

    This prevents DataLoader from producing:
        B x 1 x 1 x H x W

    when the model expects:
        B x 1 x H x W
    """
    mask = np.asarray(mask, dtype=np.float32)

    # H x W -> 1 x H x W
    if mask.ndim == 2:
        return mask[None, :, :].astype(np.float32)

    # H x W x 1 -> 1 x H x W
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask.transpose(2, 0, 1).astype(np.float32)

    # 1 x H x W, or max_rooms x H x W -> already valid
    if mask.ndim == 3:
        return mask.astype(np.float32)

    # 1 x 1 x H x W -> 1 x H x W
    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    if mask.ndim == 2:
        mask = mask[None, :, :]

    if mask.ndim != 3:
        raise ValueError(f"{name} has unexpected shape after cleanup: {mask.shape}")

    return mask.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------


class RoomToFPNPZDataset(Dataset):
    def __init__(self, dataset_root: str | Path, split: str = "train", augment: bool = True):
        self.root = Path(dataset_root)
        self.split = str(split)
        self.augment = bool(augment and split == "train")
        split_path = self.root / f"{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        self.names = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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

        # observation is H x W x C, so x flip is width axis = axis 1
        s["observation"] = s["observation"][:, ::-1, :].copy()

        # masks can be H x W, 1 x H x W, or M x H x W
        for key in ["wall_mask", "opening_mask", "furniture_mask", "boundary_mask", "corner_heatmap", "room_mask"]:
            if key in s:
                s[key] = np.flip(s[key], axis=-1).copy()

        if "polygon_xy" in s:
            s["polygon_xy"][..., 0] = 1.0 - s["polygon_xy"][..., 0]

        return s

    @staticmethod
    def _flip_y(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        s = {k: np.array(v, copy=True) for k, v in sample.items()}

        # observation is H x W x C, so y flip is height axis = axis 0
        s["observation"] = s["observation"][::-1, :, :].copy()

        # masks can be H x W, 1 x H x W, or M x H x W
        for key in ["wall_mask", "opening_mask", "furniture_mask", "boundary_mask", "corner_heatmap", "room_mask"]:
            if key in s:
                s[key] = np.flip(s[key], axis=-2).copy()

        if "polygon_xy" in s:
            s["polygon_xy"][..., 1] = 1.0 - s["polygon_xy"][..., 1]

        return s

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._load(self.names[idx])
        if self.augment:
            if random.random() < 0.5:
                s = self._flip_x(s)
            if random.random() < 0.5:
                s = self._flip_y(s)

        obs = s["observation"].astype(np.float32)  # H,W,C

        # Improvement v7 fix: normalise all dense masks before DataLoader batching.
        # Some prepared samples store masks as H x W, others as 1 x H x W.
        # Returning them consistently prevents B x 1 x 1 x H x W target tensors.
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


# -----------------------------
# Loss helpers
# -----------------------------


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

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    focal_weight = (1.0 - pt).pow(float(gamma))

    return (focal_weight * bce).mean()


def mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: LossConfig,
) -> torch.Tensor:
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



def align_target_to_logits(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Improvement v7 fix: make binary mask targets match model logit shape.

    Model logits are usually:
        B x 1 x H x W

    Some targets may accidentally arrive as:
        B x 1 x 1 x H x W

    This safely removes extra singleton dimensions and expands where valid.
    """
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


def cyclic_align_target(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, use_cyclic: bool = True) -> torch.Tensor:
    """Align target corner order to prediction using cyclic and reversed shifts."""
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


def pair_cost(pred_xy: torch.Tensor, tgt_xy: torch.Tensor, tgt_valid: torch.Tensor, use_cyclic: bool) -> float:
    aligned = cyclic_align_target(pred_xy.detach(), tgt_xy.detach(), tgt_valid.detach(), use_cyclic)
    v = tgt_valid.bool()
    if v.sum() == 0:
        return 1e3
    return float(torch.abs(pred_xy[v] - aligned[v]).mean().detach().cpu())


def hungarian_match(pred: dict, batch: dict, cfg: LossConfig) -> list[list[tuple[int, int]]]:
    b, m, n, _ = pred["corner_xy"].shape
    matches: list[list[tuple[int, int]]] = []
    room_prob = torch.sigmoid(pred["room_valid_logits"]).detach()
    class_logits = pred["room_class_logits"].detach()
    for bi in range(b):
        gt_ids = torch.where(batch["room_valid"][bi] > 0.5)[0].tolist()
        if not gt_ids:
            matches.append([])
            continue
        cost = np.zeros((m, len(gt_ids)), dtype=np.float32)
        for pi in range(m):
            for gj, gi in enumerate(gt_ids):
                c = pair_cost(pred["corner_xy"][bi, pi], batch["polygon_xy"][bi, gi], batch["corner_valid"][bi, gi], cfg.use_cyclic)
                cls = int(batch["room_class"][bi, gi].item())
                ce = float(F.cross_entropy(class_logits[bi, pi].unsqueeze(0), torch.tensor([cls], device=class_logits.device), reduction="mean").detach().cpu())
                conf = float((1.0 - room_prob[bi, pi]).detach().cpu())
                cost[pi, gj] = c + cfg.match_class_weight * ce + cfg.match_room_conf_weight * conf
        if cfg.use_hungarian and linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            matches.append([(int(r), int(gt_ids[c])) for r, c in zip(rows, cols)])
        else:
            # Greedy fallback.
            pairs = []
            used_p, used_g = set(), set()
            flat = [(cost[i, j], i, j) for i in range(m) for j in range(len(gt_ids))]
            for _, i, j in sorted(flat, key=lambda x: x[0]):
                if i in used_p or j in used_g:
                    continue
                used_p.add(i); used_g.add(j)
                pairs.append((i, gt_ids[j]))
            matches.append(pairs)
    return matches


def rasterize_poly_cv(poly_xy: torch.Tensor, size: int) -> torch.Tensor:
    pts = poly_xy.detach().cpu().numpy()
    pts = np.clip(np.round(pts * (size - 1)), 0, size - 1).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.float32)
    if len(np.unique(pts, axis=0)) >= 3:
        cv2.fillPoly(mask, [pts], 1.0)
    return torch.from_numpy(mask).to(poly_xy.device, dtype=poly_xy.dtype)


def polygon_raster_loss(pred_poly: torch.Tensor, tgt_mask: torch.Tensor, size: int) -> torch.Tensor:
    """Non-differentiable rasterizer around predicted coords, but useful as a shape-overlap penalty proxy.

    Note: OpenCV rasterization is not differentiable. It still gives monitoring and weak objective if used
    in detached form would not backprop through coords. For true differentiable rasterization, replace this
    with SoftRas/PyTorch3D-like rasterizer. We keep it here because it is dependency-free and stable.
    """
    # Differentiable approximation: soft occupancy based on distance to polygon vertices and bbox.
    # This is not exact polygon fill, but it gives gradients to coordinates.
    s = int(size)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, s, device=pred_poly.device, dtype=pred_poly.dtype),
        torch.linspace(0, 1, s, device=pred_poly.device, dtype=pred_poly.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # P,2
    v0 = pred_poly
    v1 = torch.roll(pred_poly, shifts=-1, dims=0)
    edges = v1 - v0
    gp = grid[:, None, :] - v0[None, :, :]
    t = (gp * edges[None]).sum(-1) / (edges.pow(2).sum(-1)[None] + 1e-6)
    t = t.clamp(0, 1)
    proj = v0[None] + t[..., None] * edges[None]
    dist = torch.linalg.norm(grid[:, None, :] - proj, dim=-1).min(dim=1).values
    boundary_prob = torch.exp(-dist * 80.0).view(s, s)
    # Fill proxy using bbox of polygon, good enough to push area coverage.
    mn = pred_poly.min(dim=0).values
    mx = pred_poly.max(dim=0).values
    inside_x = torch.sigmoid((xx - mn[0]) * 80) * torch.sigmoid((mx[0] - xx) * 80)
    inside_y = torch.sigmoid((yy - mn[1]) * 80) * torch.sigmoid((mx[1] - yy) * 80)
    pred_mask = (inside_x * inside_y).clamp(0, 1)
    pred_mask = torch.maximum(pred_mask, boundary_prob * 0.25)
    tgt = F.interpolate(tgt_mask[None, None].float(), size=(s, s), mode="nearest")[0, 0]
    inter = (pred_mask * tgt).sum()
    dice = (2 * inter + 1e-6) / (pred_mask.sum() + tgt.sum() + 1e-6)
    return 1.0 - dice


def make_denoising_inputs(batch: dict, noise: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xy = batch["polygon_xy"].clone()
    cv = batch["corner_valid"].clone()
    rv = batch["room_valid"].clone()
    noise_tensor = torch.randn_like(xy) * float(noise)
    xy_noisy = (xy + noise_tensor * cv.unsqueeze(-1)).clamp(0.0, 1.0)
    return xy_noisy, cv, rv


def compute_losses(pred: dict, batch: dict, cfg: LossConfig, include_aux: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    device = pred["corner_xy"].device
    b, m, n, _ = pred["corner_xy"].shape
    matches = hungarian_match(pred, batch, cfg)

    room_targets = torch.zeros_like(pred["room_valid_logits"])
    class_losses, coord_losses, corner_bce_losses = [], [], []
    edge_losses, area_losses, angle_losses, raster_losses = [], [], [], []
    room_mask_losses = []
    invalid_count = 0
    total_poly = 0

    for bi, pairs in enumerate(matches):
        for pi, gi in pairs:
            room_targets[bi, pi] = 1.0
            tgt_valid = batch["corner_valid"][bi, gi]
            pred_xy = pred["corner_xy"][bi, pi]
            tgt_xy = cyclic_align_target(pred_xy, batch["polygon_xy"][bi, gi], tgt_valid, cfg.use_cyclic)
            valid = tgt_valid.bool()
            if valid.sum() >= 3:
                class_losses.append(F.cross_entropy(pred["room_class_logits"][bi, pi].unsqueeze(0), batch["room_class"][bi, gi].unsqueeze(0)))
                coord_losses.append(torch.abs(pred_xy[valid] - tgt_xy[valid]).mean())
                corner_bce_losses.append(F.binary_cross_entropy_with_logits(pred["corner_valid_logits"][bi, pi], tgt_valid))
                ppoly = pred_xy[valid]
                tpoly = tgt_xy[valid]
                edge_losses.append(torch.abs(edge_lengths(ppoly) - edge_lengths(tpoly)).mean())
                area_losses.append(torch.abs(polygon_area_norm(ppoly) - polygon_area_norm(tpoly)))
                angle_losses.append(manhattan_angle_loss(ppoly))
                raster_losses.append(polygon_raster_loss(ppoly, batch["room_mask"][bi, gi], cfg.polygon_raster_size))
                if "room_mask_logits" in pred:
                    room_mask_losses.append(mask_loss(pred["room_mask_logits"][bi, pi].unsqueeze(0).unsqueeze(0), batch["room_mask"][bi, gi].unsqueeze(0).unsqueeze(0), cfg))
                total_poly += 1
                if float(polygon_area_norm(ppoly).detach().cpu()) < 1e-4:
                    invalid_count += 1

    # Room BCE with positive weight to avoid predicting no rooms.
    pos = room_targets.sum()
    neg = room_targets.numel() - pos
    pos_weight_val = float((neg / pos.clamp_min(1.0)).detach().cpu()) if pos > 0 else 1.0
    pos_weight_val = min(pos_weight_val, cfg.room_pos_weight_max)
    room_bce = F.binary_cross_entropy_with_logits(
        pred["room_valid_logits"], room_targets, pos_weight=torch.tensor(pos_weight_val, device=device)
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
        room_bce + corner_bce + class_l
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
        for aux in pred["aux_outputs"]:
            aux_loss, _ = compute_losses(aux | {k: pred[k] for k in ["wall_logits", "opening_logits", "boundary_logits", "corner_heatmap_logits", "room_mask_logits"] if k in pred}, batch, cfg, include_aux=False)
            aux_l_total = aux_l_total + aux_loss
        aux_l_total = aux_l_total / max(1, len(pred["aux_outputs"]))
        loss = loss + cfg.aux_decoder_weight * aux_l_total

    dn_l = torch.zeros((), device=device)
    if cfg.dn_weight > 0 and "dn_outputs" in pred:
        dn = pred["dn_outputs"]
        # Direct no-matching loss on GT slots.
        rv = batch["room_valid"]
        cv = batch["corner_valid"]
        dn_room_bce = F.binary_cross_entropy_with_logits(dn["room_valid_logits"], rv)
        dn_corner_bce = F.binary_cross_entropy_with_logits(dn["corner_valid_logits"], cv)
        valid = (rv.unsqueeze(-1) * cv).bool()
        dn_coord = torch.abs(dn["corner_xy"] - batch["polygon_xy"])[valid].mean() if valid.any() else torch.zeros((), device=device)
        cls_mask = rv.bool()
        dn_cls = F.cross_entropy(dn["room_class_logits"][cls_mask], batch["room_class"][cls_mask]) if cls_mask.any() else torch.zeros((), device=device)
        dn_l = dn_room_bce + dn_corner_bce + cfg.coord_weight * dn_coord + dn_cls
        loss = loss + cfg.dn_weight * dn_l

    parts = {
        "loss": float(loss.detach().cpu()),
        "coord_l1": float(coord_l.detach().cpu()),
        "wall_dice": dice_score_from_logits(pred["wall_logits"], batch["wall_mask"]),
        "opening_dice": dice_score_from_logits(pred["opening_logits"], batch["opening_mask"]),
        "boundary_dice": dice_score_from_logits(pred["boundary_logits"], batch["boundary_mask"]),
        "corner_heatmap_dice": dice_score_from_logits(pred["corner_heatmap_logits"], batch["corner_heatmap"]),
        "matched_room_mask_dice": float(torch.tensor([1.0]).mean().item()),
        "invalid_polygon_rate": float(invalid_count / max(1, total_poly)),
        "raster_loss": float(raster_l.detach().cpu()),
        "aux_decoder_loss": float(aux_l_total.detach().cpu()),
        "dn_loss": float(dn_l.detach().cpu()),
    }
    # Matched room mask Dice metric.
    if "room_mask_logits" in pred:
        vals = []
        for bi, pairs in enumerate(matches):
            for pi, gi in pairs:
                vals.append(dice_score_from_logits(pred["room_mask_logits"][bi, pi].unsqueeze(0).unsqueeze(0), batch["room_mask"][bi, gi].unsqueeze(0).unsqueeze(0)))
        if vals:
            parts["matched_room_mask_dice"] = float(np.mean(vals))
    return loss, parts


# -----------------------------
# Training loop
# -----------------------------


class ModelEMA:
    """
    Improvement v7-final: Exponential Moving Average checkpointing.

    EMA keeps a smoothed copy of model weights. Transformer/polygon models can be noisy,
    so EMA checkpoints often produce cleaner inference than the raw last-step weights.
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
            if k not in self.shadow or not torch.is_floating_point(v):
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for k, v in self.shadow.items():
            if k in model_state:
                model_state[k].copy_(v.to(model_state[k].device, dtype=model_state[k].dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for k, v in self.backup.items():
            if k in model_state:
                model_state[k].copy_(v)
        self.backup = {}


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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
    Improvement v7-final: fixed-scene visual monitoring.

    This runs inference on user-specified scenes every N epochs so visual topology can be checked,
    not only scalar losses.
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
            "--opening-snap-distance-px", "45",
            "--min-opening-area-px", "4",
            "--min-opening-length-px", "14",
        ]
        print("[MONITOR]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[MONITOR] failed for {scene}: {e}")


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def average_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*[x.keys() for x in items]))
    return {k: float(np.mean([x[k] for x in items if k in x])) for k in keys}


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
                pred = model(batch["x"], dn_polygon_xy=dn_xy, dn_corner_valid=dn_cv, dn_room_valid=dn_rv)
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
        pbar.set_postfix({
            "loss": f"{parts['loss']:.2f}",
            "coord": f"{parts['coord_l1']:.3f}",
            "wdice": f"{parts['wall_dice']:.3f}",
            "odice": f"{parts['opening_dice']:.3f}",
            "rmdice": f"{parts['matched_room_mask_dice']:.3f}",
            "bdice": f"{parts['boundary_dice']:.3f}",
        })
    return average_dict(all_parts)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val: float, best_epoch: int, args, history, ema: ModelEMA | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "history": history,
        "model_version": "v7-final",
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    torch.save(payload, path)


def save_ema_model_checkpoint(path: Path, model, ema: ModelEMA, optimizer, scheduler, epoch: int, best_val: float, best_epoch: int, args, history):
    """Save a normal checkpoint whose model_state_dict is the EMA-smoothed weights, so inference works unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_state = copy.deepcopy(model.state_dict())
    ema.store(model)
    ema.copy_to(model)
    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "history": history,
        "model_version": "v7-final-ema",
    }
    model.load_state_dict(raw_state)
    ema.restore(model)
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, model_only: bool = False, ema: ModelEMA | None = None):
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
        return 0, float("inf"), 0, []
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_val", float("inf"))), int(ckpt.get("best_epoch", 0)), ckpt.get("history", [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RoomToFP v7 polygon model")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
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
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--predict-furniture", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--amp", action="store_true")
    # Improvement v7-final: stability controls for the final run.
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps. Effective batch size = batch_size * accum_steps.")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="Linearly warm up LR for this many epochs.")
    ap.add_argument("--ema", action="store_true", help="Maintain EMA weights and save best_ema.pt/last_ema.pt.")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay value.")
    ap.add_argument("--fixed-scene-dirs", default="", help="Semicolon-separated scene dirs for periodic visual monitoring.")
    ap.add_argument("--fixed-scene-every", type=int, default=0, help="Run inference on fixed scenes every N epochs. 0 disables.")
    ap.add_argument("--infer-script", default="infer_roomtofp_from_scene.py", help="Path to inference script used for fixed-scene monitoring.")
    ap.add_argument("--monitor-room-threshold", type=float, default=0.35)
    ap.add_argument("--monitor-corner-threshold", type=float, default=0.45)

    ap.add_argument("--mask-loss", choices=["bce", "bce_dice", "focal_dice"], default="focal_dice")
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--coord-weight", type=float, default=10.0)
    ap.add_argument("--polygon-edge-weight", type=float, default=1.5)
    ap.add_argument("--polygon-area-weight", type=float, default=0.50)
    ap.add_argument("--polygon-angle-weight", type=float, default=0.35)
    ap.add_argument("--polygon-raster-weight", type=float, default=1.25)
    ap.add_argument("--polygon-raster-size", type=int, default=64)
    ap.add_argument("--aux-decoder-weight", type=float, default=0.40)
    ap.add_argument("--dn-weight", type=float, default=0.50)
    ap.add_argument("--dn-noise", type=float, default=0.07)
    ap.add_argument("--room-overlap-weight", type=float, default=0.10)
    ap.add_argument("--aux-room-mask", type=float, default=0.5)
    ap.add_argument("--aux-wall", type=float, default=0.5)
    ap.add_argument("--aux-opening", type=float, default=0.8)
    ap.add_argument("--aux-boundary", type=float, default=0.35)
    ap.add_argument("--aux-corner-heatmap", type=float, default=0.25)
    ap.add_argument("--aux-furniture", type=float, default=0.0)
    ap.add_argument("--boundary-weight", type=float, default=0.25)
    ap.add_argument("--match-class-weight", type=float, default=0.20)
    ap.add_argument("--match-room-conf-weight", type=float, default=0.20)
    ap.add_argument("--no-hungarian", action="store_true")
    ap.add_argument("--no-cyclic", action="store_true")

    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int, default=4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-model-only", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_ds = RoomToFPNPZDataset(args.dataset_root, "train", augment=not args.no_augment)
    val_ds = RoomToFPNPZDataset(args.dataset_root, "val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Infer input channels from first sample.
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = None

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
    )

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    start_epoch = 0
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        start_epoch, best_val, best_epoch, history = load_checkpoint(Path(args.resume), model, optimizer, scheduler, model_only=args.resume_model_only, ema=ema)

    amp = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None
    bad_epochs = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs and start_epoch == 0:
            warmup_lr = args.lr * float(epoch) / float(max(1, args.warmup_epochs))
            set_optimizer_lr(optimizer, warmup_lr)
        lr = float(optimizer.param_groups[0]["lr"])
        print(f"\nEPOCH {epoch}/{args.epochs} | lr={lr:.8f}")
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, cfg, args.device, train=True, amp=amp, accum_steps=args.accum_steps, ema=ema)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, cfg, args.device, train=False, amp=amp, accum_steps=1, ema=None)

        val_loss = float(val_metrics.get("loss", float("inf")))
        rec = {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics}
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if scheduler is not None and not (args.warmup_epochs > 0 and epoch <= args.warmup_epochs and start_epoch == 0):
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        save_checkpoint(out / "last.pt", model, optimizer, scheduler, epoch, best_val, best_epoch, args, history, ema=ema)
        if ema is not None:
            save_ema_model_checkpoint(out / "last_ema.pt", model, ema, optimizer, scheduler, epoch, best_val, best_epoch, args, history)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, best_val, best_epoch, args, history, ema=ema)
            if ema is not None:
                save_ema_model_checkpoint(out / f"epoch_{epoch:03d}_ema.pt", model, ema, optimizer, scheduler, epoch, best_val, best_epoch, args, history)

        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch, best_val, best_epoch, args, history, ema=ema)
            if ema is not None:
                save_ema_model_checkpoint(out / "best_ema.pt", model, ema, optimizer, scheduler, epoch, best_val, best_epoch, args, history)
            print(f"--Save Best-- epoch={epoch} val_loss={val_loss:.6f}")
        else:
            bad_epochs += 1
            print(f"--No Improvement-- val_loss={val_loss:.6f} | best={best_val:.6f} at epoch={best_epoch} | patience={bad_epochs}/{args.patience}")

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


if __name__ == "__main__":
    main()
