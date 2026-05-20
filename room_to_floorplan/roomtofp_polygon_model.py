from __future__ import annotations

"""
RoomToFP v7 model.

Major v7 upgrades:
- Improvement v7: ResNet/FPN-style stronger BEV backbone.
- Improvement v7: Transformer encoder over global BEV memory.
- Improvement v7: pure-PyTorch lightweight multi-scale deformable-style cross attention.
- Improvement v7: iterative coordinate refinement for polygon corners.
- Improvement v7: optional denoising polygon refinement branch used during training.

The class name remains RoomToFPPolygonNet so existing training/inference imports keep working.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinePositionEncoding2D(nn.Module):
    """DETR-style 2D sine/cosine positional encoding for BEV feature maps."""

    def __init__(self, dim: int = 256, temperature: int = 10000):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("Position encoding dim must be divisible by 4")
        self.dim = int(dim)
        self.temperature = int(temperature)

    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        dim_t = torch.arange(self.dim // 4, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / max(1, self.dim // 4))

        px = x[..., None] / dim_t
        py = y[..., None] / dim_t
        pos_x = torch.stack((px.sin(), px.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((py.sin(), py.cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        pos = pos[..., : self.dim]
        return pos.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W


# -----------------------------
# Stronger ResNet/FPN backbone
# -----------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResNetFPNEncoder(nn.Module):
    """
    Improvement v7: stronger ResNet/FPN encoder.

    Input Bx8x512x512 -> multi-scale FPN maps:
      p2: BxCx128x128
      p3: BxCx64x64
      p4: BxCx32x32
      p5: BxCx16x16

    p4 is also passed through a Transformer encoder as global memory.
    """

    def __init__(self, in_channels: int = 8, base: int = 48, embed_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.embed_dim = int(embed_dim)
        b = int(base)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, 7, stride=2, padding=3, bias=False),  # 256x256
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            BasicBlock(b, b, stride=1, dropout=dropout),
        )
        self.layer1 = nn.Sequential(BasicBlock(b, b, 1, dropout), BasicBlock(b, b, 1, dropout))  # 256
        self.layer2 = nn.Sequential(BasicBlock(b, b * 2, 2, dropout), BasicBlock(b * 2, b * 2, 1, dropout))  # 128
        self.layer3 = nn.Sequential(BasicBlock(b * 2, b * 4, 2, dropout), BasicBlock(b * 4, b * 4, 1, dropout))  # 64
        self.layer4 = nn.Sequential(BasicBlock(b * 4, b * 8, 2, dropout), BasicBlock(b * 8, b * 8, 1, dropout))  # 32
        self.layer5 = nn.Sequential(BasicBlock(b * 8, b * 8, 2, dropout), BasicBlock(b * 8, b * 8, 1, dropout))  # 16

        self.lat2 = nn.Conv2d(b * 2, embed_dim, 1)
        self.lat3 = nn.Conv2d(b * 4, embed_dim, 1)
        self.lat4 = nn.Conv2d(b * 8, embed_dim, 1)
        self.lat5 = nn.Conv2d(b * 8, embed_dim, 1)
        self.out2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.out3 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.out4 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.out5 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.stem(x)
        c1 = self.layer1(c1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        return {
            "p2": F.relu(self.out2(p2), inplace=True),
            "p3": F.relu(self.out3(p3), inplace=True),
            "p4": F.relu(self.out4(p4), inplace=True),
            "p5": F.relu(self.out5(p5), inplace=True),
            "stem": c1,
        }


# -----------------------------
# Multi-scale deformable-style attention
# -----------------------------


class MSDeformAttnLite(nn.Module):
    """
    Improvement v7: pure-PyTorch lightweight multi-scale deformable-style attention.

    This approximates Deformable DETR attention without custom CUDA ops.
    Each query predicts offsets and attention weights around a reference point.
    Values are sampled from each FPN level using grid_sample.

    It is slower than CUDA deformable attention but much easier to run in a normal PyTorch environment.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_levels: int = 4, num_points: int = 4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.head_dim = embed_dim // num_heads

        self.value_proj = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(num_levels)])
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.weight_proj = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

    def forward(self, query: torch.Tensor, multi_feats: list[torch.Tensor], reference_xy: torch.Tensor) -> torch.Tensor:
        """
        query: B,Q,C
        multi_feats: list of B,C,H,W maps
        reference_xy: B,Q,2 in [0,1]
        """
        b, q, c = query.shape
        l = min(len(multi_feats), self.num_levels)
        offsets = self.offset_proj(query).view(b, q, self.num_heads, self.num_levels, self.num_points, 2)
        weights = self.weight_proj(query).view(b, q, self.num_heads, self.num_levels, self.num_points)
        weights = F.softmax(weights[:, :, :, :l].reshape(b, q, self.num_heads, -1), dim=-1)
        weights = weights.view(b, q, self.num_heads, l, self.num_points)

        out = query.new_zeros((b, q, self.num_heads, self.head_dim))
        ref = reference_xy.clamp(0.0, 1.0)

        for li in range(l):
            feat = self.value_proj[li](multi_feats[li])
            _, _, h, w = feat.shape
            feat = feat.view(b, self.num_heads, self.head_dim, h, w)
            feat = feat.reshape(b * self.num_heads, self.head_dim, h, w)

            # Convert offset from normalized image fraction. Small initial offsets around ref.
            # Tanh prevents huge samples during early training.
            off = torch.tanh(offsets[:, :, :, li]) * 0.12  # B,Q,Hd,P,2
            sample_xy = ref[:, :, None, None, :] + off
            sample_xy = sample_xy.clamp(0.0, 1.0)
            grid = sample_xy * 2.0 - 1.0  # B,Q,Hd,P,2
            grid = grid.permute(0, 2, 1, 3, 4).reshape(b * self.num_heads, q, self.num_points, 2)

            sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            # sampled: B*heads, head_dim, Q, P
            sampled = sampled.view(b, self.num_heads, self.head_dim, q, self.num_points).permute(0, 3, 1, 4, 2)
            wgt = weights[:, :, :, li].unsqueeze(-1)
            out = out + (sampled * wgt).sum(dim=3)

        out = out.reshape(b, q, c)
        return self.out_proj(out)


class V7DecoderLayer(nn.Module):
    """Self-attention + MS deformable-style cross attention + FFN."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.10, num_levels: int = 4, num_points: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttnLite(embed_dim, num_heads, num_levels, num_points)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, multi_feats: list[torch.Tensor], reference_xy: torch.Tensor) -> torch.Tensor:
        sa, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(sa))
        ca = self.cross_attn(x, multi_feats, reference_xy)
        x = self.norm2(x + self.drop(ca))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


# -----------------------------
# Dense mask decoder
# -----------------------------


class V7FPNMaskDecoder(nn.Module):
    """FPN decoder using p2/p3/p4/p5 for dense wall/opening/boundary/corner/room masks."""

    def __init__(self, embed_dim: int = 256, out_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.proj5 = nn.Conv2d(embed_dim, out_dim, 1)
        self.proj4 = nn.Conv2d(embed_dim, out_dim, 1)
        self.proj3 = nn.Conv2d(embed_dim, out_dim, 1)
        self.proj2 = nn.Conv2d(embed_dim, out_dim, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, feats: dict[str, torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        p5 = self.proj5(feats["p5"])
        p4 = self.proj4(feats["p4"]) + F.interpolate(p5, size=feats["p4"].shape[-2:], mode="nearest")
        p3 = self.proj3(feats["p3"]) + F.interpolate(p4, size=feats["p3"].shape[-2:], mode="nearest")
        p2 = self.proj2(feats["p2"]) + F.interpolate(p3, size=feats["p2"].shape[-2:], mode="nearest")
        y = self.refine(p2)
        return F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)


# -----------------------------
# Main model
# -----------------------------


class RoomToFPPolygonNet(nn.Module):
    """
    RoomToFP v7 network.

    Keeps the existing output dictionary keys used by training/inference:
      room_valid_logits, room_class_logits, corner_valid_logits, corner_xy,
      wall_logits, opening_logits, room_mask_logits, boundary_logits,
      corner_heatmap_logits, optional furniture_logits, aux_outputs, dn_outputs.
    """

    def __init__(
        self,
        in_channels: int = 8,
        max_rooms: int = 20,
        max_corners: int = 16,
        embed_dim: int = 256,
        room_classes: int = 12,
        dropout: float = 0.10,
        predict_furniture: bool = False,
        base: int = 48,
        mask_dim: int = 128,
        decoder_layers: int = 4,
        corner_decoder_layers: int = 3,
        num_heads: int = 8,
        transformer_encoder_layers: int = 2,
        deform_points: int = 4,
        **_: object,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.max_rooms = int(max_rooms)
        self.max_corners = int(max_corners)
        self.embed_dim = int(embed_dim)
        self.room_classes = int(room_classes)
        self.predict_furniture = bool(predict_furniture)

        self.encoder = ResNetFPNEncoder(in_channels, base=base, embed_dim=embed_dim, dropout=dropout)
        self.posenc = SinePositionEncoding2D(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(transformer_encoder_layers))

        self.level_embed = nn.Parameter(torch.zeros(4, embed_dim))
        nn.init.normal_(self.level_embed, std=0.02)

        self.room_queries = nn.Embedding(max_rooms, embed_dim)
        self.corner_queries = nn.Embedding(max_corners, embed_dim)
        self.room_ref_head = MLP(embed_dim, embed_dim, 2, 3)
        self.room_decoder = nn.ModuleList([
            V7DecoderLayer(embed_dim, num_heads, dropout, num_levels=4, num_points=deform_points)
            for _ in range(int(decoder_layers))
        ])

        self.corner_init_ref = MLP(embed_dim, embed_dim, 2, 3)
        self.corner_decoder = nn.ModuleList([
            V7DecoderLayer(embed_dim, num_heads, dropout, num_levels=4, num_points=deform_points)
            for _ in range(int(corner_decoder_layers))
        ])
        self.corner_delta_heads = nn.ModuleList([MLP(embed_dim, embed_dim, 2, 3) for _ in range(int(corner_decoder_layers))])

        self.room_valid_head = MLP(embed_dim, embed_dim, 1, 3)
        self.class_head = MLP(embed_dim, embed_dim, room_classes, 3)
        self.corner_valid_head = MLP(embed_dim, embed_dim, 1, 3)

        self.mask_decoder = V7FPNMaskDecoder(embed_dim, out_dim=mask_dim, dropout=dropout)
        self.wall_head = nn.Conv2d(mask_dim, 1, 1)
        self.opening_head = nn.Conv2d(mask_dim, 1, 1)
        self.boundary_head = nn.Conv2d(mask_dim, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(mask_dim, 1, 1)
        self.room_mask_embed = MLP(embed_dim, embed_dim, mask_dim, 3)
        self.furniture_head = nn.Conv2d(mask_dim, 1, 1) if self.predict_furniture else None

        # Improvement v7: denoising query branch. It refines noisy GT polygons during training.
        self.dn_coord_embed = nn.Linear(2, embed_dim)
        self.dn_valid_embed = nn.Embedding(2, embed_dim)
        self.dn_room_embed = nn.Embedding(max_rooms, embed_dim)
        self.dn_corner_valid_head = MLP(embed_dim, embed_dim, 1, 3)
        self.dn_room_valid_head = MLP(embed_dim, embed_dim, 1, 3)
        self.dn_class_head = MLP(embed_dim, embed_dim, room_classes, 3)
        self.dn_delta_heads = nn.ModuleList([MLP(embed_dim, embed_dim, 2, 3) for _ in range(int(corner_decoder_layers))])

    def _encode_memory(self, feats: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # Transformer encoder on p4 global memory.
        p4 = feats["p4"]
        b, c, h, w = p4.shape
        pos = self.posenc(h, w, p4.device, p4.dtype)
        mem = (p4 + pos).flatten(2).transpose(1, 2)
        mem = self.memory_encoder(mem)
        p4_enc = mem.transpose(1, 2).reshape(b, c, h, w)

        multi_feats = [feats["p2"], feats["p3"], p4_enc, feats["p5"]]
        out: list[torch.Tensor] = []
        for i, f in enumerate(multi_feats):
            out.append(f + self.level_embed[i].view(1, -1, 1, 1))
        feats["p4"] = p4_enc
        return out

    def _decode_rooms(self, multi_feats: list[torch.Tensor], b: int) -> tuple[torch.Tensor, torch.Tensor]:
        room_tokens = self.room_queries.weight.unsqueeze(0).repeat(b, 1, 1)
        room_ref = torch.sigmoid(self.room_ref_head(room_tokens))
        for layer in self.room_decoder:
            room_tokens = layer(room_tokens, multi_feats, room_ref)
            # light iterative room reference update
            room_ref = torch.sigmoid(inverse_sigmoid(room_ref) + 0.1 * torch.tanh(self.room_ref_head(room_tokens)))
        return room_tokens, room_ref

    def _decode_corners(self, room_tokens: torch.Tensor, room_ref: torch.Tensor, multi_feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        b, m, c = room_tokens.shape
        n = self.max_corners
        corner_base = self.corner_queries.weight.view(1, 1, n, c)
        tokens = room_tokens.unsqueeze(2) + corner_base
        tokens = tokens.reshape(b, m * n, c)
        init_ref = torch.sigmoid(self.corner_init_ref(tokens))
        # Bias initial corner refs around room refs.
        room_ref_flat = room_ref.unsqueeze(2).expand(b, m, n, 2).reshape(b, m * n, 2)
        ref = (0.65 * room_ref_flat + 0.35 * init_ref).clamp(0.0, 1.0)

        aux: list[dict] = []
        for li, layer in enumerate(self.corner_decoder):
            tokens = layer(tokens, multi_feats, ref)
            delta = self.corner_delta_heads[li](tokens)
            ref = torch.sigmoid(inverse_sigmoid(ref) + delta)
            xy = ref.view(b, m, n, 2)
            ctok = tokens.view(b, m, n, c)
            aux.append({
                "corner_xy": xy,
                "corner_valid_logits": self.corner_valid_head(ctok).squeeze(-1),
                "room_valid_logits": self.room_valid_head(room_tokens).squeeze(-1),
                "room_class_logits": self.class_head(room_tokens),
            })
        return tokens.view(b, m, n, c), ref.view(b, m, n, 2), aux

    def _decode_denoising(self, dn_polygon_xy: torch.Tensor, dn_corner_valid: torch.Tensor, dn_room_valid: torch.Tensor, multi_feats: list[torch.Tensor]) -> dict:
        b, m, n, _ = dn_polygon_xy.shape
        device = dn_polygon_xy.device
        room_ids = torch.arange(m, device=device).view(1, m, 1).expand(b, m, n)
        valid_ids = dn_corner_valid.long().clamp(0, 1)
        tokens = self.dn_coord_embed(dn_polygon_xy) + self.dn_valid_embed(valid_ids) + self.dn_room_embed(room_ids)
        tokens = tokens.reshape(b, m * n, self.embed_dim)
        ref = dn_polygon_xy.reshape(b, m * n, 2).clamp(0.0, 1.0)
        aux: list[dict] = []
        for li, layer in enumerate(self.corner_decoder):
            tokens = layer(tokens, multi_feats, ref)
            delta = self.dn_delta_heads[li](tokens)
            ref = torch.sigmoid(inverse_sigmoid(ref) + delta)
            ctok = tokens.view(b, m, n, self.embed_dim)
            xy = ref.view(b, m, n, 2)
            room_tok = (ctok * dn_corner_valid.unsqueeze(-1).float()).sum(dim=2) / dn_corner_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
            aux.append({
                "corner_xy": xy,
                "corner_valid_logits": self.dn_corner_valid_head(ctok).squeeze(-1),
                "room_valid_logits": self.dn_room_valid_head(room_tok).squeeze(-1),
                "room_class_logits": self.dn_class_head(room_tok),
            })
        return aux[-1] | {"aux_outputs": aux[:-1]}

    def forward(
        self,
        x: torch.Tensor,
        dn_polygon_xy: Optional[torch.Tensor] = None,
        dn_corner_valid: Optional[torch.Tensor] = None,
        dn_room_valid: Optional[torch.Tensor] = None,
    ) -> dict:
        b, _, h, w = x.shape
        feats = self.encoder(x)
        multi_feats = self._encode_memory(feats)
        room_tokens, room_ref = self._decode_rooms(multi_feats, b)
        corner_tokens, corner_xy, aux_outputs = self._decode_corners(room_tokens, room_ref, multi_feats)

        mask_feat = self.mask_decoder(feats, target_size=(h, w))
        mask_embed = self.room_mask_embed(room_tokens)  # B,M,mask_dim
        room_mask_logits = torch.einsum("bmc,bchw->bmhw", mask_embed, mask_feat)

        out = {
            "room_valid_logits": self.room_valid_head(room_tokens).squeeze(-1),
            "room_class_logits": self.class_head(room_tokens),
            "corner_valid_logits": self.corner_valid_head(corner_tokens).squeeze(-1),
            "corner_xy": corner_xy,
            "room_reference_xy": room_ref,
            "wall_logits": self.wall_head(mask_feat),
            "opening_logits": self.opening_head(mask_feat),
            "boundary_logits": self.boundary_head(mask_feat),
            "corner_heatmap_logits": self.corner_heatmap_head(mask_feat),
            "room_mask_logits": room_mask_logits,
            "aux_outputs": aux_outputs[:-1],
        }
        if self.furniture_head is not None:
            out["furniture_logits"] = self.furniture_head(mask_feat)

        if dn_polygon_xy is not None and dn_corner_valid is not None and dn_room_valid is not None:
            out["dn_outputs"] = self._decode_denoising(dn_polygon_xy, dn_corner_valid, dn_room_valid, multi_feats)

        return out
