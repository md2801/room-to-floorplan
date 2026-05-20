from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Takes the input BEV tensor and converts it into smaller feature maps for the Transformer.

    8-channel BEV input -> CNN Encoder -> compressed spatial feature map -> Transformer decoder predicts polygons.
    """
    def __init__(self, in_channels: int = 8, base: int = 48, embed_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.base = int(base)
        self.embed_dim = int(embed_dim)

        # Improvement v3: widened the encoder into a multi-scale feature extractor so the model keeps high-resolution wall/opening evidence while still producing compact Transformer tokens.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 7, stride=2, padding=3),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.down1 = self._block(base, base * 2, dropout)
        self.down2 = self._block(base * 2, base * 4, dropout)
        self.down3 = self._block(base * 4, base * 8, dropout)
        self.proj = nn.Conv2d(base * 8, embed_dim, 1)

    @staticmethod
    def _block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_skips: bool = False):
        """
        For 512x512 input:
        x    : B x 8   x 512 x 512
        stem : B x 48  x 256 x 256
        down1: B x 96  x 128 x 128
        down2: B x 192 x 64  x 64
        down3: B x 384 x 32  x 32
        feat : B x 256 x 32  x 32
        """
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        feat = self.proj(s4)
        if return_skips:
            return feat, {"stem": s1, "down1": s2, "down2": s3, "down3": s4}
        return feat


class FPNMaskDecoder(nn.Module):
    """
    Lightweight FPN/U-Net style decoder for dense wall/opening/furniture masks.
    """
    def __init__(self, embed_dim: int, base: int = 48, out_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.lateral3 = nn.Conv2d(embed_dim, out_dim, 1)
        self.lateral2 = nn.Conv2d(base * 4, out_dim, 1)
        self.lateral1 = nn.Conv2d(base * 2, out_dim, 1)
        self.lateral0 = nn.Conv2d(base, out_dim, 1)
        self.refine3 = self._conv_block(out_dim, out_dim, dropout)
        self.refine2 = self._conv_block(out_dim, out_dim, dropout)
        self.refine1 = self._conv_block(out_dim, out_dim, dropout)
        self.refine0 = self._conv_block(out_dim, out_dim, dropout)

    @staticmethod
    def _conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor, skips: dict[str, torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        p3 = self.refine3(self.lateral3(feat))
        p2 = self.refine2(self.lateral2(skips["down2"]) + F.interpolate(p3, size=skips["down2"].shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.refine1(self.lateral1(skips["down1"]) + F.interpolate(p2, size=skips["down1"].shape[-2:], mode="bilinear", align_corners=False))
        p0 = self.refine0(self.lateral0(skips["stem"]) + F.interpolate(p1, size=skips["stem"].shape[-2:], mode="bilinear", align_corners=False))
        return F.interpolate(p0, size=target_size, mode="bilinear", align_corners=False)


class RoomToFPPolygonNet(nn.Module):
    """
    Main RoomToFP neural network.

    Improvement v5: keeps the existing BEV encoder and dense mask branch, but makes the polygon branch more RoomFormer-like by:
      1. exposing auxiliary corner-decoder outputs for deep supervision,
      2. keeping polygon/corner outputs as the primary final representation,
      3. leaving room masks as auxiliary support rather than final geometry.
    """
    def __init__(
        self,
        in_channels: int = 8,
        max_rooms: int = 20,
        max_corners: int = 40,
        embed_dim: int = 256,
        num_heads: int = 8,
        decoder_layers: int = 4,
        corner_decoder_layers: int = 2,
        room_classes: int = 12,
        dropout: float = 0.10,
        predict_furniture: bool = False,
        base: int = 48,
        mask_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.max_rooms = int(max_rooms)
        self.max_corners = int(max_corners)
        self.room_classes = int(room_classes)
        self.embed_dim = int(embed_dim)
        self.corner_decoder_layers = int(corner_decoder_layers)
        self.dropout = float(dropout)
        self.predict_furniture = bool(predict_furniture)
        self.base = int(base)
        self.mask_dim = int(mask_dim)

        self.encoder = ConvEncoder(in_channels=in_channels, base=base, embed_dim=embed_dim, dropout=dropout)
        self.pos_proj = nn.Linear(2, embed_dim)
        self.room_queries = nn.Embedding(max_rooms, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.valid_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.class_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, room_classes))

        self.corner_queries = nn.Embedding(max_corners, embed_dim)
        corner_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.corner_decoder = nn.TransformerDecoder(corner_decoder_layer, num_layers=corner_decoder_layers)

        self.room_reference_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
            nn.Sigmoid(),
        )
        self.corner_offset_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
            nn.Tanh(),
        )
        self.corner_valid_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mask_decoder = FPNMaskDecoder(embed_dim, base=base, out_dim=mask_dim, dropout=dropout)
        self.wall_head = nn.Conv2d(mask_dim, 1, 1)
        self.opening_head = nn.Conv2d(mask_dim, 1, 1)
        self.boundary_head = nn.Conv2d(mask_dim, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(mask_dim, 1, 1)

        self.room_mask_embed = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, mask_dim),
        )
        self.pixel_embed = nn.Conv2d(mask_dim, mask_dim, 1)
        self.furniture_head = nn.Conv2d(mask_dim, 1, 1) if self.predict_furniture else None

    def _positional_memory(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        memory = feat.flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(0.0, 1.0, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        pos = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)
        return memory + self.pos_proj(pos).expand(b, -1, -1)

    def _polygon_outputs_from_tokens(self, decoded: torch.Tensor, corner_decoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert room tokens and corner tokens into RoomFormer-like polygon predictions."""
        room_ref = self.room_reference_head(decoded).unsqueeze(2)
        corner_offsets = 0.45 * self.corner_offset_head(corner_decoded)
        corners = torch.clamp(room_ref + corner_offsets, 0.0, 1.0)
        return {
            "room_valid_logits": self.valid_head(decoded).squeeze(-1),
            "room_class_logits": self.class_head(decoded),
            "corner_xy": corners,
            "corner_valid_logits": self.corner_valid_head(corner_decoded).squeeze(-1),
            "room_reference_xy": room_ref.squeeze(2),
        }

    def _decode_room_queries(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Manual decoder loop keeps old checkpoint keys while allowing future auxiliary hooks."""
        hidden = queries
        for layer in self.decoder.layers:
            hidden = layer(hidden, memory)
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return hidden

    def _decode_corner_queries(self, decoded: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """
        Decode corner tokens and return auxiliary polygon predictions from intermediate corner-decoder layers.
        """
        b = decoded.shape[0]
        room_tokens = decoded.unsqueeze(2)
        corner_tokens = self.corner_queries.weight.view(1, 1, self.max_corners, self.embed_dim)
        hidden = (room_tokens + corner_tokens).reshape(b, self.max_rooms * self.max_corners, self.embed_dim)

        aux_outputs: list[dict[str, torch.Tensor]] = []
        num_layers = len(self.corner_decoder.layers)
        for layer_idx, layer in enumerate(self.corner_decoder.layers):
            hidden = layer(hidden, memory)
            shaped = hidden.view(b, self.max_rooms, self.max_corners, self.embed_dim)
            # Improvement v5: auxiliary decoder losses. Every intermediate corner-decoder layer is asked to predict polygons, similar in spirit to RoomFormer's intermediate decoder supervision.
            if layer_idx < num_layers - 1:
                aux_outputs.append(self._polygon_outputs_from_tokens(decoded, shaped))

        if self.corner_decoder.norm is not None:
            hidden = self.corner_decoder.norm(hidden)
        corner_decoded = hidden.view(b, self.max_rooms, self.max_corners, self.embed_dim)
        return corner_decoded, aux_outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat, skips = self.encoder(x, return_skips=True)
        b = feat.shape[0]
        memory = self._positional_memory(feat)

        queries = self.room_queries.weight.unsqueeze(0).expand(b, -1, -1)
        decoded = self._decode_room_queries(queries, memory)
        corner_decoded, aux_outputs = self._decode_corner_queries(decoded, memory)

        out = self._polygon_outputs_from_tokens(decoded, corner_decoded)
        if aux_outputs:
            out["aux_outputs"] = aux_outputs

        target_size = x.shape[-2:]
        mask_features = self.mask_decoder(feat, skips, target_size)
        out["wall_logits"] = self.wall_head(mask_features)
        out["opening_logits"] = self.opening_head(mask_features)
        out["boundary_logits"] = self.boundary_head(mask_features)
        out["corner_heatmap_logits"] = self.corner_heatmap_head(mask_features)

        pixel_features = F.normalize(self.pixel_embed(mask_features), dim=1)
        mask_embed = F.normalize(self.room_mask_embed(decoded), dim=-1)
        out["room_mask_logits"] = torch.einsum("brc,bchw->brhw", mask_embed, pixel_features)

        if self.furniture_head is not None:
            out["furniture_logits"] = self.furniture_head(mask_features)
        return out
