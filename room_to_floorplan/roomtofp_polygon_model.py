from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Takes the input BEV tensor and converts it into smaller feature maps for the Transformer

    8-channel BEV input -> CNN Encoder -> Compressed Spatial Feature Map -> Transformer decoder predicts polygons

    The encoder does not predict polygons, it just prepares evidence for the polygon model
    It receives input as (batch size x 8 x 512 x 512) where the 8 channels are:
        1. Occupancy - density of 3D points at a point in 2D space post-conversion. It answers if there is something observed in a scene or not (general evidence)
        2. Wall
        3. Floor
        4. Opening
        5. Furniture (Optional)
        6, 7, 8. RGB Channel
    """
    def __init__(self, in_channels: int = 8, base: int = 48, embed_dim: int = 256, dropout: float = 0.10):
        """
        input channels,
        starting number of CNN Feature Channels,
        final output feature dims expected by Transformer (Bx256xHxW). Flattened and passed into transformer,
        added dropout to prevent overfitting
        """
        super().__init__()
        self.base = int(base)
        self.embed_dim = int(embed_dim)

        # Improvement v3: widened the encoder into a multi-scale feature extractor so the model keeps high-resolution wall/opening evidence while still producing compact Transformer tokens.
        # first feature extraction block
        self.stem = nn.Sequential(
            # Input goes thru Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> ReLU
            # BN keeps feature values in stable range while training
            # FIRST CONV:
            # takes 8 input channels and outputs 48 channels with a 7x7 kernel
            nn.Conv2d(in_channels, base, 7, stride=2, padding=3),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),  # inplace modifies the tensor directly
            nn.Dropout2d(dropout),  # dropping whole feature maps

            # SECOND CONV
            # Keeps same num of channels, uses smaller kernel to refine features after larger conv
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        # DOWNSAMPLING BLOCKS
        # creates 2 additional cnn blocks
        # stem o/p -> b x 48 x 256 x 256, then:
        # down1 -> b x 96 x 128 x 128
        # down2 -> b x 192 x 64 x 64 -> imp since transformer wont process a huge grid directly
        self.down1 = self._block(base, base * 2, dropout)  # 48, 96
        self.down2 = self._block(base * 2, base * 4, dropout)  # 96, 192
        # Improvement v3: one more context block gives the Transformer stronger scene-level context without forcing masks to be predicted only at this low resolution.
        self.down3 = self._block(base * 4, base * 8, dropout)  # 192, 384

        # projection layer (1x1 conv -> changes num of channels w/o changing spatial size)
        # converts 192 -> 256 channels as expected by transformer
        self.proj = nn.Conv2d(base * 8, embed_dim, 1)

    @staticmethod
    def _block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
        """
        Creates repeated downsampling blocks
        downsample, extract feats, regularize, refine feats
        """
        return nn.Sequential(
            nn.MaxPool2d(2),  # halves spatial size -> keeps strongest activation in 2x2 area

            # FIRST CONV
            nn.Conv2d(c_in, c_out, 3, padding=1),  # 48 -> 96
            nn.BatchNorm2d(c_out),  # stable training
            nn.ReLU(inplace=True),  # non-linearity
            nn.Dropout2d(dropout),  # reduces overfitting

            # SECOND CONV
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_skips: bool = False):
        """
        b x 8 x 512 x 512 ->
        b x 48 x 256 x 256 ->
        b x 96 x 128 x 128 ->
        b x 192 x 64 x 64 ->
        b x 384 x 32 x 32 ->
        b x 256 x 32 x 32
        """
        s1 = self.stem(x)  # extracts early low-level features
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        feat = self.proj(s4)  # projects channels to embed_dim for transformer decoder

        # Improvement v3: return skip features for FPN/U-Net mask decoding instead of throwing away spatial detail.
        if return_skips:
            return feat, {"stem": s1, "down1": s2, "down2": s3, "down3": s4}
        return feat


class FPNMaskDecoder(nn.Module):
    """
    Lightweight FPN/U-Net style decoder for dense wall/opening/furniture masks.

    Improvement v3: this replaces the old 64x64 dense head with a multi-scale decoder that combines semantic context with high-resolution skips.
    This is closer to modern detection/segmentation backbones and is better suited to thin walls and sparse openings.
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
    Main RoomToFP Neural Network that combines CNN encoder with Transformer decoder and multiple output heads.
    Takes 8-channel top-down BEV inputs and predicts structured floorplan
    """

    def __init__(
        self,
        in_channels: int = 8,
        max_rooms: int = 20,  # max num of room polygons the model can predict per scene
        max_corners: int = 40,  # max num of corners each room polygon can have
        embed_dim: int = 256,
        num_heads: int = 8,  # controls multi-head attention in Transformer => model can look at scene in multiple ways at once
                             # eg: one head at wall layout, other at room boundary. Each head gets 256/8 = 32 dimensions
        decoder_layers: int = 4,  # num of stacked decoder layers, allows for more reasoning at cost of overfitting risk and compute
        corner_decoder_layers: int = 2,
        room_classes: int = 12,  # num of room types model can classify
        dropout: float = 0.10,
        predict_furniture: bool = False,
        base: int = 48,
        mask_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.max_rooms = max_rooms
        self.max_corners = max_corners
        self.room_classes = room_classes
        self.embed_dim = embed_dim
        self.corner_decoder_layers = int(corner_decoder_layers)
        self.dropout = dropout
        self.predict_furniture = bool(predict_furniture)
        self.base = int(base)
        self.mask_dim = int(mask_dim)

        self.encoder = ConvEncoder(in_channels=in_channels, base=base, embed_dim=embed_dim, dropout=dropout)

        # Improvement v3: explicit x/y positional memory gives the Transformer spatial awareness, which is essential for polygon coordinates.
        self.pos_proj = nn.Linear(2, embed_dim)

        # creates learned query embeddings (max_rooms x embed_dim)
        # each query is a learnable vector representing 1 room slot - inspired by DETR style object prediction
        self.room_queries = nn.Embedding(max_rooms, embed_dim)

        # transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,  # every token has feature size 256
            nhead=num_heads,  # num of attention heads
            dim_feedforward=embed_dim * 4,  # contains feedforward after attention => 1024
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # TRANSFORMER DECODER STACK
        # stacks 3 decoder layers
        # 1: rough room understanding, 2: better room/location reasoning, 3: refined polygon representation
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # room validity head: predicts if each room query responds to real room
        # 256 values -> 1 logit (raw model output converted to probability later using sigmoid)
        # LayerNorm normalizes each query vector before feeding it to the linear classifier which stabilizes preds
        self.valid_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # room class head: predicts room type, outputs 12 logits, 1 per class
        self.class_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, room_classes))

        # Improvement v4: replace the single room-vector corner MLP with two-level room/corner queries.
        # Each room keeps its room query, then each possible corner gets its own learned token inside that room.
        # This is much closer to RoomFormer-style polygon reasoning than predicting all corners from one compressed vector.
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

        # Improvement v4: predict a room-level reference point plus per-corner offsets.
        # This stabilises training because corners start near the decoded room location instead of being unconstrained MLP outputs.
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

        # Improvement v4: corner validity is now predicted from each corner token, not repeated from the room token.
        self.corner_valid_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # Structural dense heads. These support the main floorplan objective.
        # create dense segmentation heads for wall and opening masks
        # dense since they o/p pred for every pixel
        self.mask_decoder = FPNMaskDecoder(embed_dim, base=base, out_dim=mask_dim, dropout=dropout)
        self.wall_head = nn.Conv2d(mask_dim, 1, 1)
        self.opening_head = nn.Conv2d(mask_dim, 1, 1)
        self.boundary_head = nn.Conv2d(mask_dim, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(mask_dim, 1, 1)

        # Improvement v3: add query-mask prediction so each room query learns a room instance mask as well as polygon corners, similar in spirit to MaskFormer/RoomFormer-style room objects.
        self.room_mask_embed = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, mask_dim),
        )
        self.pixel_embed = nn.Conv2d(mask_dim, mask_dim, 1)

        # Optional furniture head. Disabled by default so the main model can focus
        # on walls/openings/polygons without noisy furniture supervision.
        self.furniture_head = nn.Conv2d(mask_dim, 1, 1) if self.predict_furniture else None

    @staticmethod
    def _dense_head(embed_dim: int, dropout: float) -> nn.Sequential:
        """
        Creates small segmentation head
        Used for wall, opening and furniture heads
        """
        # Mask prediction branch
        return nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),  # 256 -> 128
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim // 2, 1, 1),  # 1x1 conv
        )

    def _positional_memory(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Build flattened Transformer memory with explicit x/y position information.
        """
        b, c, h, w = feat.shape
        memory = feat.flatten(2).transpose(1, 2)  # flatten features for Transformer. Converts B x C x HxW -> B x HxW x C
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(0.0, 1.0, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        pos = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)
        return memory + self.pos_proj(pos).expand(b, -1, -1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Input: B x 8 x H x W
        Output: Dictionary of Preds

        Model's data flow
        """
        # input bev tensor passed thru CNN Encoder
        feat, skips = self.encoder(x, return_skips=True)
        b, _, _, _ = feat.shape  # get batch size

        memory = self._positional_memory(feat)
        # room_queries has weight 20x256, unsqueeze adds batch dimension and expand repeats it for every sample in the batch
        queries = self.room_queries.weight.unsqueeze(0).expand(b, -1, -1)

        # Transformer Decoding
        # Before decoding, query is a learned vector, after decoding, it has actual information
        decoded = self.decoder(queries, memory)

        # Improvement v4: decode ordered corner tokens for every room query instead of using one room embedding to emit all vertices.
        # Shapes: room tokens [B,R,D], corner tokens [B,R,C,D], flattened corner queries [B,R*C,D].
        room_tokens = decoded.unsqueeze(2)
        corner_tokens = self.corner_queries.weight.view(1, 1, self.max_corners, self.embed_dim)
        corner_queries = (room_tokens + corner_tokens).reshape(b, self.max_rooms * self.max_corners, self.embed_dim)
        corner_decoded = self.corner_decoder(corner_queries, memory).view(b, self.max_rooms, self.max_corners, self.embed_dim)

        # Improvement v4: coordinates are generated as bounded offsets around a room reference point.
        room_ref = self.room_reference_head(decoded).unsqueeze(2)
        corner_offsets = 0.45 * self.corner_offset_head(corner_decoded)
        corners = torch.clamp(room_ref + corner_offsets, 0.0, 1.0)

        # gets original spatial size of i/p
        target_size = x.shape[-2:]

        mask_features = self.mask_decoder(feat, skips, target_size)
        wall_logits = self.wall_head(mask_features)
        opening_logits = self.opening_head(mask_features)
        boundary_logits = self.boundary_head(mask_features)
        corner_heatmap_logits = self.corner_heatmap_head(mask_features)

        # Improvement v3: query masks let room queries learn spatial room extents; polygons can then be trained together with room instance masks.
        pixel_features = F.normalize(self.pixel_embed(mask_features), dim=1)
        mask_embed = F.normalize(self.room_mask_embed(decoded), dim=-1)
        room_mask_logits = torch.einsum("brc,bchw->brhw", mask_embed, pixel_features)

        out = {
            "room_valid_logits": self.valid_head(decoded).squeeze(-1),  # B x maxRooms x 1 becomes B x maxRooms
            "room_class_logits": self.class_head(decoded),
            "corner_xy": corners,
            "corner_valid_logits": self.corner_valid_head(corner_decoded).squeeze(-1),
            # Improvement v4: expose room reference points for debugging/visualisation and future refinement losses.
            "room_reference_xy": room_ref.squeeze(2),
            "wall_logits": wall_logits,
            "opening_logits": opening_logits,
            "boundary_logits": boundary_logits,
            "corner_heatmap_logits": corner_heatmap_logits,
            "room_mask_logits": room_mask_logits,
        }
        if self.furniture_head is not None:
            out["furniture_logits"] = self.furniture_head(mask_features)
        return out
