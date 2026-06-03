from __future__ import annotations

"""
RoomToFP v9 model.

Major v7 upgrades:
- Improvement v7: ResNet/FPN-style stronger BEV backbone.
- Improvement v7: Transformer encoder over global BEV memory.
- Improvement v7: pure-PyTorch lightweight multi-scale deformable-style cross attention.
- Improvement v7: iterative coordinate refinement for polygon corners.
- Improvement v7: optional denoising polygon refinement branch used during training.
- Improvement v9: accepts 10-channel BEV input from the corrected V9 preprocessing pipeline
- Improvement v9: adds a dense boundary-sequence decoder for boundary_64 and boundary_128 targets
- Improvement v9: adds a distance-to-wall auxiliary dense prediction head

The class name remains RoomToFPPolygonNet so existing training/inference imports keep working
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Utilities
# Defines a logit helper used to refine normalized coordinates while keeping them in [0, 1].
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Clamps probabilities away from 0 and 1 to avoid infinities in the logit calculation.
    x = x.clamp(min=eps, max=1.0 - eps)
    # Applies the inverse sigmoid formula log(x / (1 - x)).
    return torch.log(x / (1.0 - x))


# Defines a reusable multi-layer perceptron used by several prediction heads
class MLP(nn.Module):
    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Creates an empty list that will be filled with Linear/ReLU layers.
        layers: list[nn.Module] = []
        # Tracks the current input dimension while building the MLP.
        d = in_dim
        # Adds hidden MLP layers before the final output layer.
        for i in range(num_layers - 1):
            # Adds one Linear layer followed by ReLU activation to the MLP.
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            # Updates the next layer input size to the hidden dimension.
            d = hidden_dim
        # Adds the final Linear layer that produces the requested output dimension.
        layers.append(nn.Linear(d, out_dim))
        # Wraps all MLP layers into one sequential PyTorch module.
        self.net = nn.Sequential(*layers)

    # Runs the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passes the input through the sequential MLP layers.
        return self.net(x)


# Defines DETR-style 2D positional encoding so Transformer tokens retain BEV location information.
class SinePositionEncoding2D(nn.Module):
    """DETR-style 2D sine/cosine positional encoding for BEV feature maps."""

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, dim: int = 256, temperature: int = 10000):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Computes or stores an intermediate value used by the following model operation.
        if dim % 4 != 0:
            # Stops execution early if the supplied configuration would make tensor shapes invalid.
            raise ValueError("Position encoding dim must be divisible by 4")
        # Stores the positional encoding channel dimension.
        self.dim = int(dim)
        # Stores the frequency scaling constant for sine/cosine position encoding.
        self.temperature = int(temperature)

    # Runs the forward pass for this module.
    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        # Creates normalized y and x coordinate grids for every BEV feature-map location.
        y, x = torch.meshgrid(
            # Creates evenly spaced normalized coordinates along one spatial axis.
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            # Creates evenly spaced normalized coordinates along one spatial axis.
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            # Uses matrix-style indexing so y corresponds to rows and x to columns.
            indexing="ij",
        )
        # Creates the frequency indices used by sine/cosine positional encoding.
        dim_t = torch.arange(self.dim // 4, device=device, dtype=dtype)
        # Converts frequency indices into wavelength scales.
        dim_t = self.temperature ** (2 * (dim_t // 2) / max(1, self.dim // 4))

        # Scales x coordinates by the positional encoding frequencies.
        px = x[..., None] / dim_t
        # Scales y coordinates by the positional encoding frequencies.
        py = y[..., None] / dim_t
        # Builds sine and cosine features for x positions.
        pos_x = torch.stack((px.sin(), px.cos()), dim=-1).flatten(-2)
        # Builds sine and cosine features for y positions.
        pos_y = torch.stack((py.sin(), py.cos()), dim=-1).flatten(-2)
        # Concatenates y and x positional features into one encoding tensor.
        pos = torch.cat((pos_y, pos_x), dim=-1)
        # Trims the encoding to the requested embedding dimension.
        pos = pos[..., : self.dim]
        # Returns positional encoding in 1 x C x H x W format for addition to BEV features.
        return pos.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W


# Stronger ResNet/FPN backbone
# Defines a ResNet basic residual block for stable deeper CNN feature extraction.
class BasicBlock(nn.Module):
    # Keeps the ResNet block output channel expansion factor at 1.
    expansion = 1

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, c_in: int, c_out: int, stride: int = 1, dropout: float = 0.0):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Creates the first 3x3 convolution of the residual block, optionally downsampling with stride.
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        # Normalizes the first convolution output to stabilize training.
        self.bn1 = nn.BatchNorm2d(c_out)
        # Creates the second 3x3 convolution of the residual block.
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        # Normalizes the second convolution output before residual addition.
        self.bn2 = nn.BatchNorm2d(c_out)
        # Uses spatial dropout when requested; otherwise this becomes a no-op identity layer.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        # Checks whether the shortcut must change spatial size or channel count.
        if stride != 1 or c_in != c_out:
            # Builds a projection shortcut so the residual branch and identity branch have matching shapes.
            self.down = nn.Sequential(
                # Uses a 1x1 convolution to match residual shortcut channels and stride.
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                # Normalizes the projected residual shortcut.
                nn.BatchNorm2d(c_out),
            )
        # Handles the fallback case when the previous condition is not met.
        else:
            # Uses an unchanged identity shortcut when input and output shapes already match.
            self.down = nn.Identity()

    # Runs the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computes the residual shortcut branch.
        identity = self.down(x)
        # Applies Conv-BatchNorm-ReLU for the first residual branch operation.
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # Applies dropout to reduce overfitting in the convolutional block.
        out = self.drop(out)
        # Applies the second convolution and batch normalization before adding the shortcut.
        out = self.bn2(self.conv2(out))
        # Adds the shortcut connection and applies ReLU, completing the ResNet block.
        out = F.relu(out + identity, inplace=True)
        # Returns the final tensor or output dictionary from this module.
        return out


# Defines the CNN backbone that extracts multi-scale BEV features p2 through p5.
class ResNetFPNEncoder(nn.Module):
    """
    Improvement v7: stronger ResNet/FPN encoder.

    Input Bx10x512x512 -> multi-scale FPN maps:
      p2: BxCx128x128
      p3: BxCx64x64
      p4: BxCx32x32
      p5: BxCx16x16

    p4 is also passed through a Transformer encoder as global memory.
    """

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, in_channels: int = 10, base: int = 48, embed_dim: int = 256, dropout: float = 0.10):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Stores the shared internal feature dimension.
        self.embed_dim = int(embed_dim)
        # Converts the base CNN width to an integer channel count.
        b = int(base)
        # Improvement v9: creates the initial CNN stem that downsamples the 10-channel BEV input
        self.stem = nn.Sequential(
            # Improvement v9: maps the configurable BEV input channels to the base CNN width using a 7x7 stride-2 convolution
            nn.Conv2d(in_channels, b, 7, stride=2, padding=3, bias=False),  # 256x256
            # Normalizes the stem feature channels.
            nn.BatchNorm2d(b),
            # Applies ReLU non-linearity after convolutional normalization.
            nn.ReLU(inplace=True),
            # Adds spatial dropout for regularization.
            nn.Dropout2d(dropout),
            # Adds a residual block at the current stem resolution.
            BasicBlock(b, b, stride=1, dropout=dropout),
        )
        # Creates residual blocks that refine 256x256 features without further downsampling.
        self.layer1 = nn.Sequential(BasicBlock(b, b, 1, dropout), BasicBlock(b, b, 1, dropout))  # 256
        # Creates residual blocks that downsample to 128x128 and double channels.
        self.layer2 = nn.Sequential(BasicBlock(b, b * 2, 2, dropout), BasicBlock(b * 2, b * 2, 1, dropout))  # 128
        # Creates residual blocks that downsample to 64x64 and increase feature capacity.
        self.layer3 = nn.Sequential(BasicBlock(b * 2, b * 4, 2, dropout), BasicBlock(b * 4, b * 4, 1, dropout))  # 64
        # Creates residual blocks that downsample to 32x32 for broader layout context.
        self.layer4 = nn.Sequential(BasicBlock(b * 4, b * 8, 2, dropout), BasicBlock(b * 8, b * 8, 1, dropout))  # 32
        # Creates residual blocks that downsample to 16x16 for the deepest semantic features.
        self.layer5 = nn.Sequential(BasicBlock(b * 8, b * 8, 2, dropout), BasicBlock(b * 8, b * 8, 1, dropout))  # 16

        # Projects a backbone feature map to the shared FPN embedding dimension with a 1x1 convolution.
        self.lat2 = nn.Conv2d(b * 2, embed_dim, 1)
        # Projects a backbone feature map to the shared FPN embedding dimension with a 1x1 convolution.
        self.lat3 = nn.Conv2d(b * 4, embed_dim, 1)
        # Projects a backbone feature map to the shared FPN embedding dimension with a 1x1 convolution.
        self.lat4 = nn.Conv2d(b * 8, embed_dim, 1)
        # Projects a backbone feature map to the shared FPN embedding dimension with a 1x1 convolution.
        self.lat5 = nn.Conv2d(b * 8, embed_dim, 1)
        # Refines an FPN level with a 3x3 convolution after top-down fusion.
        self.out2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        # Refines an FPN level with a 3x3 convolution after top-down fusion.
        self.out3 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        # Refines an FPN level with a 3x3 convolution after top-down fusion.
        self.out4 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        # Refines an FPN level with a 3x3 convolution after top-down fusion.
        self.out5 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

    # Runs the forward pass for this module.
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Runs the input BEV tensor through the initial CNN stem.
        c1 = self.stem(x)
        # Refines the high-resolution stem features.
        c1 = self.layer1(c1)
        # Creates the 128x128 backbone feature map.
        c2 = self.layer2(c1)
        # Creates the 64x64 backbone feature map.
        c3 = self.layer3(c2)
        # Creates the 32x32 backbone feature map.
        c4 = self.layer4(c3)
        # Creates the 16x16 backbone feature map.
        c5 = self.layer5(c4)

        # Starts top-down FPN fusion from the deepest feature map.
        p5 = self.lat5(c5)
        # Fuses 32x32 features with upsampled 16x16 context.
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        # Fuses 64x64 features with upsampled 32x32 context.
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        # Fuses 128x128 features with upsampled 64x64 context.
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        # Returns all FPN feature maps after 3x3 refinement and ReLU activation.
        return {
            # High-resolution FPN feature map useful for dense masks and local detail.
            "p2": F.relu(self.out2(p2), inplace=True),
            # Medium-resolution FPN feature map with more context than p2.
            "p3": F.relu(self.out3(p3), inplace=True),
            # Lower-resolution FPN feature map used as global Transformer memory.
            "p4": F.relu(self.out4(p4), inplace=True),
            # Deepest FPN feature map with strongest global/semantic context.
            "p5": F.relu(self.out5(p5), inplace=True),
            # Keeps stem features available for debugging or future extensions.
            "stem": c1,
        }


# Multi-scale deformable-style attention
# Defines lightweight deformable-style attention for sampling useful locations from FPN levels.
class MSDeformAttnLite(nn.Module):
    """
    Improvement v7: pure-PyTorch lightweight multi-scale deformable-style attention.

    This approximates Deformable DETR attention without custom CUDA ops.
    Each query predicts offsets and attention weights around a reference point.
    Values are sampled from each FPN level using grid_sample.

    It is slower than CUDA deformable attention but much easier to run in a normal PyTorch environment.
    """

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_levels: int = 4, num_points: int = 4):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Checks that each attention head receives an equal feature dimension.
        if embed_dim % num_heads != 0:
            # Stops execution early if the supplied configuration would make tensor shapes invalid.
            raise ValueError("embed_dim must be divisible by num_heads")
        # Stores the shared internal feature dimension.
        self.embed_dim = int(embed_dim)
        # Stores the number of parallel attention heads.
        self.num_heads = int(num_heads)
        # Stores the number of FPN feature levels used for attention.
        self.num_levels = int(num_levels)
        # Stores the number of sampled points per level/head.
        self.num_points = int(num_points)
        # Computes the feature dimension handled by each attention head.
        self.head_dim = embed_dim // num_heads

        # Creates a 1x1 value projection for each FPN level before sampling.
        self.value_proj = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(num_levels)])
        # Predicts x/y offsets around each reference point for every head, level and sample point.
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        # Predicts attention weights for all sampled points.
        self.weight_proj = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        # Projects the combined sampled features back into the embedding dimension.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Starts offsets at zero so early sampling begins near the reference point.
        nn.init.zeros_(self.offset_proj.weight)
        # Starts offsets at zero so early sampling begins near the reference point.
        nn.init.zeros_(self.offset_proj.bias)

    # Runs the forward pass for this module.
    def forward(self, query: torch.Tensor, multi_feats: list[torch.Tensor], reference_xy: torch.Tensor) -> torch.Tensor:
        """
        query: B,Q,C
        multi_feats: list of B,C,H,W maps
        reference_xy: B,Q,2 in [0,1]
        """
        # Reads batch size, query count and feature dimension.
        b, q, c = query.shape
        # Uses only the available FPN levels, capped by configured num_levels.
        l = min(len(multi_feats), self.num_levels)
        # Predicts sampling offsets and reshapes them by batch/query/head/level/point/xy.
        offsets = self.offset_proj(query).view(b, q, self.num_heads, self.num_levels, self.num_points, 2)
        # Predicts raw attention weights and reshapes them by batch/query/head/level/point.
        weights = self.weight_proj(query).view(b, q, self.num_heads, self.num_levels, self.num_points)
        # Applies softmax across all sampled level-point combinations.
        weights = F.softmax(weights[:, :, :, :l].reshape(b, q, self.num_heads, -1), dim=-1)
        # Restores attention weights to B x Q x heads x levels x points.
        weights = weights.view(b, q, self.num_heads, l, self.num_points)

        # Initializes the output accumulator for all attention heads.
        out = query.new_zeros((b, q, self.num_heads, self.head_dim))
        # Keeps reference coordinates inside normalized BEV bounds.
        ref = reference_xy.clamp(0.0, 1.0)

        # Iterates through each FPN level and samples features around the reference points.
        for li in range(l):
            # Projects the current FPN level into value features.
            feat = self.value_proj[li](multi_feats[li])
            # Reads the spatial size of the current feature map.
            _, _, h, w = feat.shape
            # Splits channels into attention heads.
            feat = feat.view(b, self.num_heads, self.head_dim, h, w)
            # Merges batch and head dimensions so grid_sample can process each head separately.
            feat = feat.reshape(b * self.num_heads, self.head_dim, h, w)

            # Convert offset from normalized image fraction. Small initial offsets around ref.
            # Tanh prevents huge samples during early training.
            # Bounds learned offsets and scales them to a local region around each reference point.
            off = torch.tanh(offsets[:, :, :, li]) * 0.12  # B,Q,Hd,P,2
            # Adds learned offsets to query reference points.
            sample_xy = ref[:, :, None, None, :] + off
            # Keeps sampling coordinates inside the normalized BEV range.
            sample_xy = sample_xy.clamp(0.0, 1.0)
            # Converts coordinates from [0, 1] to [-1, 1] for PyTorch grid_sample.
            grid = sample_xy * 2.0 - 1.0  # B,Q,Hd,P,2
            # Rearranges grid coordinates into the shape expected by grid_sample.
            grid = grid.permute(0, 2, 1, 3, 4).reshape(b * self.num_heads, q, self.num_points, 2)

            # Samples feature values at the learned offset locations using bilinear interpolation.
            sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            # sampled: B*heads, head_dim, Q, P
            # Reshapes sampled features back to B x Q x heads x points x head_dim.
            sampled = sampled.view(b, self.num_heads, self.head_dim, q, self.num_points).permute(0, 3, 1, 4, 2)
            # Selects attention weights for the current FPN level.
            wgt = weights[:, :, :, li].unsqueeze(-1)
            # Adds the weighted sum of sampled features to the output accumulator.
            out = out + (sampled * wgt).sum(dim=3)

        # Combines all attention heads back into the full embedding dimension.
        out = out.reshape(b, q, c)
        # Returns the final projected attention output.
        return self.out_proj(out)


# Defines one Transformer-style decoder layer with self-attention, cross-attention and FFN.
class V7DecoderLayer(nn.Module):
    """Self-attention + MS deformable-style cross attention + FFN."""

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.10, num_levels: int = 4, num_points: int = 4):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Lets query tokens communicate with each other.
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Lets query tokens sample scene evidence from multi-scale FPN features.
        self.cross_attn = MSDeformAttnLite(embed_dim, num_heads, num_levels, num_points)
        # Defines the feed-forward subnetwork used after attention updates.
        self.ffn = nn.Sequential(
            # Expands the feature dimension before nonlinear processing.
            nn.Linear(embed_dim, embed_dim * 4),
            # Applies ReLU non-linearity after convolutional normalization.
            nn.ReLU(inplace=True),
            # Adds dropout for regularization.
            nn.Dropout(dropout),
            # Projects features back to the original embedding dimension.
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # Normalizes after self-attention residual addition.
        self.norm1 = nn.LayerNorm(embed_dim)
        # Normalizes after cross-attention residual addition.
        self.norm2 = nn.LayerNorm(embed_dim)
        # Normalizes after FFN residual addition.
        self.norm3 = nn.LayerNorm(embed_dim)
        # Adds dropout to attention/FFN residual branches.
        self.drop = nn.Dropout(dropout)

    # Runs the forward pass for this module.
    def forward(self, x: torch.Tensor, multi_feats: list[torch.Tensor], reference_xy: torch.Tensor) -> torch.Tensor:
        # Applies self-attention so queries can exchange information.
        sa, _ = self.self_attn(x, x, x, need_weights=False)
        # Adds the self-attention residual and normalizes it.
        x = self.norm1(x + self.drop(sa))
        # Applies deformable-style cross-attention from queries to BEV/FPN features.
        ca = self.cross_attn(x, multi_feats, reference_xy)
        # Adds the cross-attention residual and normalizes it.
        x = self.norm2(x + self.drop(ca))
        # Applies the FFN residual block and final normalization.
        x = self.norm3(x + self.drop(self.ffn(x)))
        # Returns the updated query tokens.
        return x


# Dense mask decoder
# Defines an FPN-style decoder for dense wall/opening/boundary/corner/room mask features.
class V7FPNMaskDecoder(nn.Module):
    """
    FPN decoder using p2/p3/p4/p5 for dense wall/opening/boundary/corner/room masks.
    """

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(self, embed_dim: int = 256, out_dim: int = 128, dropout: float = 0.10):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Projects p5 FPN features into the mask feature dimension.
        self.proj5 = nn.Conv2d(embed_dim, out_dim, 1)
        # Projects p4 FPN features into the mask feature dimension.
        self.proj4 = nn.Conv2d(embed_dim, out_dim, 1)
        # Projects p3 FPN features into the mask feature dimension.
        self.proj3 = nn.Conv2d(embed_dim, out_dim, 1)
        # Projects p2 FPN features into the mask feature dimension.
        self.proj2 = nn.Conv2d(embed_dim, out_dim, 1)
        # Refines the fused high-resolution mask features using convolutional blocks.
        self.refine = nn.Sequential(
            # Applies a 3x3 convolution to refine the mask feature map.
            nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            # Applies another 3x3 convolution to further refine dense mask features.
            nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )

    # Runs the forward pass for this module.
    def forward(self, feats: dict[str, torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        # Starts dense top-down fusion from the deepest FPN level.
        p5 = self.proj5(feats["p5"])
        # Fuses p4 with upsampled p5 context.
        p4 = self.proj4(feats["p4"]) + F.interpolate(p5, size=feats["p4"].shape[-2:], mode="nearest")
        # Fuses p3 with upsampled p4 context.
        p3 = self.proj3(feats["p3"]) + F.interpolate(p4, size=feats["p3"].shape[-2:], mode="nearest")
        # Fuses p2 with upsampled p3 context for high-resolution detail.
        p2 = self.proj2(feats["p2"]) + F.interpolate(p3, size=feats["p2"].shape[-2:], mode="nearest")
        # Refines the fused p2 mask feature.
        y = self.refine(p2)
        # Upsamples mask features back to the original BEV input size.
        return F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)


# Main model
# Defines the full RoomToFP v7 network used by training and inference.
class RoomToFPPolygonNet(nn.Module):
    """
    RoomToFP v9 network.

    Keeps the existing output dictionary keys used by training/inference:
      room_valid_logits, room_class_logits, corner_valid_logits, corner_xy,
      wall_logits, opening_logits, room_mask_logits, boundary_logits,
      corner_heatmap_logits, optional furniture_logits, aux_outputs, dn_outputs,
      boundary64_xy, boundary128_xy, boundary_point_valid_logits, distance_logits.
    """

    # Initializes the module and creates all layers/parameters used later in forward passes.
    def __init__(
        # Continues the current multi-line statement or argument list.
        self,
        # Improvement v9: number of input BEV channels, normally 10 for occupancy/wall/floor/opening/furniture/RGB plus observed corner prior and distance-to-observed-wall
        in_channels: int = 10,
        # Maximum number of room queries predicted per scene.
        max_rooms: int = 20,
        # Maximum number of polygon corner slots per room.
        max_corners: int = 16,
        # Shared Transformer/query/FPN feature dimension.
        embed_dim: int = 256,
        # Number of room type classes.
        room_classes: int = 12,
        # Dropout probability used throughout the model.
        dropout: float = 0.10,
        # Enables an optional furniture mask head when True.
        predict_furniture: bool = False,
        # Starting CNN channel width for the ResNet/FPN backbone.
        base: int = 48,
        # Channel width used by the dense mask decoder.
        mask_dim: int = 128,
        # Number of room-query decoder layers.
        decoder_layers: int = 4,
        # Number of corner-query decoder/refinement layers.
        corner_decoder_layers: int = 3,
        # Number of multi-head attention heads.
        num_heads: int = 8,
        # Number of Transformer encoder layers applied to p4 global memory.
        transformer_encoder_layers: int = 2,
        # Number of sampled points per level/head in deformable-style attention.
        deform_points: int = 4,
        # Improvement v9: number of dense boundary sequence points predicted per room
        max_boundary_points: int = 128,
        # Improvement v9: number of decoder layers used for dense boundary sequence prediction
        boundary_decoder_layers: int = 3,
        # Accepts and ignores extra config arguments so older scripts do not break.
        **_: object,
    ):
        # Initializes the parent nn.Module so PyTorch can register layers and parameters correctly.
        super().__init__()
        # Stores model input channel count.
        self.in_channels = int(in_channels)
        # Stores maximum room query count.
        self.max_rooms = int(max_rooms)
        # Stores maximum polygon corner count.
        self.max_corners = int(max_corners)
        # Improvement v9: stores maximum dense boundary point count used for boundary_128 supervision
        self.max_boundary_points = int(max_boundary_points)
        # Stores the shared internal feature dimension.
        self.embed_dim = int(embed_dim)
        # Stores number of room classes.
        self.room_classes = int(room_classes)
        # Stores whether furniture prediction is enabled.
        self.predict_furniture = bool(predict_furniture)

        # Creates the ResNet/FPN BEV encoder.
        self.encoder = ResNetFPNEncoder(in_channels, base=base, embed_dim=embed_dim, dropout=dropout)
        # Creates positional encodings for Transformer global memory.
        self.posenc = SinePositionEncoding2D(embed_dim)
        # Creates one Transformer encoder layer for p4 global BEV memory.
        enc_layer = nn.TransformerEncoderLayer(
            # Sets the Transformer feature dimension.
            d_model=embed_dim,
            # Sets the number of attention heads.
            nhead=num_heads,
            # Sets FFN hidden dimension inside the Transformer layer.
            dim_feedforward=embed_dim * 4,
            # Applies dropout inside the Transformer layer.
            dropout=dropout,
            # Uses B x sequence x C tensor layout.
            batch_first=True,
            # Uses pre-norm Transformer behavior for more stable training.
            norm_first=True,
        )
        # Stacks Transformer encoder layers to create global BEV memory.
        self.memory_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(transformer_encoder_layers))

        # Learns one embedding per FPN level so attention can distinguish p2/p3/p4/p5 features.
        self.level_embed = nn.Parameter(torch.zeros(4, embed_dim))
        # Initializes FPN level embeddings with small random values.
        nn.init.normal_(self.level_embed, std=0.02)

        # Creates learnable room queries, where each query represents one possible room.
        self.room_queries = nn.Embedding(max_rooms, embed_dim)
        # Creates learnable corner queries, where each query represents one possible corner slot.
        self.corner_queries = nn.Embedding(max_corners, embed_dim)
        # Predicts initial room reference points in normalized [0, 1] BEV coordinates.
        self.room_ref_head = MLP(embed_dim, embed_dim, 2, 3)
        # Builds the stack of room-query decoder layers.
        self.room_decoder = nn.ModuleList([
            # Each layer updates room queries using self-attention and multi-scale cross-attention.
            V7DecoderLayer(embed_dim, num_heads, dropout, num_levels=4, num_points=deform_points)
            # Repeats decoder layer creation for the configured number of layers.
            for _ in range(int(decoder_layers))
        ])

        # Predicts initial corner reference points before iterative refinement.
        self.corner_init_ref = MLP(embed_dim, embed_dim, 2, 3)
        # Builds the stack of corner-query decoder layers.
        self.corner_decoder = nn.ModuleList([
            # Each layer refines corner tokens using scene features and current corner references.
            V7DecoderLayer(embed_dim, num_heads, dropout, num_levels=4, num_points=deform_points)
            # Repeats decoder layer creation for the configured number of corner decoder layers.
            for _ in range(int(corner_decoder_layers))
        ])
        # Creates one coordinate-delta head per corner refinement layer.
        self.corner_delta_heads = nn.ModuleList([MLP(embed_dim, embed_dim, 2, 3) for _ in range(int(corner_decoder_layers))])

        # Improvement v9: creates learnable dense boundary point queries used to predict boundary_128-style room outlines
        self.boundary_queries = nn.Embedding(max_boundary_points, embed_dim)
        # Improvement v9: predicts initial dense boundary reference points before iterative boundary refinement
        self.boundary_init_ref = MLP(embed_dim, embed_dim, 2, 3)
        # Improvement v9: builds a separate dense boundary decoder so boundary_128 supervision does not overload the sparse corner decoder
        self.boundary_decoder = nn.ModuleList([
            # Improvement v9: each layer refines boundary point tokens using scene features and current boundary references
            V7DecoderLayer(embed_dim, num_heads, dropout, num_levels=4, num_points=deform_points)
            # Improvement v9: repeats decoder layer creation for the configured number of boundary decoder layers
            for _ in range(int(boundary_decoder_layers))
        ])
        # Improvement v9: creates one coordinate-delta head per dense boundary refinement layer
        self.boundary_delta_heads = nn.ModuleList([MLP(embed_dim, embed_dim, 2, 3) for _ in range(int(boundary_decoder_layers))])
        # Improvement v9: predicts whether each dense boundary point slot is valid
        self.boundary_valid_head = MLP(embed_dim, embed_dim, 1, 3)

        # Predicts whether each room query corresponds to a real room.
        self.room_valid_head = MLP(embed_dim, embed_dim, 1, 3)
        # Predicts the room type/class for each room query.
        self.class_head = MLP(embed_dim, embed_dim, room_classes, 3)
        # Predicts whether each predicted corner slot is valid.
        self.corner_valid_head = MLP(embed_dim, embed_dim, 1, 3)

        # Creates the dense FPN mask decoder for wall/opening/boundary/corner/room masks.
        self.mask_decoder = V7FPNMaskDecoder(embed_dim, out_dim=mask_dim, dropout=dropout)
        # Dense 1x1 head for wall mask logits.
        self.wall_head = nn.Conv2d(mask_dim, 1, 1)
        # Dense 1x1 head for opening/door/window mask logits.
        self.opening_head = nn.Conv2d(mask_dim, 1, 1)
        # Dense 1x1 head for room boundary mask logits.
        self.boundary_head = nn.Conv2d(mask_dim, 1, 1)
        # Dense 1x1 head for corner heatmap logits.
        self.corner_heatmap_head = nn.Conv2d(mask_dim, 1, 1)
        # Improvement v9: dense 1x1 head for distance-to-wall regression logits
        self.distance_head = nn.Conv2d(mask_dim, 1, 1)
        # Converts room query features into mask embeddings for query-conditioned room masks.
        self.room_mask_embed = MLP(embed_dim, embed_dim, mask_dim, 3)
        # Optional furniture mask head, disabled unless predict_furniture=True.
        self.furniture_head = nn.Conv2d(mask_dim, 1, 1) if self.predict_furniture else None

        # Improvement v7: denoising query branch. It refines noisy GT polygons during training.
        # Embeds noisy denoising polygon coordinates into the model feature dimension.
        self.dn_coord_embed = nn.Linear(2, embed_dim)
        # Embeds corner-valid flags for denoising tokens.
        self.dn_valid_embed = nn.Embedding(2, embed_dim)
        # Embeds room IDs for denoising tokens.
        self.dn_room_embed = nn.Embedding(max_rooms, embed_dim)
        # Predicts denoising corner-valid logits.
        self.dn_corner_valid_head = MLP(embed_dim, embed_dim, 1, 3)
        # Predicts denoising room-valid logits.
        self.dn_room_valid_head = MLP(embed_dim, embed_dim, 1, 3)
        # Predicts denoising room class logits.
        self.dn_class_head = MLP(embed_dim, embed_dim, room_classes, 3)
        # Predicts denoising coordinate deltas at each refinement layer.
        self.dn_delta_heads = nn.ModuleList([MLP(embed_dim, embed_dim, 2, 3) for _ in range(int(corner_decoder_layers))])

    # Encodes p4 with a Transformer and prepares p2/p3/p4/p5 for multi-scale attention.
    def _encode_memory(self, feats: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # Transformer encoder on p4 global memory.
        # Selects the p4 FPN feature map as global memory.
        p4 = feats["p4"]
        # Reads batch/channel/height/width from p4.
        b, c, h, w = p4.shape
        # Builds 2D sine/cosine positional encoding for p4.
        pos = self.posenc(h, w, p4.device, p4.dtype)
        # Adds positional encoding and flattens p4 into B x sequence x C tokens.
        mem = (p4 + pos).flatten(2).transpose(1, 2)
        # Applies the Transformer encoder so p4 tokens can exchange global context.
        mem = self.memory_encoder(mem)
        # Reshapes encoded memory back into B x C x H x W format.
        p4_enc = mem.transpose(1, 2).reshape(b, c, h, w)

        # Creates the multi-scale feature list used by deformable-style cross-attention.
        multi_feats = [feats["p2"], feats["p3"], p4_enc, feats["p5"]]
        # Creates an output list for level-embedded multi-scale features.
        out: list[torch.Tensor] = []
        # Adds a learned level embedding to each FPN feature map.
        for i, f in enumerate(multi_feats):
            # Adds the FPN level identity embedding to distinguish p2/p3/p4/p5.
            out.append(f + self.level_embed[i].view(1, -1, 1, 1))
        # Replaces p4 in the feature dict with Transformer-encoded p4 for later mask decoding.
        feats["p4"] = p4_enc
        # Returns the list of encoded multi-scale features.
        return out

    # Decodes room queries into room tokens and reference points.
    def _decode_rooms(self, multi_feats: list[torch.Tensor], b: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Repeats learnable room queries across the batch.
        room_tokens = self.room_queries.weight.unsqueeze(0).repeat(b, 1, 1)
        # Predicts initial room reference coordinates in [0, 1].
        room_ref = torch.sigmoid(self.room_ref_head(room_tokens))
        # Applies each room decoder layer to refine room tokens and room references.
        for layer in self.room_decoder:
            # Updates room tokens using self-attention and deformable-style BEV cross-attention.
            room_tokens = layer(room_tokens, multi_feats, room_ref)
            # light iterative room reference update
            # Slightly refines room reference points in logit space after each decoder layer.
            room_ref = torch.sigmoid(inverse_sigmoid(room_ref) + 0.1 * torch.tanh(self.room_ref_head(room_tokens)))
        # Returns refined room tokens and their reference coordinates.
        return room_tokens, room_ref

    # Decodes polygon corners for each room query.
    def _decode_corners(self, room_tokens: torch.Tensor, room_ref: torch.Tensor, multi_feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        # Reads batch size, number of room queries and feature dimension.
        b, m, c = room_tokens.shape
        # Stores the number of corner slots per room.
        n = self.max_corners
        # Reshapes learnable corner queries so they can be added to every room token.
        corner_base = self.corner_queries.weight.view(1, 1, n, c)
        # Combines each room token with each corner query to create room-specific corner tokens.
        tokens = room_tokens.unsqueeze(2) + corner_base
        # Flattens room-corner tokens into one sequence for the decoder.
        tokens = tokens.reshape(b, m * n, c)
        # Predicts initial normalized corner reference coordinates.
        init_ref = torch.sigmoid(self.corner_init_ref(tokens))
        # Bias initial corner refs around room refs.
        # Repeats each room reference point for all corner slots in that room.
        room_ref_flat = room_ref.unsqueeze(2).expand(b, m, n, 2).reshape(b, m * n, 2)
        # Initializes corner references mostly near the room reference for stability.
        ref = (0.65 * room_ref_flat + 0.35 * init_ref).clamp(0.0, 1.0)

        # Stores intermediate decoder outputs for auxiliary training losses.
        aux: list[dict] = []
        # Runs each corner decoder layer and saves its intermediate prediction.
        for li, layer in enumerate(self.corner_decoder):
            # Updates corner tokens using scene features sampled around current references.
            tokens = layer(tokens, multi_feats, ref)
            # Predicts coordinate offsets for iterative corner refinement.
            delta = self.corner_delta_heads[li](tokens)
            # Applies coordinate refinement in logit space and maps back to [0, 1].
            ref = torch.sigmoid(inverse_sigmoid(ref) + delta)
            # Reshapes refined references into room-by-corner coordinate tensors.
            xy = ref.view(b, m, n, 2)
            # Reshapes corner tokens back into B x rooms x corners x C.
            ctok = tokens.view(b, m, n, c)
            # Stores this decoder layer prediction for auxiliary supervision.
            aux.append({
                # Stores predicted polygon corner coordinates.
                "corner_xy": xy,
                # Stores logits indicating which corner slots are valid.
                "corner_valid_logits": self.corner_valid_head(ctok).squeeze(-1),
                # Stores logits indicating which room queries are real rooms.
                "room_valid_logits": self.room_valid_head(room_tokens).squeeze(-1),
                # Stores room type/class logits.
                "room_class_logits": self.class_head(room_tokens),
            })
        # Returns final corner tokens, final corner coordinates and auxiliary decoder outputs.
        return tokens.view(b, m, n, c), ref.view(b, m, n, 2), aux

    # Improvement v9: resamples any predicted dense boundary sequence to 64 points for boundary_64 supervision
    def _boundary_to_64(self, boundary_xy: torch.Tensor) -> torch.Tensor:
        # Reads batch size, number of rooms, sequence length and xy dimension.
        b, m, p, d = boundary_xy.shape
        # Returns the sequence unchanged when it already has 64 points.
        if p == 64:
            return boundary_xy
        # Uses every second point when the native prediction is boundary_128.
        if p == 128:
            return boundary_xy[:, :, ::2, :]
        # Flattens batch and room dimensions so PyTorch can interpolate the boundary sequence.
        seq = boundary_xy.reshape(b * m, p, d).transpose(1, 2)
        # Linearly resamples the point sequence to 64 points for compatibility with boundary_64 targets.
        seq64 = F.interpolate(seq, size=64, mode="linear", align_corners=False)
        # Restores the tensor to B x rooms x 64 x 2 format.
        return seq64.transpose(1, 2).reshape(b, m, 64, d)

    # Improvement v9: decodes dense room boundary sequences for each room query
    def _decode_boundary(self, room_tokens: torch.Tensor, room_ref: torch.Tensor, multi_feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        # Reads batch size, number of room queries and feature dimension.
        b, m, c = room_tokens.shape
        # Stores the number of dense boundary point slots per room.
        n = self.max_boundary_points
        # Reshapes learnable boundary queries so they can be added to every room token.
        boundary_base = self.boundary_queries.weight.view(1, 1, n, c)
        # Combines each room token with each boundary query to create room-specific boundary tokens.
        tokens = room_tokens.unsqueeze(2) + boundary_base
        # Flattens room-boundary tokens into one sequence for the decoder.
        tokens = tokens.reshape(b, m * n, c)
        # Predicts initial normalized boundary reference coordinates.
        init_ref = torch.sigmoid(self.boundary_init_ref(tokens))
        # Repeats each room reference point for all dense boundary slots in that room.
        room_ref_flat = room_ref.unsqueeze(2).expand(b, m, n, 2).reshape(b, m * n, 2)
        # Initializes boundary references mostly near the room reference for stability.
        ref = (0.65 * room_ref_flat + 0.35 * init_ref).clamp(0.0, 1.0)

        # Improvement v9: stores intermediate dense boundary decoder outputs for auxiliary training losses
        aux: list[dict] = []
        # Runs each boundary decoder layer and saves its intermediate prediction.
        for li, layer in enumerate(self.boundary_decoder):
            # Updates boundary tokens using scene features sampled around current references.
            tokens = layer(tokens, multi_feats, ref)
            # Predicts coordinate offsets for iterative dense boundary refinement.
            delta = self.boundary_delta_heads[li](tokens)
            # Applies coordinate refinement in logit space and maps back to [0, 1].
            ref = torch.sigmoid(inverse_sigmoid(ref) + delta)
            # Reshapes refined references into room-by-boundary-point coordinate tensors.
            boundary128_xy = ref.view(b, m, n, 2)
            # Reshapes boundary tokens back into B x rooms x boundary points x C.
            btok = tokens.view(b, m, n, c)
            # Improvement v9: stores this decoder layer prediction for dense boundary auxiliary supervision
            aux.append({
                # Improvement v9: stores predicted 128-point dense boundary coordinates
                "boundary128_xy": boundary128_xy,
                # Improvement v9: stores predicted 64-point dense boundary coordinates derived from the dense sequence
                "boundary64_xy": self._boundary_to_64(boundary128_xy),
                # Improvement v9: stores logits indicating which dense boundary point slots are valid
                "boundary_point_valid_logits": self.boundary_valid_head(btok).squeeze(-1),
                # Stores logits indicating which room queries are real rooms.
                "room_valid_logits": self.room_valid_head(room_tokens).squeeze(-1),
                # Stores room type/class logits.
                "room_class_logits": self.class_head(room_tokens),
            })
        # Reshapes final dense boundary tokens back into B x rooms x boundary points x C.
        boundary_tokens = tokens.view(b, m, n, c)
        # Reshapes final dense boundary references back into B x rooms x boundary points x 2.
        boundary128_xy = ref.view(b, m, n, 2)
        # Improvement v9: creates a 64-point boundary prediction for direct boundary_64 loss
        boundary64_xy = self._boundary_to_64(boundary128_xy)
        # Returns final boundary tokens, 128-point boundary, 64-point boundary and auxiliary outputs.
        return boundary_tokens, boundary128_xy, boundary64_xy, aux

    # Runs the training-only denoising branch that refines noisy ground-truth polygons.
    def _decode_denoising(self, dn_polygon_xy: torch.Tensor, dn_corner_valid: torch.Tensor, dn_room_valid: torch.Tensor, multi_feats: list[torch.Tensor]) -> dict:
        # Reads the denoising polygon tensor shape.
        b, m, n, _ = dn_polygon_xy.shape
        # Stores the tensor device so newly created tensors are placed correctly.
        device = dn_polygon_xy.device
        # Creates room ID indices for every denoising room-corner token.
        room_ids = torch.arange(m, device=device).view(1, m, 1).expand(b, m, n)
        # Converts denoising corner-valid labels to valid embedding indices.
        valid_ids = dn_corner_valid.long().clamp(0, 1)
        # Builds denoising tokens from noisy coordinates, validity labels and room IDs.
        tokens = self.dn_coord_embed(dn_polygon_xy) + self.dn_valid_embed(valid_ids) + self.dn_room_embed(room_ids)
        # Flattens room-corner tokens into one sequence for the decoder.
        tokens = tokens.reshape(b, m * n, self.embed_dim)
        # Uses noisy polygon coordinates as initial denoising reference points.
        ref = dn_polygon_xy.reshape(b, m * n, 2).clamp(0.0, 1.0)
        # Stores intermediate decoder outputs for auxiliary training losses.
        aux: list[dict] = []
        # Runs each corner decoder layer and saves its intermediate prediction.
        for li, layer in enumerate(self.corner_decoder):
            # Updates corner tokens using scene features sampled around current references.
            tokens = layer(tokens, multi_feats, ref)
            # Predicts denoising coordinate correction offsets.
            delta = self.dn_delta_heads[li](tokens)
            # Applies coordinate refinement in logit space and maps back to [0, 1].
            ref = torch.sigmoid(inverse_sigmoid(ref) + delta)
            # Reshapes corner tokens back into B x rooms x corners x C.
            ctok = tokens.view(b, m, n, self.embed_dim)
            # Reshapes refined references into room-by-corner coordinate tensors.
            xy = ref.view(b, m, n, 2)
            # Averages valid corner tokens to form a room-level token for denoising heads.
            room_tok = (ctok * dn_corner_valid.unsqueeze(-1).float()).sum(dim=2) / dn_corner_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
            # Stores this decoder layer prediction for auxiliary supervision.
            aux.append({
                # Stores predicted polygon corner coordinates.
                "corner_xy": xy,
                # Stores logits indicating which corner slots are valid.
                "corner_valid_logits": self.dn_corner_valid_head(ctok).squeeze(-1),
                # Stores logits indicating which room queries are real rooms.
                "room_valid_logits": self.dn_room_valid_head(room_tok).squeeze(-1),
                # Stores room type/class logits.
                "room_class_logits": self.dn_class_head(room_tok),
            })
        # Returns the final denoising prediction plus earlier auxiliary denoising outputs.
        return aux[-1] | {"aux_outputs": aux[:-1]}

    # Runs the full model forward pass.
    def forward(
        self,
        x: torch.Tensor,
        dn_polygon_xy: Optional[torch.Tensor] = None,
        dn_corner_valid: Optional[torch.Tensor] = None,
        dn_room_valid: Optional[torch.Tensor] = None,
    ) -> dict:
        # Reads batch size and original BEV height/width.
        b, _, h, w = x.shape
        # Extracts multi-scale BEV features using the ResNet/FPN encoder.
        feats = self.encoder(x)
        # Builds the multi-scale feature list used by deformable attention.
        multi_feats = self._encode_memory(feats)
        # Decodes room queries into room-level features and reference points.
        room_tokens, room_ref = self._decode_rooms(multi_feats, b)
        # Decodes room-specific corner tokens into sparse polygon corner coordinates.
        corner_tokens, corner_xy, aux_outputs = self._decode_corners(room_tokens, room_ref, multi_feats)
        # Improvement v9: decodes room-specific dense boundary sequences for boundary_64 and boundary_128 supervision
        boundary_tokens, boundary128_xy, boundary64_xy, boundary_aux_outputs = self._decode_boundary(room_tokens, room_ref, multi_feats)

        # Builds the high-resolution dense mask feature map.
        mask_feat = self.mask_decoder(feats, target_size=(h, w))
        # Converts each room token into a room-specific mask embedding.
        mask_embed = self.room_mask_embed(room_tokens)  # B,M,mask_dim
        # Computes per-room masks by dot-producting room embeddings with dense mask features.
        room_mask_logits = torch.einsum("bmc,bchw->bmhw", mask_embed, mask_feat)

        # Creates the final output dictionary consumed by the training and inference code.
        out = {
            # Stores logits indicating which room queries are real rooms.
            "room_valid_logits": self.room_valid_head(room_tokens).squeeze(-1),
            # Stores room type/class logits.
            "room_class_logits": self.class_head(room_tokens),
            # Stores logits indicating which corner slots are valid.
            "corner_valid_logits": self.corner_valid_head(corner_tokens).squeeze(-1),
            # Stores predicted polygon corner coordinates.
            "corner_xy": corner_xy,
            # Improvement v9: stores predicted 128-point dense room boundary coordinates
            "boundary128_xy": boundary128_xy,
            # Improvement v9: stores predicted 64-point dense room boundary coordinates
            "boundary64_xy": boundary64_xy,
            # Improvement v9: stores logits indicating which dense boundary point slots are valid
            "boundary_point_valid_logits": self.boundary_valid_head(boundary_tokens).squeeze(-1),
            # Stores refined room reference points for debugging or inference support.
            "room_reference_xy": room_ref,
            # Stores dense wall mask logits.
            "wall_logits": self.wall_head(mask_feat),
            # Stores dense opening mask logits.
            "opening_logits": self.opening_head(mask_feat),
            # Stores dense boundary mask logits.
            "boundary_logits": self.boundary_head(mask_feat),
            # Stores dense corner heatmap logits.
            "corner_heatmap_logits": self.corner_heatmap_head(mask_feat),
            # Improvement v9: stores dense distance-to-wall logits for distance_to_wall_target supervision
            "distance_logits": self.distance_head(mask_feat),
            # Stores query-conditioned per-room mask logits.
            "room_mask_logits": room_mask_logits,
            # Stores auxiliary decoder predictions from earlier corner refinement layers.
            "aux_outputs": aux_outputs[:-1],
            # Improvement v9: stores auxiliary dense boundary decoder predictions from earlier boundary refinement layers
            "boundary_aux_outputs": boundary_aux_outputs[:-1],
        }
        # Adds optional furniture mask prediction when that head exists.
        if self.furniture_head is not None:
            # Stores optional dense furniture mask logits.
            out["furniture_logits"] = self.furniture_head(mask_feat)

        # Runs the denoising branch only when denoising inputs are provided during training.
        if dn_polygon_xy is not None and dn_corner_valid is not None and dn_room_valid is not None:
            # Stores denoising branch predictions in the output dictionary.
            out["dn_outputs"] = self._decode_denoising(dn_polygon_xy, dn_corner_valid, dn_room_valid, multi_feats)

        # Returns the final output dictionary from this module.
        return out