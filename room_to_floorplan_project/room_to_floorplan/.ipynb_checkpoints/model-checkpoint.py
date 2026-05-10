from __future__ import annotations

import torch
import torch.nn as nn

# all inputs are (batch_size, 6, 512, 512)
# all outputs are (batch_size, 2, 512, 512)

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),  # input, output, kernel-size, padding to keep image size the same
            nn.BatchNorm2d(c_out),  # for stabilizing training
            nn.ReLU(inplace=True),  # add non-linearity (inplace to modify tensor directly and save memory

            # Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
            # learns wall edges, room boundaries, furniture regions and structural patterns
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # pass through sequential block
        return self.block(x)


class FloorplanUNet(nn.Module):
    # base: first hidden layer has 32 channels
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base: int = 32):
        super().__init__()
        # 1st encoder block - eg: input -> batch, 6, 512, 512 => output -> batch, 32, 512, 512
        self.d1 = ConvBlock(in_channels, base)
        self.p1 = nn.MaxPool2d(2)  # half spatial size => 512x512 -> 256x256

        # 2nd encoder block => 32 -> 64
        # shape becomes: batch, 64, 256, 256
        self.d2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)  # half again => 128x128

        # 3rd encoder block => 64 -> 128
        # shape becomes: batch, 128, 128, 128
        self.d3 = ConvBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)  # half again => 64x64

        # bottleneck, doubles size => 128 -> 256
        # shape becomes: batch, 256, 64, 64
        # deepest layer => captures most abstract floorplan structure like overall room layout, wall arrangement and large-scale geom
        self.mid = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)

        # 1st decoder to upsample feature map
        # 256 -> 128 and 64x64 -> 128x128
        # shape becomes: batch, 128, 128, 128
        self.c3 = ConvBlock(base * 8, base * 4)  # * 8 since after upsampling, devoder has 128 channels and d3 also has 128 channels. Concatenating = 128 + 128 = 256 and base * 8 = 256 -> ConvBlock redices it back to 128
        # upsamples 128 -> 64 and 128x128 -> 256x256
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)

        # after concat, d2 => 64 + 64 = 128, therefore base*4 -> base*2 => 128 -> 64
        self.c2 = ConvBlock(base * 4, base * 2)
        # upsamples 64 -> 32 and 256x256 -> 512x512
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)

        # after concat, d1: 32+32=64, then reduced to 32
        self.c1 = ConvBlock(base * 2, base)
        # final output layer => 32 -> 2
        # Kernel size is 1, meaning it does not look at neighbouring pixels anymore. It only converts each pixel’s 32 learned features into 2 class scores.
        # final output -> batch, 2, 512, 512
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.d1(x)  # first feature map, high res/low depth (batch, 32, 512, 512)
        d2 = self.d2(self.p1(d1))  # pool first then convblock (b, 64, 256, 256)
        d3 = self.d3(self.p2(d2))  # (b, 128, 128, 128)

        m = self.mid(self.p3(d3))  # (b, 256, 64, 64)

        x = self.u3(m)  # upsample (b, 128, 128, 128)
        x = self.c3(torch.cat([x, d3], dim=1))  # concatenate channel dims => (b, 128, 128, 128) + (b, 128, 128, 128) = (b, 256, 128, 128) convert to (b, 128, 128, 128)

        # repeat 
        x = self.u2(x)
        x = self.c2(torch.cat([x, d2], dim=1))

        x = self.u1(x)
        x = self.c1(torch.cat([x, d1], dim=1))

        return self.head(x)  # returns logits


class AttentionGate(nn.Module):
    # filter skip connections before sending into decoder
    # normal UNet -> decoder + encoder skip
    # in attention -> decoder + filtered encoder skip
    def __init__(self, F_g, F_l, F_int):  # num of channels in decoder gating signal, num of channels in encoder skip conn, intermediate compressed channel size
        super().__init__()
        # transform decoder feature g -> what we try to reconstruct
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        # transform encoder skip feature x: contains detailed spatial info
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # produces single-channel attention mask. sigmoid converts value into 0 -> 1
        # values near 0 -> suppress, else keep
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # non-linearity
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):  # decoder, encoder skip
        # project into same channel size
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # combine decoder context and encoder detail
        psi = self.relu(g1 + x1)
        # create attention map (b, 1, H, W)
        psi = self.psi(psi)

        # multiply encoder skip by attention mask. Psi has 1 channel that is broadcasted by PyTorch across all feature maps so skip conn is filtered spatially => helps focus on walls, boundaries and structures; ignores noise, irrelevant clutter and weak artifacts
        return x * psi


class FloorplanUNetAttention(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base: int = 32):
        super().__init__()

        # Encoder
        # Level 1: 6 -> 32; 512x512
        self.d1 = ConvBlock(in_channels, base)
        self.p1 = nn.MaxPool2d(2)

        # Level 2: 32 -> 64; 256x256
        self.d2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)

        # Level 3: 64 -> 128; 128x128
        self.d3 = ConvBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)

        # bottleneck: 128 -> 256; 64x64
        self.mid = ConvBlock(base * 4, base * 8)

        # Decoder with attention
        # 256 -> 128; 64x64 -> 128x128
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        # f_g: 128; f_l: 128, f_int: 64; filters d3 before concat
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        # after concat: 128 decoder + 128 attended skip = 256, then 256 -> 128
        self.c3 = ConvBlock(base * 8, base * 4)

        # repeat => 64 decoder + 64 attended skip = 128, then 128 -> 64
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.c2 = ConvBlock(base * 4, base * 2)

        # repeat => 32 decoder + 32 attended skip = 64, then 64 -> 32
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.c1 = ConvBlock(base * 2, base)

        # final pred => 32 -> 2
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        # same as UNet encoder
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        m = self.mid(self.p3(d3))

        # decoder + attention skip
        # normal UNet: torch.cat([x, d3], dim=1); Attention UNet: d3_att = self.att3(x, d3) torch.cat([x, d3_att], dim=1) -> So the skip feature is filtered before being used.
        x = self.u3(m)
        d3_att = self.att3(x, d3)
        x = self.c3(torch.cat([x, d3_att], dim=1))

        # repeat
        x = self.u2(x)
        d2_att = self.att2(x, d2)
        x = self.c2(torch.cat([x, d2_att], dim=1))

        # repeat
        x = self.u1(x)
        d1_att = self.att1(x, d1)
        x = self.c1(torch.cat([x, d1_att], dim=1))

        return self.head(x)


class FloorplanUNetPlusPlus(nn.Module):
    # uses nested dense skip conns. normal UNet has 1 skip per level, unet++ has multiple intermediate feature nodes between encoder and decoder. Goal: reduce semantic gap between encoder and decoder features
    def __init__(self, in_channels: int=6, out_channels:int=2, base:int=32, deep_supervision: bool=False):  # if deep_supervision is False, model outputs from final node, else, makes preds from multiple decoder nodes and averages
        super().__init__()

        # stores if d_s is enabled
        self.deep_supervision = deep_supervision
        b = base

        # Encoder
        self.d1 = ConvBlock(in_channels, b)
        self.d2 = ConvBlock(b, b * 2)
        self.d3 = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = ConvBlock(b * 4, b * 8)

        # up0: b*2 -> b, up1: b*4 -> b*2, up2: b*8 -> b*4
        self.up0 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.up2 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)

        # Column 1 nodes
        self.c01 = ConvBlock(b * 2,  b)
        self.c11 = ConvBlock(b * 4,  b * 2)
        self.c21 = ConvBlock(b * 8,  b * 4)

        # Column 2 nodes
        self.c02 = ConvBlock(b * 3,  b)
        self.c12 = ConvBlock(b * 6,  b * 2)

        # Column 3 nodes
        self.c03 = ConvBlock(b * 4,  b)

        # Output head(s)
        if deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv2d(b, out_channels, 1) for _ in range(3)
            ])
        else:
            self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x00 = self.d1(x)
        x10 = self.d2(self.pool(x00))
        x20 = self.d3(self.pool(x10))
        x30 = self.mid(self.pool(x20))

        # Column 1
        x21 = self.c21(torch.cat([self.up2(x30), x20], dim=1))
        x11 = self.c11(torch.cat([self.up1(x20), x10], dim=1))
        x01 = self.c01(torch.cat([self.up0(x10), x00], dim=1))

        # Column 2
        x12 = self.c12(torch.cat([self.up1(x21), x10, x11], dim=1))
        x02 = self.c02(torch.cat([self.up0(x11), x00, x01], dim=1))

        # Column 3
        x03 = self.c03(torch.cat([self.up0(x12), x00, x01, x02], dim=1))

        # Output
        if self.deep_supervision:
            preds = [
                self.heads[0](x01),
                self.heads[1](x02),
                self.heads[2](x03)
            ]
            return torch.mean(torch.stack(preds, dim=0), dim=0)

        return self.head(x03)