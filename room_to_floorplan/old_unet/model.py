from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FloorplanUNet(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base: int = 32):
        super().__init__()
        self.d1 = ConvBlock(in_channels, base)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = ConvBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)

        self.c3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)

        self.c2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)

        self.c1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))

        m = self.mid(self.p3(d3))

        x = self.u3(m)
        x = self.c3(torch.cat([x, d3], dim=1))

        x = self.u2(x)
        x = self.c2(torch.cat([x, d2], dim=1))

        x = self.u1(x)
        x = self.c1(torch.cat([x, d1], dim=1))

        return self.head(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class FloorplanUNetAttention(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base: int = 32):
        super().__init__()

        # This encodes
        self.d1 = ConvBlock(in_channels, base)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = ConvBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 4, base * 8)

        # This decodes
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.c3 = ConvBlock(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.c2 = ConvBlock(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.c1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        # This encodes
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        m = self.mid(self.p3(d3))

        # This decodes + Attention skip connections
        x = self.u3(m)
        d3_att = self.att3(x, d3)
        x = self.c3(torch.cat([x, d3_att], dim=1))

        x = self.u2(x)
        d2_att = self.att2(x, d2)
        x = self.c2(torch.cat([x, d2_att], dim=1))

        x = self.u1(x)
        d1_att = self.att1(x, d1)
        x = self.c1(torch.cat([x, d1_att], dim=1))

        return self.head(x)

class FloorplanUNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int=6, out_channels:int=2, base:int=32, deep_supervision: bool=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        b = base

        # Encoder
        self.d1 = ConvBlock(in_channels, b)
        self.d2 = ConvBlock(b, b * 2)
        self.d3 = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = ConvBlock(b * 4, b * 8)

        # up0: b*2  → b,    up1: b*4  → b*2,   up2: b*8 → b*4
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