from __future__ import annotations

import torch
import torch.nn as nn


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
