#!/usr/bin/env python
"""Hong et al. (2021) 3-D U/V-Net architecture.

This is a literal PyTorch transcription of Sec. 3.1 and Fig. 1 of
Hong, Jeong, Hwang & Kim (2021, ApJ 913, 76; arXiv:2008.01738).
The published TNG100 model maps (2,64,64,64) to (1,64,64,64).

The paper used Keras/TensorFlow.  PyTorch's BatchNorm ``momentum`` has the
opposite convention from Keras, hence momentum=0.01 here corresponds to the
Keras default moving-average momentum of 0.99.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


PAPER_CHANNELS = (128, 256, 512, 1024, 2048)


class EncoderBlock(nn.Module):
    """reflect-pad(2) -> Conv3d(5,stride=2) -> ReLU -> BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=0
        )
        self.bn = nn.BatchNorm3d(out_channels, eps=1.0e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (2, 2, 2, 2, 2, 2), mode="reflect")
        return self.bn(F.relu(self.conv(x), inplace=False))


class DecoderBlock(nn.Module):
    """nearest upsample -> concatenate skip -> BN -> reflect Conv3d(3) -> ReLU."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        joined = in_channels + skip_channels
        self.bn = nn.BatchNorm3d(joined, eps=1.0e-3, momentum=0.01)
        self.conv = nn.Conv3d(joined, out_channels, kernel_size=3, padding=0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, skip), dim=1)
        x = self.bn(x)
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="reflect")
        return F.relu(self.conv(x), inplace=False)


class OutputBlock(nn.Module):
    """Published Output layer: upsample, input skip, BN, 3^3 conv, tanh."""

    def __init__(self, in_channels: int, input_channels: int = 2) -> None:
        super().__init__()
        joined = in_channels + input_channels
        self.bn = nn.BatchNorm3d(joined, eps=1.0e-3, momentum=0.01)
        self.conv = nn.Conv3d(joined, 1, kernel_size=3, padding=0)

    def forward(self, x: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, raw_input), dim=1)
        x = self.bn(x)
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="reflect")
        return torch.tanh(self.conv(x))


class Hong2021Net(nn.Module):
    """Five-level network used for both TNG100 and TNG300."""

    def __init__(
        self,
        in_channels: int = 2,
        channels: Sequence[int] = PAPER_CHANNELS,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("Hong2021Net requires exactly five encoder widths")
        c = tuple(int(v) for v in channels)
        self.in_channels = int(in_channels)
        self.channels = c
        enc_in = (in_channels,) + c[:-1]
        self.encoder = nn.ModuleList(
            EncoderBlock(ci, co) for ci, co in zip(enc_in, c, strict=True)
        )
        self.decoder = nn.ModuleList(
            (
                DecoderBlock(c[4], c[3], c[3]),
                DecoderBlock(c[3], c[2], c[2]),
                DecoderBlock(c[2], c[1], c[1]),
                DecoderBlock(c[1], c[0], c[0]),
            )
        )
        self.output = OutputBlock(c[0], in_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras Conv3D default: Glorot-uniform kernel and zero bias.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (batch,{self.in_channels},N,N,N), "
                f"received {tuple(x.shape)}"
            )
        if x.shape[-3:] not in ((64, 64, 64), (128, 128, 128)):
            raise ValueError("published models accept spatial size 64 or 128")
        raw = x
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1]), strict=True):
            x = block(x, skip)
        return self.output(x, raw)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64, choices=(64, 128))
    parser.add_argument(
        "--smoke-widths",
        default=None,
        help="comma-separated test widths; omit for paper widths",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    widths = (
        tuple(int(v) for v in args.smoke_widths.split(","))
        if args.smoke_widths
        else PAPER_CHANNELS
    )
    model = Hong2021Net(channels=widths).to(args.device)
    n_params = parameter_count(model)
    print(f"channels={widths} parameters={n_params:,} ({n_params/1e6:.3f} M)")
    with torch.inference_mode():
        x = torch.zeros((1, 2, args.grid, args.grid, args.grid), device=args.device)
        y = model(x)
    print(f"input={tuple(x.shape)} output={tuple(y.shape)} range=[{y.min():.3g},{y.max():.3g}]")


if __name__ == "__main__":
    main()
