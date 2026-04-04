import torch
from torch import nn
from torch.nn import functional as F
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_in, feature))
            current_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        current_in = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(current_in, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
            current_in = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _center_crop(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """
        Center crop a tensor to the target height and width.

        If the tensor is smaller than the target size, it will be padded with zeros before cropping
        to ensure the target size is met.
        """
        _, _, h, w = x.shape
        target_h, target_w = target_hw

        if h == target_h and w == target_w:
            return x

        if h < target_h or w < target_w:
            pad_h = max(target_h - h, 0)
            pad_w = max(target_w - w, 0)
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
            _, _, h, w = x.shape

        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return x[:, :, top : top + target_h, left : left + target_w]

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        Pad NCHW tensor so H and W are divisible by `multiple`.

        Returns the padded tensor and the padding tuple (pad_left, pad_right, pad_top, pad_bottom)
        suitable for slicing the padding away later.
        """
        if multiple <= 1:
            return x, (0, 0, 0, 0)

        _, _, h, w = x.shape
        target_h = int(math.ceil(h / multiple) * multiple)
        target_w = int(math.ceil(w / multiple) * multiple)

        pad_h = target_h - h
        pad_w = target_w - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return x, (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        # Ensure dimensions survive down/up-sampling without spatial drift.
        # With `len(self.downs)` pooling stages, we need H and W divisible by 2**depth.
        x, (pad_left, pad_right, pad_top, pad_bottom) = self._pad_to_multiple(x, multiple=2 ** len(self.downs))

        skip_connections: list[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections.reverse()

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]

            # If the skip connection and the upsampled tensor have different spatial dimensions, center crop the skip connection
            if skip.shape[2:] != x.shape[2:]:
                skip = self._center_crop(skip, (x.shape[2], x.shape[3]))

            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)

        out = self.final_conv(x)
        # Remove padding so output matches the original input spatial size.
        if pad_left or pad_right or pad_top or pad_bottom:
            out = out[:, :, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]
        return out
