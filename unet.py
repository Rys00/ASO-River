import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Identity() if dropout_p == 0.0 else nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.dropout(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        dropout_p: float = 0.1,
        dropout_min_features: int = 256,
        dropout_bottleneck_p: float | None = 0.2,
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.dropout_min_features = dropout_min_features
        self.dropout_bottleneck_p = dropout_bottleneck_p

        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")
        if dropout_bottleneck_p is not None and not (0.0 <= dropout_bottleneck_p < 1.0):
            raise ValueError(f"dropout_bottleneck_p must be in [0.0, 1.0), got {dropout_bottleneck_p}")
        if dropout_min_features < 0:
            raise ValueError(f"dropout_min_features must be >= 0, got {dropout_min_features}")

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.required_mod = 2 ** len(features)

        current_in = in_channels
        for feature in features:
            p = dropout_p if feature >= dropout_min_features else 0.0
            self.downs.append(DoubleConv(current_in, feature, dropout_p=p))
            current_in = feature

        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2,
            dropout_p=dropout_p if dropout_bottleneck_p is None else dropout_bottleneck_p,
        )

        current_in = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(current_in, feature, kernel_size=2, stride=2))
            p = dropout_p if feature >= dropout_min_features else 0.0
            self.ups.append(DoubleConv(feature * 2, feature, dropout_p=p))
            current_in = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] % self.required_mod != 0 or x.shape[3] % self.required_mod != 0:
            raise ValueError(f"Input height and width must be divisible by {self.required_mod} (2**{len(self.downs)}) to ensure proper downsampling and upsampling of the network.")
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
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)
