import enum
import math
import typing as tp

import torch
import torch.nn as nn


HiddenStateT = tp.Tuple[torch.Tensor, torch.Tensor]

class ResnetResampleModes(enum.Enum):
    IDENTITY = 'identity'
    UP = 'up'
    DOWN = 'down'


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()

        position = torch.arange(d_model // 3)
        div_term = torch.exp(torch.arange(0, d_model // 3, 2) * (-math.log(10000.0) / d_model))
        points_enc = torch.zeros(d_model // 3)
        points_enc[0::2] = torch.sin(position[0::2] * div_term)
        points_enc[1::2] = torch.cos(position[1::2] * div_term)
        pos_enc = points_enc.repeat_interleave(3)

        self.pos_enc: torch.Tensor
        self.register_buffer('pos_enc', pos_enc, persistent=False)

        self.requires_grad_(False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor + self.pos_enc
        return tensor


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.positional_embeddings = PositionalEncoding(d_model=42)
        self.lstm1 = nn.LSTM(
            input_size=42,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.0
        )
        self.lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=num_classes,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.hidden_states = [None] * 2

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        tensor = self.positional_embeddings(tensor)
        tensor, hidden_state1 = self.lstm1(tensor, self.hidden_states[0])
        tensor, hidden_state2 = self.lstm2(tensor, self.hidden_states[1])
        self.hidden_states = (hidden_state1, hidden_state2)
        return tensor


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 1,
        resample_factor: int = 2,
        mode: ResnetResampleModes = ResnetResampleModes.IDENTITY,
    ) -> None:
        super().__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        adapted_kernel_size = 2 * int(kernel_size * (stride + 1) / 2) - 1
        adapted_stride = stride ** 2
        adapted_dilation = dilation
        adapted_padding = padding * (stride + 1) - dilation * int(1 - stride / 2)
        self.identity_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=adapted_kernel_size,
            stride=adapted_stride,
            dilation=adapted_dilation,
            padding=adapted_padding,
            bias=False,
        )

        if mode == ResnetResampleModes.IDENTITY:
            self.identity_resample = nn.Identity()
        elif mode == ResnetResampleModes.UP:
            self.identity_resample = nn.Upsample(scale_factor=resample_factor, mode='nearest')
        elif mode == ResnetResampleModes.DOWN:
            self.identity_resample = nn.MaxPool2d(kernel_size=resample_factor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.bn1(tensor)
        out = self.activation(out)
        out = self.identity_resample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        identity = self.identity_resample(tensor)
        identity = self.identity_conv(identity)
        out = out + identity

        return out


class LinearHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_dim, 48, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(48, num_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.blocks(tensor)


class CNNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ResNetBlock(in_channels, 8, kernel_size=7, dilation=2, padding=6),
            ResNetBlock(8, 8, kernel_size=5, padding=2),
            ResNetBlock(8, 16, mode=ResnetResampleModes.DOWN, kernel_size=5, padding=2),
            ResNetBlock(16, 16, kernel_size=5, padding=2),
            ResNetBlock(16, 16, kernel_size=5, padding=2),
            ResNetBlock(16, 32, mode=ResnetResampleModes.DOWN, kernel_size=5, padding=2),
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 64, mode=ResnetResampleModes.DOWN),
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
            ResNetBlock(64, out_channels, mode=ResnetResampleModes.DOWN),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.blocks(tensor)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        image_size: tp.Tuple[int, int],
        frames: int,
        batch_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.register_buffer(
            'queue_rgb',
            torch.zeros((batch_size, 3 * frames, *image_size)),
            persistent=False,
        )
        self.register_buffer(
            'queue_depth',
            torch.zeros((batch_size, 1 * frames, *image_size)),
            persistent=False,
        )

        self.cnn_model_rgb = CNNModel(
            in_channels=3*frames,
            out_channels=64,
        )
        self.cnn_model_depth = CNNModel(
            in_channels=frames,
            out_channels=64,
        )

        self.head = LinearHead(
            # in_dim=2*64*image_size[0]*image_size[1] // (4**4),
            in_dim=4096,
            num_classes=num_classes,
        )

    def _push_to_tensor_rgb(
        self,
        rgb: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((self.queue_rgb[:, 3:], rgb), dim=1)

    def _push_to_tensor_depth(
        self,
        depth: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((self.queue_depth[:, 1:], depth), dim=1)

    def forward(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        rgb = image[:, :3]
        depth = image[:, 3:]

        self.queue_rgb = self._push_to_tensor_rgb(rgb)
        self.queue_depth = self._push_to_tensor_depth(depth)

        rgb_features = self.cnn_model_rgb(self.queue_rgb)
        depth_features = self.cnn_model_depth(self.queue_depth)

        return self.head(torch.cat((rgb_features, depth_features), dim=1))
