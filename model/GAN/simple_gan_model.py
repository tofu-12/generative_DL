import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from module_blocks.CNN import ConvBlock, ConvTransposeBlock


class Discriminator(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        """
        SimpleGANの判別器のインスタンスの初期化
        32x32の白黒imageに対応
        """
        super().__init__()

        self.dropout = nn.Dropout(p=drop_rate)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.conv_block_1 = ConvBlock(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            activation_function="leaky_relu"
        )
        self.conv_block_2 = ConvBlock(
            in_channels=128,
            out_channels=128, 
            kernel_size=3, 
            stride=2,
            padding=1, 
            activation_function="leaky_relu"
        )
        self.conv_block_3 = ConvBlock(
            in_channels=128,
            out_channels=256, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            activation_function="leaky_relu"
        )

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(256 * 1 * 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理

        Args:
            x: 入力画像テンソル
        
        Returns:
            torch.Tensor: 分類結果 (sigmoidなし)
        """
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.dropout(x)

        x = self.conv_block_1(x)
        x = self.dropout(x)

        x = self.conv_block_2(x)
        x = self.dropout(x)

        x = self.conv_block_3(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = x.view(-1, 256 * 1 * 1)
        x = self.fc(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        """
        SimpleGANの生成器のインスタンスの初期化
        32x32の白黒imageに対応

        Args:
            latent_dim: 潜在空間の次元
        """
        super().__init__()
        
        self.latent_dim = latent_dim

        self.conv_transpose_block_1 = ConvTransposeBlock(
            in_channels=latent_dim,
            out_channels=256, 
            kernel_size=3, 
            stride=2,
            padding=1,
            output_padding=1,
            activation_function="leaky_relu"
        )
        self.conv_transpose_block_2 = ConvTransposeBlock(
            in_channels=256,
            out_channels=128, 
            kernel_size=3, 
            stride=2,
            padding=1,
            output_padding=1,
            activation_function="leaky_relu"
        )
        self.conv_transpose_block_3 = ConvTransposeBlock(
            in_channels=128,
            out_channels=128, 
            kernel_size=3, 
            stride=2,
            padding=1,
            output_padding=1,
            activation_function="leaky_relu"
        )
        self.conv_transpose_block_4 = ConvTransposeBlock(
            in_channels=128,
            out_channels=64, 
            kernel_size=3, 
            stride=2,
            padding=1,
            output_padding=1,
            activation_function="leaky_relu"
        )

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1, 
            kernel_size=3, 
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理

        Args:
            x: 入力画像テンソル
        
        Returns:
            torch.Tensor: 生成画像
        """
        x = x.view(-1, self.latent_dim, 1, 1)
        
        x = self.conv_transpose_block_1(x)
        x = self.conv_transpose_block_2(x)
        x = self.conv_transpose_block_3(x)
        x = self.conv_transpose_block_4(x)

        x = self.conv_transpose(x)

        return x


class SimpleGAN(nn.Module):
    def __init__(self, discriminator_drop_rate: float=0.3, latent_dim: int=100):
        """
        SimpleGANのインスタンスの初期化
        32x32の白黒imageに対応

        Args:
            discriminator_drop_rate: 識別器のDropoutのprob
            latent_dim: 潜在空間の次元
        """
        super().__init__()

        self.discriminator = Discriminator(discriminator_drop_rate)
        self.generator = Generator(latent_dim)
