import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.join(__file__), os.pardir))
from module_blocks.CNN import ConvBlock, ConvTransposeBlock
from module_blocks.VAE import Reparameterize


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        """
        Encoderクラスのインスタンスの初期化
        Variational Autoencoder (VAE) のエンコーダー部分を構成する。
        入力画像を処理し、潜在空間の平均(z_mean)と対数分散(z_log_var)を出力する。
        また、再パラメータ化トリックを用いて潜在変数zをサンプリングする。
        """
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_2 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_3 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_4 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.fc_z_mean = nn.Linear(512, latent_dim)
        self.fc_z_log_var = nn.Linear(512, latent_dim)
        self.reparameterize = Reparameterize()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播の処理

        Args:
            x: 入力画像テンソル

        Returns:
            tuple:
                z: サンプリングされた潜在変数zのテンソル
                z_mean: 潜在空間の平均z_meanのテンソル
                z_log_var: 潜在空間の対数分散z_log_varのテンソル
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = x.view(-1, 128 * 2 * 2)
        z_mean = self.fc_z_mean(x)
        z_log_var = self.fc_z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)

        return z, z_mean, z_log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        """
        VAEDecoderクラスのインスタンスの初期化
        Variational Autoencoder (VAE) のデコーダー部分を構成する。
        潜在変数zを入力として受け取り、画像を再構成する。
        エンコーダーと逆の処理を行うように設計されている。
        """
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 2 * 2) 
        self.decoder_conv_block_1 = ConvTransposeBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv_block_2 = ConvTransposeBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv_block_3 = ConvTransposeBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv_block_4 = ConvTransposeBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv_t = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理
        潜在変数zを入力として、画像を再構成する。

        Args:
            z: 潜在変数zのテンソル

        Returns:
            torch.Tensor: 再構成された画像テンソル
        """
        x = self.fc(z)
        x = x.view(-1, 128, 2, 2) 
        x = self.decoder_conv_block_1(x)
        x = self.decoder_conv_block_2(x)
        x = self.decoder_conv_block_3(x)
        x = self.decoder_conv_block_4(x)
        x = self.final_conv_t(x)

        return x


class ColorVAE(nn.Module):
    def __init__(self, latent_dim: int=200):
        """
        VAEクラスのインスタンスの初期化
        Variational Autoencoder (VAE) モデル全体を構成する。
        エンコーダーとデコーダーを内部に持つ。
        """
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAEモデルの順伝播
        入力画像xをエンコード、潜在変数をサンプリング、デコードして再構成画像を生成する。

        Args:
            x: 入力画像テンソル

        Returns:
            tuple:
                output: 再構成された画像テンソル
                z: 潜在空間
                z_mean: エンコーダーが出力した潜在空間の平均
                z_log_var: エンコーダーが出力した潜在空間の対数分散
        """
        z, z_mean, z_log_var = self.encoder(x)
        output = self.decoder(z)
        
        return output ,z, z_mean, z_log_var
