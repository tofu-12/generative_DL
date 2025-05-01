import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int, padding: int):
        """
        畳み込みブロックのインスタンスの初期化
        Conv -> BatchNorm -> Relu

        Args:
            in_channels: 入力チャネル数
            out_channels: 出力チャネル数
            kernel_size: カーネルサイズ
            stride: ストライドの幅
            padding: パディングサイズ
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理

        Args:
            x: 入力テンソル
        
        Returns:
            torch.Tensor: 出力データ
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int, padding: int, output_padding: int):
        """
        デコーダー用の転置畳み込みブロックのインスタンスの初期化
        ConvTranspose -> BatchNorm -> Relu

        Args:
            in_channels: 入力チャネル数
            out_channels: 出力チャネル数
            kernel_size: カーネルサイズ
            stride: ストライドの幅
            padding: パディングサイズ
            output_padding: 出力の追加パディングサイズ
        """
        super(DecoderConvBlock, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理
        入力テンソルxに対して、ConvTranspose -> BatchNorm -> Reluの順で適用

        Args:
            x: 入力テンソル

        Returns:
            torch.Tensor: 処理後の出力テンソル
        """
        x = self.conv_t(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Reparameterize(nn.Module):
    def __init__(self):
        """
        再パラメータ化トリックを実装するクラスのインスタンスの初期化
        """
        super(Reparameterize, self).__init__()

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        再パラメータ化トリックにより、潜在変数zをサンプリング

        Args:
            z_mean: 潜在空間の平均を表すテンソル
            z_log_var: 潜在空間の対数分散を表すテンソル

        Returns:
            torch.Tensor: サンプリングされた潜在変数zのテンソル
        """
        std_dev = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(z_mean)
        z = z_mean + std_dev * epsilon
        
        return z    


class VAEEncoder(nn.Module):
    def __init__(self):
        """
        Encoderクラスのインスタンスの初期化
        Variational Autoencoder (VAE) のエンコーダー部分を構成する。
        入力画像を処理し、潜在空間の平均(z_mean)と対数分散(z_log_var)を出力する。
        また、再パラメータ化トリックを用いて潜在変数zをサンプリングする。
        """
        super(VAEEncoder, self).__init__()
        self.conv_block_1 = ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_2 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_3 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_block_4 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.fc_z_mean = nn.Linear(512, 200)
        self.fc_z_log_var = nn.Linear(512, 200)
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
        x = x.view(-1, 128 * 2 * 2)
        z_mean = self.fc_z_mean(x)
        z_log_var = self.fc_z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)

        return z, z_mean, z_log_var


class VAEDecoder(nn.Module):
    def __init__(self):
        """
        VAEDecoderクラスのインスタンスの初期化
        Variational Autoencoder (VAE) のデコーダー部分を構成する。
        潜在変数zを入力として受け取り、画像を再構成する。
        エンコーダーと逆の処理を行うように設計されている。
        """
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(200, 128 * 2 * 2) 
        self.decoder_conv_block_1 = DecoderConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv_block_2 = DecoderConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv_block_3 = DecoderConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
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
        x = self.final_conv_t(x)
        x = F.sigmoid(x)

        return x

class VAE(nn.Module):
    def __init__(self):
        """
        VAEクラスのインスタンスの初期化
        Variational Autoencoder (VAE) モデル全体を構成する。
        エンコーダーとデコーダーを内部に持つ。
        """
        super(VAE, self).__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

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


def loss_function(predict, target, z_mean, z_log_var):
    """
    VAEの損失関数
    再構成誤差とKL情報量の和

    Args:
        predict: 予測値
        target: 真値
        z_mean: 潜在空間の平均値
        z_log_var: 潜在空間の対数分散
    
    Returns:
        損失
    """
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp())
    loss = bce_loss + kl_loss
    return loss
