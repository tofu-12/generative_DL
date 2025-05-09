import torch
import torch.nn as nn
import torch.nn.functional as F


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
