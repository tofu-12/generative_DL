from pydantic import BaseModel, ConfigDict
import torch


# データ関連
class Dataloaders(BaseModel):
    """
    データローダをまとめるデータ型

    Args:
        train: 訓練用データローダ
        val: 検証用データローダ
        test: テスト用データローダ
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


# history関連
class History(BaseModel):
    """
    history

    Args:
        train_loss_per_batch: バッチごとの訓練の損失
        train_loss_per_epoch: エポックごとの訓練の損失
        val_loss_per_epoch: 検証の損失
        test_loss: テストの損失
    """
    train_loss_per_batch: list = []
    train_loss_per_epoch: list = []
    val_loss_per_epoch: list = []
    test_loss: list = []

class VAEHistory(History):
    """
    VAEの記録用history

    Args:
        train_loss_per_batch: バッチごとの訓練の損失
        train_loss_per_epoch: エポックごとの訓練の損失
        val_loss_per_epoch: 検証の損失
        test_loss: テストの損失
        val_z_per_epoch: 検証データの潜在空間上の座標
        val_z_label_per_epoch: 検証データのラベル
    """
    val_z_per_epoch: list = []
    val_z_label_per_epoch: list = []
