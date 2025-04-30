import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from model import VAE
from dataset.CelebA_dataset import CelebADataset
from utils import get_device


def train_loop(dataloader: torch.utils.data.DataLoader, batch_size: int, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, device) -> None:
    """
    訓練ループ（1エポック）

    データローダーからバッチごとにデータを取得し、モデルの訓練を行います。
    損失を計算し、勾配を計算してモデルのパラメータを更新します。

    Args:
        dataloader: 訓練データを提供するPyTorchのDataLoaderオブジェクト
        batch_size: バッチサイズ
        model: 訓練対象のPyTorchモデル
        loss_fn: 損失関数
        optimizer: モデルのパラメータを更新するためのオプティマイザ
        device: 使用するデバイス
    """
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred, _, _ = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_item: float = loss.item()
            current: int = batch * batch_size + len(X)
            print(f"loss: {loss_item:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn: nn.Module, device) -> None:
    """
    評価ループ（1エポック）

    データローダーからバッチごとにデータを取得し、モデルの評価を行います。
    勾配計算なしで実行し、平均損失を計算・表示します。

    Args:
        dataloader (DataLoader): 評価データを提供するPyTorchのDataLoaderオブジェクト
        model (nn.Module): 評価対象のPyTorchモデル
        loss_fn (nn.Module): 損失関数
        device: 使用するデバイス
    """
    model.eval()
    num_batches: int = len(dataloader)
    test_loss: float = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred, _, _ = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # デバイスの決定
    device = get_device()

    # データローダの作成
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    img_dir_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "img_align_celeba")
    attr_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_attr_celeba.csv")
    partition_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_eval_partition.csv")

    train_set = CelebADataset(img_dir_path, attr_path, partition_path, "train", transform)
    test_set = CelebADataset(img_dir_path, attr_path, partition_path, "test", transform)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=2)

    # モデルの設定
    model = VAE().to(device)

    # 損失関数とオプティマイザの定義
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, batch_size, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")
