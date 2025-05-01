import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from model import VAE, loss_function
from dataset.CelebA_dataset import CelebADataset
from utils import get_device
from schemas import VAEHistory


def train_loop(history: VAEHistory, dataloader: torch.utils.data.DataLoader, batch_size: int, model: nn.Module, optimizer: optim.Optimizer, device) -> None:
    """
    訓練ループ（1エポック）

    データローダーからバッチごとにデータを取得し、モデルの訓練を行います。
    損失を計算し、勾配を計算してモデルのパラメータを更新します。

    Args:
        history: 記録の保存
        dataloader: 訓練データを提供するPyTorchのDataLoaderオブジェクト
        batch_size: バッチサイズ
        model: 訓練対象のPyTorchモデル
        optimizer: モデルのパラメータを更新するためのオプティマイザ
        device: 使用するデバイス
    """
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)

        pred, z, z_mean, z_log_var = model(X)

        history.z.append(z)
        history.z_mean.append(z_mean)
        history.z_log_var.append(z_log_var)
        history.label.append(y)

        loss = loss_function(pred, X, z_mean, z_log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {(loss/len(X)):>7f}  [{current:>5d}/{size:>5d}]")
        
        history.train_loss.append(loss/len(X))


def test_loop(history: VAEHistory, dataloader: torch.utils.data.DataLoader, model: nn.Module, device) -> None:
    """
    評価ループ（1エポック）

    データローダーからバッチごとにデータを取得し、モデルの評価を行います。
    勾配計算なしで実行し、平均損失を計算・表示します。

    Args:
        history: 記録の保存
        dataloader: 評価データを提供するPyTorchのDataLoaderオブジェクト
        model: 評価対象のPyTorchモデル
        device: 使用するデバイス
    """
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)

            pred, z, z_mean, z_log_var = model(X)
            test_loss += loss_function(pred, X, z_mean, z_log_var).item()

    test_loss /= size
    history.test_loss.append(test_loss)
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    try:
        # デバイスの決定
        device = get_device()
    
    except Exception as e:
        print(f"デバイスの決定に失敗しました: {str(e)}")
        sys.exit(1)

    try:
        # データローダの作成
        batch_size = 100

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
    
    except Exception as e:
        print(f"データローダの作成に失敗しました: {str(e)}")
        sys.exit(1)

    try:
        # モデルの設定
        model = VAE().to(device)

        # オプティマイザの定義
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 重みのパス
        weight_file_path = os.path.join(os.path.dirname(__file__), "model_weight.pth")

        # 重みの読み込み
        try:
            if os.path.exists(weight_file_path):
                print("loading weight...")
                model.load_state_dict(torch.load(weight_file_path))
        except Exception as e:
            print(f"重みの読み込みに失敗しました: {str(e)}")

        # 学習ループ
        epochs = 10
        history = VAEHistory()

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(history, train_dataloader, batch_size, model, optimizer, device)
            test_loop(history, test_dataloader, model, device)

        print("Done!")

        print("Saving Weight")
        torch.save(model.state_dict(), weight_file_path)
        print("Done!")
    
    except KeyboardInterrupt:
        print("Saving Weight")
        torch.save(model.state_dict(), weight_file_path)
        print("Done!")

    except Exception as e:
        print(f"モデルの学習に失敗しました: {str(e)}")
        sys.exit(1)
