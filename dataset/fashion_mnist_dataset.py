import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from schemas import Dataloaders


class FashionMnistDataset(Dataset):
    def __init__(self, train_data_file_path: str, test_data_file_path: str, is_train: bool, transform: Optional[callable]=None):
        """
        Fashion MNISTデータセットのインスタンスの初期化
        data: https://www.kaggle.com/datasets/zalando-research/fashionmnist

        Args:
            train_data_file: 訓練データのcsvファイルパス
            test_data_file: テストデータのcsvファイルパス
            is_train: 訓練かどうかのフラグ
            transform: 画像に適用する変換処理
        """
        try:
            self.is_train = is_train
            self.transform = transform

            # データの選択と読み込み
            if is_train:
                self.data_file_path = train_data_file_path
            else:
                self.data_file_path = test_data_file_path

            # CSVファイルを読み込む
            self.data_df = pd.read_csv(self.data_file_path, header=0)

            # データ長は読み込んだDataFrameの行数
            self.len = len(self.data_df)

        except FileNotFoundError:
             raise FileNotFoundError(f"データファイルが見つかりません: {self.data_file_path}")
        
        except Exception as e:
            raise Exception(f"データセットの初期化に失敗しました: {str(e)}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """
        指定されたインデックスのデータサンプルを取得する

        Args:
            idx: 取得するサンプルのインデックス

        Returns:
            tuple:
                image: 画像データ
                label: 整数
        """
        # インデックスがデータ範囲内にあるかチェック
        if not (0 <= idx < self.len):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.len}")

        try:
            # データ取得
            row = self.data_df.iloc[idx]

            label = int(row.iloc[0])

            image_pixels = row.iloc[1:].values.astype(np.uint8)
            image = image_pixels.reshape(28, 28)

            # 変換処理
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            raise Exception(f"アイテムの取得に失敗しました: {str(e)}")


def get_fashion_mnist_data(batch_size: int) -> Dataloaders:
    """
    データローダの作成

    Args:
        batch_size: バッチサイズ

    Returns:
        Dataloaders
    """
    # パスの設定
    train_data_file_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "fashion_mnist", "fashion-mnist_train.csv")
    test_data_file_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "fashion_mnist", "fashion-mnist_test.csv")

    # データセットの作成
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ])

        train_dataset = FashionMnistDataset(train_data_file_path, test_data_file_path, True, transform)
        test_dataset = FashionMnistDataset(train_data_file_path, test_data_file_path, False, transform)
    
    except Exception as e:
        raise Exception(f"データセットの作成に失敗しました: \n {str(e)}")

    # データローダの作成
    try:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=2)
        dataloaders = Dataloaders(train=train_dataloader, val=test_dataloader, test=test_dataloader)

        return dataloaders

    except Exception as e:
        raise Exception(f"データローダの作成に失敗しました: \n {str(e)}")

if __name__ == "__main__":
    data = get_fashion_mnist_data(batch_size=1)
    print(data.train.dataset[0])
    print("max: ", str(data.train.dataset[0][0].max()))
    print("min: ", str(data.train.dataset[0][0].min()))
