import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from schemas import Dataloaders


class CelebADataset(Dataset):
    def __init__(self, img_dir: str, annotation_file: str, partition_file: str, partition_type: str, transform: Optional[callable]=None):
        """
        CelebAデータセットのインスタンスの初期化

        Args:
            img_dir: 画像ファイルが格納されているディレクトリのパス
            annotation_file: 属性情報が記述されたtxtファイルのパス
            partition_path: 分割情報 (train/val/test) が記述されたtxtファイルのパス
            partition_type: 'train', 'val', または 'test' を指定
            transform: 画像に適用する変換処理
        """
        self.img_dir = img_dir
        self.transform = transform

        # 分割情報による使用する画像のラベルの抽出
        partition_dict = {"train": 0, "val": 1, "test": 2}
        partition_list = pd.read_csv(partition_file)

        target_partition_type = partition_dict[partition_type]
        target_img_names = partition_list[partition_list["partition"] == target_partition_type]["image_id"].to_list()

        annotation_list = pd.read_csv(annotation_file)
        self.target_img_labels = annotation_list[annotation_list["image_id"].isin(target_img_names)]
            
    def __len__(self):
        return len(self.target_img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.target_img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_numpy = (self.target_img_labels.iloc[idx, 1:].to_numpy(dtype=np.int64) + 1) // 2
        label_tensor = torch.tensor(label_numpy, dtype=torch.float32)

        return image, label_tensor


def get_CelebA_data(batch_size: int) -> Dataloaders:
    """
    CelebAデータセットのデータローダを取得する関数

    Args:
        batch_size: バッチサイズ
    
    Returns:
        Dataloaders
    
    Raise:
        Exception: 取得できない際のエラー
    """
    # pathの設定
    annotation_file = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_attr_celeba.csv")
    partition_file = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_eval_partition.csv")
    img_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "img_align_celeba")

    # データセットの生成
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])

        train_dataset = CelebADataset(img_dir, annotation_file, partition_file, "train", transform)
        validation_dataset = CelebADataset(img_dir, annotation_file, partition_file, "val", transform)
        test_dataset = CelebADataset(img_dir, annotation_file, partition_file, "test", transform)
    
    except Exception as e:
        raise Exception(f"データセットの作成に失敗しました: \n {str(e)}")

    # データローダの作成
    try:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=2)
        dataloaders = Dataloaders(train=train_dataloader, val=validation_dataloader, test=test_dataloader)

        return dataloaders

    except Exception as e:
        raise Exception(f"データローダの作成に失敗しました: \n {str(e)}")

    
if __name__ == "__main__":
    data = get_CelebA_data(batch_size=1)
    print(data.train.dataset[0])
    print("max: ", str(data.train.dataset[0][0].max()))
    print("min: ", str(data.train.dataset[0][0].min()))
