import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

        return image, image


if __name__ == "__main__":
    # pathの設定
    annotation_file = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_attr_celeba.csv")
    partition_file = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_eval_partition.csv")
    img_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "img_align_celeba")

    # データセットの初期化
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(img_dir, annotation_file, partition_file, "train", transform)
    print(dataset[0])
