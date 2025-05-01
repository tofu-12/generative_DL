import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from model import VAE
from dataset.CelebA_dataset import CelebADataset
from utils import get_device


if __name__ == "__main__":
    try:
        device = get_device()

    except Exception as e:
        print(f"デバイスの決定に失敗しました: {str(e)}")
        sys.exit(1)

    try:
        # データセットの作成
        print("データセットを作成中...")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        img_dir_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "img_align_celeba")
        attr_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_attr_celeba.csv")
        partition_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CelebA", "list_eval_partition.csv")

        if not os.path.exists(img_dir_path):
             print(f"エラー: 画像ディレクトリが見つかりません: {img_dir_path}")
             sys.exit(1)
        if not os.path.exists(attr_path):
             print(f"エラー: 属性ファイルが見つかりません: {attr_path}")
             sys.exit(1)
        if not os.path.exists(partition_path):
             print(f"エラー: パーティションファイルが見つかりません: {partition_path}")
             sys.exit(1)

        test_set = CelebADataset(img_dir_path, attr_path, partition_path, "test", transform)
        print(f"テストセットのサイズ: {len(test_set)}")

    except Exception as e:
        print(f"データセットの作成に失敗しました: {str(e)}")
        sys.exit(1)

    try:
        print("モデルを設定中...")
        model = VAE().to(device)

        # 重みのパス
        weight_file_path = os.path.join(os.path.dirname(__file__), "model_weight.pth")

        # 重みの読み込み
        try:
            if os.path.exists(weight_file_path):
                print(f"'{weight_file_path}' から重みを読み込み中...")
                model.load_state_dict(torch.load(weight_file_path, map_location=device)) # Add map_location for safety
                print("重みの読み込みに成功しました。")
            else:
                print(f"エラー: 重みのファイルが見つかりません: {weight_file_path}")
                sys.exit(1)
        
        except Exception as e:
            print(f"重みの読み込みに失敗しました: {str(e)}")
            sys.exit(1)

        # 推論と結果の収集
        model.eval()
        check_size = 5
        original_images = []
        reconstructed_images = []

        print(f"{check_size} 枚の画像を推論中...")
        with torch.no_grad():
            for i in range(min(check_size, len(test_set))):
                X, y = test_set[i]
                X = X.unsqueeze(0).to(device)
                pred, z, z_mean, z_log_var = model(X)

                original_images.append(X.squeeze(0).cpu())
                reconstructed_images.append(pred.squeeze(0).cpu())

                print(f"--- {i+1}/{min(check_size, len(test_set))} の処理が完了 ---")

        # 可視化
        print("結果を可視化中...")
        fig, axes = plt.subplots(min(check_size, len(test_set)), 2, figsize=(8, min(check_size, len(test_set)) * 4)) # Adjust figsize as needed

        if min(check_size, len(test_set)) == 1:
             axes = axes.reshape(1, -1)

        for i in range(len(original_images)):
            orig_img = original_images[i].permute(1, 2, 0)
            recon_img = reconstructed_images[i].permute(1, 2, 0)

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(recon_img)
            axes[i, 1].set_title(f"Reconstructed {i+1}")
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"モデルの推論または可視化に失敗しました: {str(e)}")
        sys.exit(1)

    print("スクリプトが正常に完了しました。")
