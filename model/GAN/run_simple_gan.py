import os
import sys
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from dataset.fashion_mnist_dataset import get_fashion_mnist_data
from schemas import Dataloaders, History
from simple_gan_model import Discriminator, Generator
from utils import get_device, save_model


# 学習パラメータの設定
BATCH_SIZE = 64
MAX_EPOCH = 50
GENERATOR_TRAIN_PER_EPOCH = 1
DROP_RATE = 0.5
LATENT_DIM = 100

# 学習履歴
history_discriminator = History()
history_generator = History()

# モデルを保存するパス
# 識別器
weights_file_path_discriminator = os.path.join(os.path.dirname(__file__), "weights_file", "simple_gan", "discriminator.pth")
model_file_path_discriminator = os.path.join(os.path.dirname(__file__), "model_file", "simple_gan", "discriminator.pth")

# 生成器
weights_file_path_generator = os.path.join(os.path.dirname(__file__), "weights_file", "simple_gan", "generator.pth")
model_file_path_generator = os.path.join(os.path.dirname(__file__), "model_file", "simple_gan", "generator.pth")



def select_mode() -> str:
    """
    実行モードの選択

    Returns:
        str: 実行モード (train or train_with_loading_weight or test)
    """
    try:
        # モードの標準入力
        print("-" * 25)
        print("1: train")
        print("2: train with loading weight")
        print("3: test")
        mode_str_number = input("実行モードを番号で選択してください >> ")
        print("-" * 25)

        # モードの形式の変換
        mode_dict = {
            "1": "train",
            "2": "train_with_loading_weight",
            "3": "test"
        }
        mode_str = mode_dict[mode_str_number]

        return mode_str

    except KeyError as e:
        print("モードの番号が誤っています")
        return select_mode()

    except Exception as e:
        raise e


def train(
    device: torch.device,
    dataloaders: Dataloaders,
    discriminator: nn.Module,
    generator: nn.Module,
    loss_function: Callable,
    optim_discriminator: optim,
    optim_generator: optim
) -> None:
    """
    訓練を実行する関数

    Args:
        device: デバイス
        detaloaders: Dataloadersクラスのデータ
        discriminator: 識別器
        generator: 生成器
        loss_function: 損失関数
        optim_discriminator: 識別器の最適化手法
        optim_generator: 生成器の最適化手法
    """
    try:
        for epoch in range(MAX_EPOCH):
            print(f"Epoch {epoch+1}")
            print("-" * 25)

            train_dataset_size = len(dataloaders.train.dataset)
            total_processed_train_num = 0

            for batch, (X, _) in enumerate(dataloaders.train):
                batch_size_actual = len(X)
                total_processed_train_num += batch_size_actual
                X = X.to(device)

                # 識別器のラベルの作成
                label_real = torch.ones((batch_size_actual, 1), dtype=torch.float32).to(device)
                label_fake = torch.zeros((batch_size_actual, 1), dtype=torch.float32).to(device)

                # ----- train discriminator ----- #
                # 実画像に対する識別器の損失の算出
                pred_real = discriminator.forward(X)
                loss_real = loss_function(pred_real, label_real)

                # 生成器で画像を生成
                latent_vec = torch.randn((batch_size_actual, LATENT_DIM), dtype=torch.float32).to(device)
                img_fake = generator.forward(latent_vec)

                # 偽画像に対する識別器の損失の算出
                pred_fake = discriminator.forward(img_fake.detach())
                loss_fake = loss_function(pred_fake, label_fake)

                # 識別器の重みの更新
                loss_discriminator = (loss_real + loss_fake) / 2.0
                optim_discriminator.zero_grad()
                loss_discriminator.backward()
                optim_discriminator.step()
                history_discriminator.train_loss_per_batch.append(loss_discriminator.item() / batch_size_actual)

                # ----- train generator ----- #
                for _ in range(GENERATOR_TRAIN_PER_EPOCH):
                    latent_vec = torch.randn((batch_size_actual, LATENT_DIM), dtype=torch.float32).to(device)
                    img_fake = generator.forward(latent_vec)

                    pred_fake = discriminator(img_fake)
                    loss_generator = loss_function(pred_fake, label_real)
                    optim_generator.zero_grad()
                    loss_generator.backward()
                    optim_generator.step()

                history_generator.train_loss_per_batch.append(loss_generator.item() / batch_size_actual)

                # ----- visualize progress ----- #
                if batch % 100 == 0:
                    print(f"[{total_processed_train_num:>5d}/{train_dataset_size:>5d}]")
                    print(f"discriminator loss: {(history_discriminator.train_loss_per_batch[-1]):>7f}")
                    print(f"generator loss: {(history_generator.train_loss_per_batch[-1]):>7f} \n")
            
            # epochごとの学習履歴の保存
            history_discriminator.train_loss_per_epoch.append(history_discriminator.train_loss_per_batch[-1])
            history_generator.train_loss_per_epoch.append(history_generator.train_loss_per_batch[-1])

            # 学習結果の表示
            print("-" * 25)
            print("Finish training in this epoch")
            print(f"discriminator loss: {(history_discriminator.train_loss_per_batch[-1]):>7f}")
            print(f"generator loss: {(history_generator.train_loss_per_batch[-1]):>7f}")
            print("-" * 25)
        
        print("Finish training!")
        print("saving model...")
        save_model(discriminator, weights_file_path_discriminator, model_file_path_discriminator)
        save_model(generator, weights_file_path_generator, model_file_path_generator)
        
    except KeyboardInterrupt:
        print("ユーザによる割り込みが発生しました")

        is_save_str = input("モデルを保存しますか (y/n) >> ")
        is_save = False if is_save_str == "n" else True
        if is_save:
            print("saving model...")
            save_model(discriminator, weights_file_path_discriminator, model_file_path_discriminator)
            save_model(generator, weights_file_path_generator, model_file_path_generator)
    
    except Exception as e:
        print(f"学習中にエラーが発生しました: {str(e)}")

        is_save_str = input("モデルを保存しますか (y/n) >> ")
        is_save = False if is_save_str == "n" else True
        if is_save:
            print("saving model...")
            save_model(discriminator, weights_file_path_discriminator, model_file_path_discriminator)
            save_model(generator, weights_file_path_generator, model_file_path_generator)
        
    finally:
        visualize_history(history_discriminator, history_generator)
        generate_img(device, generator)


def test(
    device: torch.device,
    dataloaders: Dataloaders,
    discriminator: nn.Module,
    generator: nn.Module,
    loss_function: Callable,
) -> None:
    """
    テストを実行する関数

    Args:
        device: デバイス
        detaloaders: Dataloadersクラスのデータ
        discriminator: 識別器
        generator: 生成器
        loss_function: 損失関数
    
    Raise:
        Exception: エラーが発生した場合
    """
    try:
        test_dataset_size = len(dataloaders.test.dataset)

        total_loss_discriminator = 0.0
        total_loss_generator = 0.0

        discriminator.eval()
        generator.eval()

        with torch.no_grad():
            with tqdm(total=test_dataset_size, desc="Test") as pbar:
                for (X, _) in dataloaders.test:
                    batch_size_actual = len(X)
                    X = X.to(device)

                    # 識別器のラベルの作成
                    label_real = torch.ones((batch_size_actual, 1), dtype=torch.float32).to(device)
                    label_fake = torch.zeros((batch_size_actual, 1), dtype=torch.float32).to(device)

                    # 画像の生成
                    latent_vec = torch.randn((batch_size_actual, LATENT_DIM), dtype=torch.float32).to(device)
                    img_fake = generator.forward(latent_vec)

                    # 識別器の損失
                    # 実画像
                    pred_real = discriminator.forward(X)
                    loss_real = loss_function(pred_real, label_real)

                    # 偽画像
                    pred_fake = discriminator.forward(img_fake)
                    loss_fake = loss_function(pred_fake, label_fake)

                    loss_discriminator = (loss_real + loss_fake) / 2.0
                    total_loss_discriminator += loss_discriminator.item()

                    # 生成器の損失
                    loss_generator = loss_function(pred_fake, label_real)
                    total_loss_generator += loss_generator.item()

                    pbar.update(batch_size_actual)
        
        # ----- visualize test result --- #
        print("Finish test!")
        print(f"discriminator loss: {(total_loss_discriminator / test_dataset_size):>7f}")
        print(f"generator loss: {(total_loss_generator / test_dataset_size):>7f}")

        generate_img(device, generator)
        
    except Exception as e:
        raise Exception(f"テストに失敗しました: {str(e)}")
    


def visualize_history(history_discriminator: History, history_generator: History) -> None:
    """
    学習履歴の可視化

    Args:
        history_discriminator: 識別器の学習履歴
        history_generator: 生成器の学習履歴
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 左上 (discriminator_loss_per_batch)
    axs[0, 0].plot([i + 1 for i in range(len(history_discriminator.train_loss_per_batch))], history_discriminator.train_loss_per_batch)
    axs[0, 0].set_title("Discriminator Loss per Batch")
    axs[0, 0].set_xlabel("batch")
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].grid(True)

    # 右上 (discriminator_loss_per_epoch)
    axs[0, 1].plot([i + 1 for i in range(len(history_discriminator.train_loss_per_epoch))], history_discriminator.train_loss_per_epoch)
    axs[0, 1].set_title("Discriminator Loss per Epoch")
    axs[0, 1].set_xlabel("epoch")
    axs[0, 1].set_ylabel("loss")
    axs[0, 1].grid(True)

    # 左下 (generator_loss_per_batch)
    axs[1, 0].plot([i + 1 for i in range(len(history_generator.train_loss_per_batch))], history_generator.train_loss_per_batch)
    axs[1, 0].set_title("Generator Loss per Batch")
    axs[1, 0].set_xlabel("batch")
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].grid(True)

    # 右下 (generator_loss_per_epoch)
    axs[1, 1].plot([i + 1 for i in range(len(history_generator.train_loss_per_epoch))], history_generator.train_loss_per_epoch)
    axs[1, 1].set_title("Generator Loss per Epoch")
    axs[1, 1].set_xlabel("epoch")
    axs[1, 1].set_ylabel("loss")
    axs[1, 1].grid(True)

    # タイトルやラベルが重なるのを防ぐ
    plt.tight_layout()

    plt.show()


def generate_img(device: torch.device, generator: nn.Module) -> None:
    """
    ランダムに画像を生成する

    Args:
        devoce: デバイス
        generator: 生成器
    """
    # 画像の生成
    img_num = 9
    latent_vec = torch.randn((img_num, LATENT_DIM), dtype=torch.float32).to(device)

    generator.eval()
    with torch.no_grad():
        img_generated = generator.forward(latent_vec).detach().cpu().numpy()

    # 画像の表示
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    axs = axs.ravel()

    for i in range(img_num):
        img = img_generated[i].squeeze()
        
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')

    plt.suptitle('Generated Fashion MNIST Images')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    """
    全体の制御を行う関数
    """
    # データの設定
    try:
        dataloaders = get_fashion_mnist_data(BATCH_SIZE)

    except Exception as e:
        raise Exception(f"データの設定でエラーが発生しました: {str(e)}")

    # モデルの設定
    try:
        device = get_device()

        discriminator = Discriminator(DROP_RATE)
        generator = Generator(LATENT_DIM)

        discriminator.to(device)
        generator.to(device)

    except Exception as e:
        raise Exception(f"モデルの設定でエラーが発生しました: {str(e)}")

    # 損失関数の設定
    try:
        loss_function = F.binary_cross_entropy_with_logits

    except Exception as e:
        raise Exception(f"損失関数の設定でエラーが発生しました: {str(e)}")

    # 最適化手法の設定
    try:
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.0001)
        optim_generator = optim.Adam(generator.parameters(), lr=0.0001)

    except Exception as e:
        raise Exception(f"最適化手法の設定でエラーが発生しました: {str(e)}")
    
    # モードの選択
    try:
        mode_str = select_mode()

    except Exception as e:
        raise Exception(f"モードの選択でエラーが発生しました: {str(e)}")
    
    # モードの実行
    try:
        if mode_str == "train":
            train(
                device,
                dataloaders,
                discriminator,
                generator,
                loss_function,
                optim_discriminator,
                optim_generator
            )
        
        elif mode_str == "train_with_loading_weight":
            discriminator.load_state_dict(torch.load(weights_file_path_discriminator))
            generator.load_state_dict(torch.load(weights_file_path_generator))
            train(
                device,
                dataloaders,
                discriminator,
                generator,
                loss_function,
                optim_discriminator,
                optim_generator
            )

        elif mode_str == "test":
            discriminator = torch.load(model_file_path_discriminator, weights_only=False)
            generator = torch.load(model_file_path_generator, weights_only=False)
            test(
                device,
                dataloaders,
                discriminator,
                generator,
                loss_function
            )
    
    except Exception as e:
        raise e
    

if __name__ == "__main__":
    main()
