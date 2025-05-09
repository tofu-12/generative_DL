import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from base_client.model_run_client import ModelRunClient
from schemas import VAEHistory


class VAERunClient(ModelRunClient):
    def __init__(self):
        """
        VAEを実行するクライエントのインスタンスの初期化
        """
        super().__init__()
    
    def set_model(self, model: torch.nn.Module, latent_dim: int):
        """
        モデルの設定

        Args:
            model: モデル
        
        Raise:
            Exception: 任意のエラー
        """
        try:
            # デバイスの設定
            self.device = self._get_device()

            # モデルの設定
            self.model = model(latent_dim).to(self.device)
        
        except Exception as e:
            raise Exception(f"モデルの設定に失敗しました:\n{str(e)}")
        
    
    def _training(self, batch_size: int, epoch: int, weights_file_path: str, model_file_path: str) -> None:
        """
        訓練を実行

        Args:
            batch_size: バッチサイズ
            epoch: エポック数
            weights_file_path: パラメータファイルパス
            model_file_path: モデルファイルパス
        
        Raise:
            KeyboardInterrupt: Control+Cが押された場合
            Exception: その他のエラーが発生した場合
        """
        self.history = VAEHistory()
        try:
            for t in range(epoch):
                print(f"Epoch {t+1}\n-------------------------------")
                self._train_loop(batch_size)
                self._val_loop()
            print("Done!")

            # モデルの保存
            self._save_weights(weights_file_path)
            self._save_model(model_file_path)
    
        except KeyboardInterrupt as e:
            print("\nユーザーによる割り込みがありました")
            self._save_with_checking(weights_file_path, model_file_path)
            raise Exception("ユーザーによる割り込みがありました")

        except Exception as e:
            print(f"モデルの学習の途中でエラーが発生しました: {str(e)}")
            self._save_with_checking(weights_file_path, model_file_path)
            raise Exception(f"モデルの学習の途中でエラーが発生しました: {str(e)}")

            
    def _train_loop(self, batch_size: int) -> None:
        """
        訓練ループ

        Args:
            batch_size: バッチサイズ
        
        Raise:
            KeyboardInterrupt: Control+Cが押された場合
            Exception: その他のエラーが発生した場合
        """
        try:
            size = len(self.dataloaders.train.dataset)
            self.model.train()

            total_epoch_loss_sum = 0
            total_processed_samples = 0

            for batch, (X, _) in enumerate(self.dataloaders.train):
                X = X.to(self.device)
                batch_size_actual = len(X)

                pred, _, z_mean, z_log_var = self.model(X)

                loss = self.loss_function(pred, X, z_mean, z_log_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_epoch_loss_sum += loss.item()
                total_processed_samples += batch_size_actual

                current = (batch + 1) * batch_size

                if batch % 10 == 0:
                    print(f"loss: {(loss.item()/batch_size_actual):>7f}  [{current:>5d}/{size:>5d}]")

                self.history.train_loss_per_batch.append(loss.item() / batch_size_actual)

            self.history.train_loss_per_epoch.append(loss.item() / batch_size_actual)
        
        except KeyboardInterrupt as e:
            raise e

        except Exception as e:
            raise Exception(f"学習ループでエラーが発生しました: {str(e)}")


    def _val_loop(self, sample_size: int=int(1e4)) -> None:
        """
        検証ループ

        Args:
            sample_size: 検証データの個数

        Raise:
            Exception: 任意のエラーが発生した場合
        """
        try:
            self.model.eval()

            processed_samples_num = 0
            total_val_loss_sum = 0
            all_z_batches = []
            all_y_batches = []

            with torch.no_grad():
                with tqdm(total=sample_size, desc="Validating") as pbar:
                    for X, y in self.dataloaders.val:
                        # sample_sizeとの調整
                        current_batch_size = X.size(0)
                        samples_to_process_in_batch = min(current_batch_size, sample_size - processed_samples_num)

                        # 使用する部分だけを選択
                        X_processed = X[:samples_to_process_in_batch].to(self.device)
                        y_processed = y[:samples_to_process_in_batch]

                        pred, z, z_mean, z_log_var = self.model(X_processed)
                        batch_loss = self.loss_function(pred, X_processed, z_mean, z_log_var).item()
                        total_val_loss_sum += batch_loss

                        z_numpy = z.to('cpu').detach().numpy()
                        all_z_batches.append(z_numpy)

                        y_numpy = y_processed.to('cpu').detach().numpy()
                        all_y_batches.append(y_numpy)

                        processed_samples_num += samples_to_process_in_batch
                        pbar.update(samples_to_process_in_batch)

                        if processed_samples_num >= sample_size:
                            break

            average_val_loss = total_val_loss_sum / processed_samples_num if processed_samples_num > 0 else 0

            if all_z_batches:
                val_z_concatenated = np.concatenate(all_z_batches, axis=0)
            else:
                val_z_concatenated = np.array([])

            if all_y_batches:
                val_y_concatenated = np.concatenate(all_y_batches, axis=0)
            else:
                val_y_concatenated = np.array([])

            print(f"Validation loss: {average_val_loss:>8f} \n")

            self.history.val_loss_per_epoch.append(average_val_loss)
            self.history.val_z_per_epoch.append(val_z_concatenated)
            self.history.val_z_label_per_epoch.append(val_y_concatenated)

        except Exception as e:
            raise Exception(f"検証中にエラーが発生しました: {str(e)}")
        

    def run_training(
            self,
            batch_size: int,
            epoch: int,
            weights_file_path: str,
            model_file_path: str,
            loading_weights: bool=True
    ) -> None:
        """
        モデルの学習の実行

        Args:
            batch_size: バッチサイズ
            epoch: エポック数
            weights_file_path: モデルの重みを保存・読み込みするファイルパス
            model_file_path: モデル全体を保存するファイルパス
            is_loading_weights: ファイルからパラメータ読み込んで学習を再開するかどうかを示すフラグ
        """
        try:
            # 重みのロード
            if loading_weights:
                self._load_params(weights_file_path)

            # モデルの学習
            self._training(batch_size, epoch, weights_file_path, model_file_path)
        
        except Exception as e:
            print(f"学習の実行の際にエラーが発生しました:\n{str(e)}")
    
    
    def run_test(
            self,
            model_file_path: str,
            checking_test_loss: bool=True
        ) -> None:
        """
        テストの実行

        Args:
            model_file_path: モデルファイルパス
            checking_test_loss: テストデータの損失を確認するか否か
        
        Raise:
            Exception: 任意のエラーが発生した場合
        """
        try:
            # モデルの設定
            self._load_model(model_file_path)

            # モデルのモード変更
            self.model.eval()

            # テストデータの損失の確認
            if checking_test_loss:
                total_test_loss_sum = 0

                with torch.no_grad():
                    test_loop = tqdm(self.dataloaders.test, desc='Test')

                    for X, _ in test_loop:
                        X = X.to(self.device)

                        pred, _, z_mean, z_log_var = self.model(X)

                        batch_loss = self.loss_function(pred, X, z_mean, z_log_var).item()
                        total_test_loss_sum += batch_loss

                    total_samples = len(self.dataloaders.test.dataset)
                    average_test_loss = total_test_loss_sum / total_samples if total_samples > 0 else 0
                    print(f"Test Loss: {average_test_loss:>8f} ")

            # 再構成の可視化
            with torch.no_grad():
                data_iter = iter(self.dataloaders.test)
                X, _ = next(data_iter)
                X = X.to(self.device)

                # predを取得
                pred, _, _, _ = self.model(X)

                # sigmoidに通す
                pred = F.sigmoid(pred)

                X_cpu = X.cpu().permute(0, 2, 3, 1).numpy()
                pred_cpu = pred.cpu().permute(0, 2, 3, 1).numpy()

                num_samples_to_show = min(X_cpu.shape[0], 5)

                plt.figure(figsize=(10, 4))
                for i in range(num_samples_to_show):
                    # 元画像の表示
                    ax = plt.subplot(2, num_samples_to_show, i + 1)

                    plt.imshow(X_cpu[i].squeeze(), cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    if i == 0:
                        ax.set_title('Original')

                    # 再構成画像の表示
                    ax = plt.subplot(2, num_samples_to_show, i + 1 + num_samples_to_show)
                    plt.imshow(pred_cpu[i].squeeze(), cmap="gray")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    if i == 0:
                        ax.set_title("Reconstruction")

                plt.suptitle("Original vs Reconstruction")
                plt.show()

            print("Test run complete.")
        
        except Exception as e:
            print(f"テストの際にエラーが発生しました:\n{str(e)}")
    

    def visualize_final_z(self, using_label: bool) -> None:
        """
        学習履歴（潜在空間分布）をラベルで色分けして可視化する
        """
        try:
            if not hasattr(self, 'history') or self.history is None:
                print("学習履歴がありません。モデルを訓練してから実行してください。")
                return

            history = self.history

            # 最後のepochのzの分布とラベルの取得
            if history.val_z_per_epoch and history.val_z_label_per_epoch:
                print("Plotting latent space distribution with labels for the last epoch...")
                last_epoch_z = history.val_z_per_epoch[-1]
                last_epoch_y = history.val_z_label_per_epoch[-1]

                if last_epoch_z.shape[0] != last_epoch_y.shape[0]:
                    print(f"潜在空間データとラベルのサンプル数が一致しません ({last_epoch_z.shape[0]} vs {last_epoch_y.shape[0]})。可視化をスキップします。")
                    return


                if last_epoch_z.shape[1] >= 2:
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(last_epoch_z[:, 0], last_epoch_z[:, 1], s=10, c=last_epoch_y, cmap='tab10', alpha=0.8)

                    plt.title('Latent Space Distribution by Label (Last Epoch)')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                    plt.grid(True)

                    if using_label:
                        # カラーバーを追加してラベルと色の対応を示す
                        cbar = plt.colorbar(scatter)
                        cbar.set_label('Label')

                        # 可能な場合は、カラーバーの目盛りに実際のラベル名を表示する
                        if len(np.unique(last_epoch_y)) <= 10: # クラス数が少ない場合のみ目盛りを設定
                            unique_labels = np.unique(last_epoch_y)
                            cbar.set_ticks(unique_labels + 0.5) # 目盛りの位置を調整
                            cbar.set_ticklabels(unique_labels)

                    plt.show()
                else:
                    print("Latent space dimension is less than 2, skipping latent space 2D plot.")
            else:
                print("潜在空間データまたはラベルの履歴がありません。可視化をスキップします。")


        except Exception as e:
            print(f"最後の潜在空間の可視化に失敗しました: {str(e)}")